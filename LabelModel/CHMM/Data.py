import sys
sys.path.append('../..')

import os
import copy
import logging
import numpy as np

from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from nltk.tokenize import word_tokenize, sent_tokenize
from tokenizations import get_alignments
from tqdm import tqdm

from seqlbtoolkit.Data import one_hot

from ALT.Args import AltConfig
from LabelModel.CHMM.Args import CHMMConfig
from Src.IO import load_data_from_json, load_data_from_pt

logger = logging.getLogger(__name__)


class MultiSrcNERDataset(torch.utils.data.Dataset):
    def __init__(self,
                 text: Optional[List[List[str]]] = None,
                 embs: Optional[List[torch.Tensor]] = None,
                 obs: Optional[List[torch.Tensor]] = None,  # batch, src, token
                 lbs: Optional[List[List[str]]] = None,
                 src: Optional[List[str]] = None
                 ):
        super().__init__()
        self._embs = embs
        self._obs = obs
        self._text = text
        self._lbs = lbs
        self._src = src

    @property
    def n_insts(self):
        return len(self._obs)

    @property
    def embs(self):
        return self._embs

    @property
    def text(self):
        return self._text

    @property
    def lbs(self):
        return self._lbs

    @property
    def obs(self):
        return self._obs

    @property
    def src(self):
        return self._src

    @obs.setter
    def obs(self, value):
        logger.warning('The value of observations has been changed')
        self._obs = value

    @embs.setter
    def embs(self, value):
        logger.warning('The value of embeddings has been changed')
        self._embs = value

    @src.setter
    def src(self, value):
        logger.warning('The sources have been changed')
        self._src = value

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._lbs is not None:
            return self._text[idx], self._embs[idx], self._obs[idx], self._lbs[idx]
        else:
            return self._text[idx], self._embs[idx], self._obs[idx]

    def load_file(self,
                  file_dir: str,
                  config: Optional[CHMMConfig] = None) -> "MultiSrcNERDataset":
        """
        Load data from disk

        Parameters
        ----------
        file_dir: the directory of the file. In JSON or PT
        config: chmm configuration; Optional to make function testing easier.

        Returns
        -------
        self (MultiSrcNERDataset)
        """
        if config:
            bert_model = config.bert_model_name_or_path
            device = config.device

        file_dir = os.path.normpath(file_dir)
        logger.info(f'Loading data from {file_dir}')

        file_loc, file_name = os.path.split(file_dir)
        if file_dir.endswith('.json'):
            sentence_list, label_list, weak_label_list = load_data_from_json(file_dir, config)
            # get embedding directory
            emb_name = f"{'.'.join(file_name.split('.')[:-1])}-emb.pt"
            emb_dir = os.path.join(file_loc, emb_name)
        # for backward compatibility
        elif file_dir.endswith('.pt'):
            sentence_list, label_list, weak_label_list = load_data_from_pt(file_dir, config)
            # get embedding directory
            emb_dir = file_dir.replace('linked', 'emb')
        else:
            logger.error(f"Unsupported data type: {file_dir}")
            raise TypeError(f"Unsupported data type: {file_dir}")

        self._text = sentence_list
        self._lbs = label_list
        self._obs = weak_label_list
        logger.info(f'Data loaded from {file_dir}.')

        logger.info(f'Searching for corresponding BERT embeddings...')
        if os.path.isfile(emb_dir):
            logger.info(f"Found embedding file: {emb_dir}. Loading to memory...")
            embs = torch.load(emb_dir)
            if isinstance(embs[0], torch.Tensor):
                self._embs = embs
            elif isinstance(embs[0], np.ndarray):
                self._embs = [torch.from_numpy(emb).to(torch.float) for emb in embs]
            else:
                logger.error("Embedding is stored in an unknown type")
                raise TypeError("Embedding is stored in an unknown type")
        else:
            logger.info(f"{emb_dir} does not exist. Building embeddings instead...")
            try:
                self.build_embs(bert_model, device, emb_dir)
            except NameError:
                logger.error("To enable BERT embedding construction, config cannot be None.")
                raise NameError("To enable BERT embedding construction, config cannot be None.")
        if config:
            self._src = copy.deepcopy(config.sources)
            config.d_emb = self._embs[0].shape[-1]
            if config.debug_mode:
                self._embs = self._embs[:100]

        # append dummy token/labels in front of the text/lbs/obs
        logger.info("Appending dummy token/labels in front of the text/lbs/obs for CHMM compatibility")
        self._text = [['[CLS]'] + txt for txt in self._text]
        self._lbs = [['O'] + lb for lb in self._lbs]
        prefix = torch.zeros([1, self._obs[0].shape[-2], self._obs[0].shape[-1]])  # shape: 1, n_src, d_obs
        prefix[:, :, 0] = 1
        self._obs = [torch.cat([prefix, inst]) for inst in self._obs]

        return self

    def build_embs(self,
                   bert_model,
                   device: Optional[torch.device] = torch.device('cpu'),
                   save_dir: Optional[str] = None) -> "MultiSrcNERDataset":
        """
        build bert embeddings

        Parameters
        ----------
        bert_model: the location/name of the bert model to use
        device: device
        save_dir: location to update/store the BERT embeddings. Leave None if do not want to save

        Returns
        -------
        self (MultiSrcNERDataset)
        """
        assert bert_model is not None, AssertionError('Please specify BERT model to build embeddings')
        logger.info(f'Building BERT embeddings with {bert_model} on {device}')
        self._embs = build_embeddings(self._text, bert_model, device)
        if save_dir:
            save_dir = os.path.normpath(save_dir)
            logger.info(f'Saving embeddings to {save_dir}...')
            embs = [emb.numpy().astype(np.float32) for emb in self.embs]
            torch.save(embs, save_dir)
        return self

    def update_obs(self,
                   obs: List[List[Union[int, str]]],
                   src_name: str,
                   config: Union[CHMMConfig, AltConfig]):
        """
        update weak labels (chmm observations)

        Parameters
        ----------
        obs: input observations (week annotations)
        src_name: source name
        config: configuration file

        Returns
        -------
        self (MultiSrcNERDataset)
        """
        if isinstance(obs[0][0], str):
            lb2ids = {lb: i for i, lb in enumerate(config.bio_label_types)}
            np_map = np.vectorize(lambda lb: lb2ids[lb])
            obs = [np_map(np.asarray(weak_lbs)).tolist() for weak_lbs in obs]

        if len(obs[0]) == len(self.text[0]):
            weak_lbs_one_hot = [one_hot(np.asarray(weak_lbs), n_class=config.n_lbs) for weak_lbs in obs]
        elif len(obs[0]) == len(self.text[0]) - 1:
            weak_lbs_one_hot = [one_hot(np.asarray([0] + weak_lbs), n_class=config.n_lbs) for weak_lbs in obs]
        else:
            logger.error("The length of the input observation does not match the dataset sentences!")
            raise ValueError("The length of the input observation does not match the dataset sentences!")

        if src_name in self._src:
            src_idx = self._src.index(src_name)
            for i in range(len(self._obs)):
                self._obs[i][:, src_idx, :] = torch.tensor(weak_lbs_one_hot[i])
        else:
            self._src.append(src_name)
            for i in range(len(self._obs)):
                self._obs[i] = torch.cat([self._obs[i], torch.tensor(weak_lbs_one_hot[i]).unsqueeze(1)], dim=1)
            # add the source into config and give a heuristic source prior
            if src_name not in config.sources:
                config.sources.append(src_name)
                config.src_priors[src_name] = {lb: (0.9, 0.9) for lb in config.entity_types}
        return self


def batch_prep(emb_list: List[torch.Tensor],
               obs_list: List[torch.Tensor],
               txt_list: Optional[List[List[str]]] = None,
               lbs_list: Optional[List[dict]] = None):
    """
    Pad the instance to the max seq max_seq_length in batch
    """
    for emb, obs, txt in zip(emb_list, obs_list, txt_list):
        assert len(obs) == len(emb) == len(txt)
    d_emb = emb_list[0].shape[-1]
    _, n_src, n_obs = obs_list[0].size()
    seq_lens = [len(obs) for obs in obs_list]
    max_seq_len = np.max(seq_lens)

    emb_batch = torch.stack([
        torch.cat([inst, torch.zeros([max_seq_len-len(inst), d_emb])], dim=-2) for inst in emb_list
    ])

    prefix = torch.zeros([1, n_src, n_obs])
    prefix[:, :, 0] = 1
    obs_batch = torch.stack([
        torch.cat([inst, prefix.repeat([max_seq_len-len(inst), 1, 1])]) for inst in obs_list
    ])
    obs_batch /= obs_batch.sum(dim=-1, keepdim=True)

    seq_lens = torch.tensor(seq_lens, dtype=torch.long)

    # we don't need to append the length of txt_list and lbs_list
    return emb_batch, obs_batch, seq_lens, txt_list, lbs_list


def collate_fn(insts):
    """
    Principle used to construct dataloader

    :param insts: original instances
    :return: padded instances
    """
    all_insts = list(zip(*insts))
    if len(all_insts) == 4:
        txt, embs, obs, lbs = all_insts
        batch = batch_prep(emb_list=embs, obs_list=obs, txt_list=txt, lbs_list=lbs)
    elif len(all_insts) == 3:
        txt, embs, obs = all_insts
        batch = batch_prep(emb_list=embs, obs_list=obs, txt_list=txt)
    else:
        logger.error('Unsupported number of instances')
        raise ValueError('Unsupported number of instances')
    return batch


def build_bert_emb(sents: List[str],
                   tokenizer,
                   model,
                   device: str):
    bert_embs = list()
    for i, sent in enumerate(tqdm(sents)):

        joint_sent = ' '.join(sent)
        bert_tokens = tokenizer.tokenize(joint_sent)

        input_ids = torch.tensor([tokenizer.encode(joint_sent, add_special_tokens=True)], device=device)
        # calculate BERT last layer embeddings
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0].squeeze(0).to('cpu')
            trunc_hidden_states = last_hidden_states[1:-1, :]

        ori2bert, bert2ori = get_alignments(sent, bert_tokens)

        emb_list = list()
        for idx in ori2bert:
            emb = trunc_hidden_states[idx, :]
            emb_list.append(emb.mean(dim=0))

        # It does not matter whether to add the embedding of [CLS]
        # since that embedding is not used in the training
        emb_list = [last_hidden_states[0, :]] + emb_list
        bert_emb = torch.stack(emb_list)
        bert_embs.append(bert_emb.cpu().detach())
    return bert_embs


# noinspection PyTypeChecker
def build_embeddings(src_sents, bert_model, device):

    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    model = AutoModel.from_pretrained(bert_model).to(device)

    standarized_sents = list()
    o2n_map = list()
    n = 0

    # update input sentences so that every sentence has BERT length < 510
    logger.info('Checking lengths. Paragraphs longer than 510 tokens will be broken down.')
    for i, sents in enumerate(src_sents):
        sent_str = ' '.join(sents)
        len_bert_tokens = len(tokenizer.tokenize(sent_str))

        # Deal with sentences that are longer than 512 BERT tokens
        if len_bert_tokens >= 510:
            sents_list = [sents]
            bert_length_list = [len(tokenizer.tokenize(' '.join(t))) for t in sents_list]
            while (np.asarray(bert_length_list) >= 510).any():
                splitted_sents_list = list()
                for tokens, bert_len in zip(sents_list, bert_length_list):

                    if bert_len < 510:
                        splitted_sents_list.append(tokens)
                        continue

                    sent_str = ' '.join(tokens)
                    splitted_sents = sent_tokenize(sent_str)

                    sent_lens = list()
                    for st in splitted_sents:
                        sent_lens.append(len(word_tokenize(st)))
                    ends = [np.sum(sent_lens[:i]) for i in range(1, len(sent_lens) + 1)]

                    nearest_end_idx = np.argmin((np.array(ends) - len(tokens) / 2) ** 2)
                    split_1 = tokens[:ends[nearest_end_idx]]
                    split_2 = tokens[ends[nearest_end_idx]:]
                    splitted_sents_list.append(split_1)
                    splitted_sents_list.append(split_2)
                sents_list = splitted_sents_list
                bert_length_list = [len(tokenizer.tokenize(' '.join(t))) for t in sents_list]
            n_splits = len(sents_list)
            standarized_sents += sents_list

            o2n_map.append(list(range(n, n+n_splits)))
            n += n_splits

        else:
            standarized_sents.append(sents)
            o2n_map.append([n])
            n += 1
    logger.info('Start constructing embeddings...')
    embs = build_bert_emb(standarized_sents, tokenizer, model, device)

    # Combine embeddings so that the embedding lengths equal to the lengths of the original sentences
    logger.info('Combining results...')
    combined_embs = list()
    for o2n in o2n_map:
        if len(o2n) == 1:
            combined_embs.append(embs[o2n[0]])
        else:
            cat_emb = torch.cat([embs[o2n[0]], embs[o2n[1]][1:], embs[o2n[2]][1:]], dim=0)
            combined_embs.append(cat_emb)

    # The embeddings of [CLS] + original tokens
    for emb, sent in zip(combined_embs, src_sents):
        assert len(emb) == len(sent) + 1

    return combined_embs
