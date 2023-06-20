import os
import json
import copy
import logging
import functools
import numpy as np
from typing import List, Optional, Union, Tuple
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

import torch
from torch.utils.data import DataLoader

from .args import CHMMConfig

from seqlbtoolkit.training.dataset import load_data_from_json, load_data_from_pt
from seqlbtoolkit.data import entity_to_bio_labels, one_hot, probs_to_lbs
from seqlbtoolkit.embs import build_bert_token_embeddings

logger = logging.getLogger(__name__)


# noinspection PyBroadException
class CHMMBaseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 text: Optional[List[List[str]]] = None,
                 embs: Optional[List[torch.Tensor]] = None,
                 obs: Optional[List[torch.Tensor]] = None,  # batch, src, token
                 lbs: Optional[List[List[str]]] = None,
                 src: Optional[List[str]] = None,
                 ents: Optional[List[str]] = None):
        super().__init__()
        self._embs = embs
        self._obs = obs
        self._text = text
        self._lbs = lbs
        self._src = src
        self._ents = ents
        self._src_metrics = None

    @property
    def n_insts(self):
        return len(self.obs)

    @property
    def embs(self):
        return self._embs if self._embs else list()

    @property
    def text(self):
        return self._text if self._text else list()

    @property
    def lbs(self):
        return self._lbs if self._lbs else list()

    @property
    def obs(self):
        return self._obs if self._obs else list()

    @property
    def src(self):
        return self._src

    @property
    def ents(self):
        return self._ents

    @property
    def src_metrics(self):
        if self._src_metrics is None:
            self._src_metrics = self._get_src_metrics()
        return self._src_metrics

    @text.setter
    def text(self, value):
        logger.warning(f'{type(self)}: text has been changed')
        self._text = value

    @obs.setter
    def obs(self, value):
        logger.warning(f'{type(self)}: observations have been changed')
        self._obs = value

    @lbs.setter
    def lbs(self, value):
        logger.warning(f'{type(self)}: labels have been changed')
        self._lbs = value

    @embs.setter
    def embs(self, value):
        logger.warning(f'{type(self)}: embeddings have been changed')
        self._embs = value

    @src.setter
    def src(self, value):
        logger.warning(f'{type(self)}: sources have been changed')
        self._src = value

    @ents.setter
    def ents(self, value):
        logger.warning(f'{type(self)}: entity types have been changed')
        self._ents = value

    @src_metrics.setter
    def src_metrics(self, value):
        self._src_metrics = value

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._lbs is not None and len(self._lbs) > 0:
            return self._text[idx], self._embs[idx], self._obs[idx], self._lbs[idx]
        else:
            return self._text[idx], self._embs[idx], self._obs[idx]

    def __add__(self, other: "CHMMBaseDataset") -> "CHMMBaseDataset":
        assert self.src and other.src and self.src == other.src, ValueError("Sources not matched!")
        assert self.ents and other.ents and self.ents == other.ents, ValueError("Entity types not matched!")

        return CHMMBaseDataset(
            text=copy.deepcopy(self.text + other.text),
            embs=copy.deepcopy(self.embs + other.embs),
            obs=copy.deepcopy(self.obs + other.obs),
            lbs=copy.deepcopy(self.lbs + other.lbs),
            ents=copy.deepcopy(self.ents),
            src=copy.deepcopy(self.src)
        )

    def __iadd__(self, other: "CHMMBaseDataset") -> "CHMMBaseDataset":
        if self.src:
            assert other.src and self.src == other.src, ValueError("Sources do not match!")
        else:
            assert other.src, ValueError("Attribute `src` not found!")

        if self.ents:
            assert other.ents and self.ents == other.ents, ValueError("Entity types do not match!")
        else:
            assert other.ents, ValueError("Attribute `ents` not found!")

        self.text = copy.deepcopy(self.text + other.text)
        self.embs = copy.deepcopy(self.embs + other.embs)
        self.obs = copy.deepcopy(self.obs + other.obs)
        self.lbs = copy.deepcopy(self.lbs + other.lbs)
        self.ents = copy.deepcopy(other.ents)
        self.src = copy.deepcopy(other.src)
        return self

    def save(self, file_dir: str, dataset_type: str, config: CHMMConfig, force_save: Optional[bool] = False):
        """
        Save dataset for future usage

        Parameters
        ----------
        file_dir: the folder which the dataset will be stored in.
        dataset_type: decides if the dataset is training, validation or test set
        config: configuration file
        force_save: force to save the file even if a file of the same path exists.

        Returns
        -------
        None
        """
        assert dataset_type in ['train', 'valid', 'test']
        output_path = os.path.join(file_dir, f"{dataset_type}.chmmdp")
        if os.path.exists(output_path) and not force_save:
            return None

        chmm_data_dict = {
            'text': self.text,
            'lbs': self.lbs,
            'obs': self.obs,
            'src': self.src,
            'ents': self.ents,
            'embs': self.embs,
            'src_priors': config.src_priors
        }
        torch.save(chmm_data_dict, output_path)
        return None

    def load(self, file_dir: str, dataset_type: str, config: CHMMConfig):
        """
        Load saved datasets and configurations

        Parameters
        ----------
        file_dir: the folder which the dataset is stored in.
        dataset_type: decides if the dataset is training, validation or test set
        config: configuration file

        Returns
        -------
        self
        """
        assert dataset_type in ['train', 'valid', 'test']
        if os.path.isdir(file_dir):
            file_path = os.path.join(file_dir, f'{dataset_type}.chmmdp')
            assert os.path.isfile(file_path), FileNotFoundError(f"{file_path} does not exist!")
        else:
            raise FileNotFoundError(f"{file_dir} does not exist!")

        chmm_data_dict = torch.load(file_path)
        for attr, value in chmm_data_dict.items():
            if attr == 'src_priors':
                continue
            try:
                setattr(self, f'_{attr}', value)
            except AttributeError as err:
                logger.exception(f"Failed to set attribute {attr}: {err}")
                raise err

        config.sources = copy.deepcopy(self.src)
        config.entity_types = copy.deepcopy(self.ents)
        config.bio_label_types = entity_to_bio_labels(self.ents)
        config.src_priors = chmm_data_dict['src_priors']
        config.d_emb = self._embs[0].shape[-1]
        return self

    def load_file(self,
                  file_path: str,
                  config: Optional[CHMMConfig] = None) \
            -> Union["CHMMBaseDataset", Tuple["CHMMBaseDataset", "CHMMConfig"]]:
        """
        Load data from disk

        Parameters
        ----------
        file_path: the directory of the file. In JSON or PT
        config: chmm configuration; Optional to make function testing easier.

        Returns
        -------
        self (MultiSrcNERDataset)
        """
        if config is None:
            has_config_input = False
            config = CHMMConfig()
        else:
            has_config_input = True

        bert_model = getattr(config, "bert_model_name_or_path", 'bert-base-uncased')
        device = getattr(config, "device", torch.device('cpu'))

        file_path = os.path.normpath(file_path)
        logger.info(f'Loading data from {file_path}')

        file_dir, file_name = os.path.split(file_path)
        if file_path.endswith('.json'):
            sentence_list, label_list, weak_label_list = load_data_from_json(file_path, config)
            # get embedding directory
            emb_name = f"{'.'.join(file_name.split('.')[:-1])}-emb.pt"
            emb_dir = os.path.join(file_dir, emb_name)
        # for backward compatibility
        elif file_path.endswith('.pt'):
            sentence_list, label_list, weak_label_list = load_data_from_pt(file_path, config)
            # get embedding directory
            emb_dir = file_path.replace('linked', 'emb')
        else:
            logger.error(f"Unsupported data type: {file_path}")
            raise TypeError(f"Unsupported data type: {file_path}")

        if getattr(config, 'load_src_metrics', False):
            try:
                metrics = load_src_metrics(os.path.join(file_dir, 'src_metrics.json'))
                metrics = {k: metrics[k] for k in config.sources}
                self.src_metrics = metrics
            except Exception as e:
                logger.exception(f"Failed to load source metrics: {e}.")

        self._text = sentence_list
        self._lbs = label_list
        self._obs = weak_label_list
        logger.info(f'Data loaded from {file_path}.')

        logger.info(f'Searching for corresponding BERT embeddings...')
        if os.path.isfile(emb_dir):
            logger.info(f"Found embedding file: {emb_dir}. Loading to memory...")
            embs = torch.load(emb_dir)
            if isinstance(embs[0], torch.Tensor):
                self._embs = embs
            elif isinstance(embs[0], np.ndarray):
                self._embs = [torch.from_numpy(emb).to(torch.float) for emb in embs]
            else:
                logger.error(f"Unknown embedding type: {type(embs[0])}")
                raise RuntimeError
        else:
            logger.info(f"{emb_dir} does not exist. Building embeddings instead...")

            if not has_config_input:
                logger.warning(f"No configuration found. Using default bert model `bert-base_model-uncased` "
                               f"and default device `cpu`.")
            self.build_embs(bert_model, device, emb_dir)

        self._src = copy.deepcopy(config.sources)
        self._ents = config.entity_types
        config.d_emb = self._embs[0].shape[-1]
        if getattr(config, 'debug_mode', False):
            self._embs = self._embs[:100]

        # append dummy token/labels in front of the text/lbs/obs
        logger.info("Appending dummy token/labels in front of the text/lbs/obs for CHMM compatibility")
        self._text = [['[CLS]'] + txt for txt in self._text]
        self._lbs = [['O'] + lb for lb in self._lbs]
        prefix = torch.zeros([1, self._obs[0].shape[-2], self._obs[0].shape[-1]])  # shape: 1, n_src, d_obs
        prefix[:, :, 0] = 1
        self._obs = [torch.cat([prefix, inst]) for inst in self._obs]

        if has_config_input:
            return self
        else:
            return self, config

    def build_embs(self,
                   bert_model,
                   device: Optional[torch.device] = torch.device('cpu'),
                   save_dir: Optional[str] = None) -> "CHMMBaseDataset":
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
        self._embs = build_bert_token_embeddings(
            self._text, bert_model, bert_model, device=device, prepend_cls_embs=True
        )
        if save_dir:
            save_dir = os.path.normpath(save_dir)
            logger.info(f'Saving embeddings to {save_dir}...')
            embs = [emb.numpy().astype(np.float32) for emb in self.embs]
            torch.save(embs, save_dir)
        return self

    def update_obs(self,
                   obs: List[List[Union[int, str]]],
                   src_name: str,
                   config: CHMMConfig):
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
                config.src_priors[src_name] = {ent: (0.7, 0.7) for ent in config.entity_types}

            # remove the cached property
            try:
                delattr(self, "src_metrics")
            except Exception:
                pass

        return self

    def remove_src(self,
                   src_name: str,
                   config: CHMMConfig):
        """
        remove a source and its observations from the dataset

        Parameters
        ----------
        src_name: source name
        config: configuration file

        Returns
        -------
        self (MultiSrcNERDataset)
        """

        if src_name in config.sources:
            config.sources.remove(src_name)
        if src_name in config.src_priors.keys():
            config.src_priors.pop(src_name, None)

        if src_name not in self._src:
            logger.warning(f"Labeling function {src_name} is not presented in dataset. Nothing is changed!")
            return self

        # remove source name
        src_idx = self._src.index(src_name)
        other_idx = list(range(len(self.src)))
        other_idx.remove(src_idx)

        # remove the corresponding observation
        self._src.remove(src_name)
        for i in range(len(self._obs)):
            self._obs[i] = self._obs[i][:, other_idx, :]

        # remove the cached property
        try:
            delattr(self, "src_metrics")
        except Exception:
            pass

        return self

    @functools.cache
    def _get_src_metrics(self):
        src_record_list = [list() for _ in range(len(self.src))]
        for obs in self._obs:
            src_lbs_list = probs_to_lbs(obs, entity_to_bio_labels(self.ents)).T.tolist()
            for src_lbs, src_record in zip(src_lbs_list, src_record_list):
                src_record.append(src_lbs)
        metric_dict = dict()
        for src, record in zip(self.src, src_record_list):
            report = classification_report(
                self.lbs, record, output_dict=True, mode='strict', zero_division=0, scheme=IOB2
            )
            report_dict = dict()
            for ent in self.ents:
                report_dict[ent] = {
                    'precision': report[ent]['precision'],
                    'recall': report[ent]['recall'],
                    'f1': report[ent]['f1-score']
                }
                report_dict['micro-avg'] = {
                    'precision': report['micro avg']['precision'],
                    'recall': report['micro avg']['recall'],
                    'f1': report['micro avg']['f1-score']
                }
            metric_dict[src] = report_dict
        return metric_dict


def batch_prep(emb_list: List[torch.Tensor],
               obs_list: List[torch.Tensor],
               txt_list: Optional[List[List[str]]] = None,
               lbs_list: Optional[List[dict]] = None):
    """
    Pad the instance to the max seq max_seq_length in batch

    All input should already have the dummy element appended to the beginning of the sequence
    """
    for emb, obs, txt, lbs in zip(emb_list, obs_list, txt_list, lbs_list):
        assert len(obs) == len(emb) == len(txt) == len(lbs)
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


def load_src_metrics(file_path: str):
    """
    Load the file that contains the performance of each source

    Parameters
    ----------
    file_path: file path

    Returns
    -------
    source performance: dict[dict]
    """
    with open(file_path, 'r', encoding='UTF-8') as f:
        metrics = json.load(f)
    return metrics
