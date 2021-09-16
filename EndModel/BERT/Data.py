import sys
sys.path.append('../..')

import os
import logging
import numpy as np
from tqdm.auto import tqdm
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    BatchEncoding
)

from seqlbtoolkit.Text import break_overlength_bert_text
from Src.IO import load_data_from_json, load_data_from_pt
from EndModel.BERT.Args import BertConfig

logger = logging.getLogger(__name__)


class BertNERDataset(torch.utils.data.Dataset):
    def __init__(self,
                 text: Optional[List[List[str]]] = None,
                 lbs: Optional[List[List[str]]] = None,
                 encoded_texts: Optional[BatchEncoding] = BatchEncoding(),
                 encoded_lbs: Optional[List[List[int]]] = None,
                 mapping_ids: Optional[List[List[int]]] = None,
                 token_masks: Optional[List[List[int]]] = None,
                 ):
        super().__init__()
        self._text = text
        self._lbs = lbs
        # splitted text so that every sentence is within maximum length when they are converted to BERT tokens
        self._encoded_texts = encoded_texts
        self._encoded_lbs = encoded_lbs if encoded_lbs is not None else list()
        # mapping from original sentences to splitted ones ([[1.1, 1.2], [2], [3]])
        self._mapping_ids = mapping_ids if mapping_ids is not None else list()
        # mask out sub-tokens and paddings
        self._token_masks = token_masks if token_masks is not None else list()

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text_: List[List[str]]):
        logger.warning("Setting text instances. Need to run `encode_text` or `encode_text_and_lbs` "
                       "to update encoded text.")
        self._text = text_

    @property
    def lbs(self):
        return self._lbs

    @lbs.setter
    def lbs(self, labels: List[List[str]]):
        assert len(self._text) == len(labels), ValueError("The number of text & labels instances does not match!")
        for txt, lbs_ in zip(self._text, labels):
            assert len(txt) == len(lbs_), ValueError("The lengths of text & labels instances does not match!")
        logger.warning("Setting label instances. Need to run `encode_text_and_lbs` to update encoded labels.")
        self._lbs = labels

    @property
    def token_masks(self):
        return np.asarray(self._token_masks)

    @property
    def encoded_lbs(self):
        return np.asarray(self._encoded_lbs)

    @property
    def mapping_ids(self):
        return self._mapping_ids

    @property
    def n_insts(self):
        return len(self._encoded_texts.input_ids)

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self._encoded_texts.items() if key != 'offset_mapping'}
        item['token_masks'] = torch.tensor(self._token_masks[idx])
        if self._encoded_lbs:
            item['labels'] = torch.tensor(self._encoded_lbs[idx])
        return item

    def encode_text(self, config: BertConfig):
        """
        Encode tokens so that they match the BERT data format

        Parameters
        ----------
        config: configuration file

        Returns
        -------
        self (BertNERDataset)
        """
        assert self._text, ValueError("Need to specify text")
        logger.info("Encoding BERT text")

        tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name_or_path)

        sp_text = list()
        mapping_ids = list()
        for text in self._text:
            # break a sentence into several pieces if it exceeds the maximum length
            st, st_lens, st_ids = break_overlength_bert_text(text, tokenizer, config.max_length)
            sp_text += st

            if not mapping_ids:
                mapping_ids.append(st_ids)
            else:
                mapping_ids.append(st_ids + mapping_ids[-1][-1] + 1)

        self._mapping_ids = mapping_ids

        logger.info('Encoding sentences into BERT tokens')
        self._encoded_texts = tokenizer(self._text,
                                        is_split_into_words=True,
                                        return_offsets_mapping=True,
                                        padding='max_length',
                                        max_length=config.max_length,
                                        truncation=True)

        token_masks = list()
        for doc_offset in tqdm(self._encoded_texts.offset_mapping):
            arr_offset = np.array(doc_offset)

            # create an empty array of False
            masks = np.zeros(len(doc_offset), dtype=np.bool)
            masks[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = True
            token_masks.append(masks)
        self._token_masks = token_masks

        return self

    def encode_text_and_lbs(self, config: BertConfig):
        """
        Encode tokens and labels so that they match the BERT data format

        Parameters
        ----------
        config: configuration file

        Returns
        -------
        self (BertNERDataset)
        """
        logger.info("Encoding BERT text and labels")

        assert self._text and self._lbs, ValueError("Need to specify text and labels")

        tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name_or_path)

        sp_text = list()
        sp_lbs = list()
        mapping_ids = list()
        for text, lbs in zip(self._text, self._lbs):
            # break a sentence into several pieces if it exceeds the maximum length
            st, st_lens, st_ids = break_overlength_bert_text(text, tokenizer, config.max_length)
            sp_text += st

            start_idx = 0
            for st_len in st_lens:
                sp_lbs.append(lbs[start_idx: st_len])
                start_idx += st_len

            if not mapping_ids:
                mapping_ids.append(st_ids)
            else:
                mapping_ids.append(st_ids + mapping_ids[-1][-1] + 1)

        self._mapping_ids = mapping_ids

        logger.info('Encoding sentences into BERT tokens')
        self._encoded_texts = tokenizer(sp_text,
                                        is_split_into_words=True,
                                        return_offsets_mapping=True,
                                        padding='max_length',
                                        max_length=config.max_length,
                                        truncation=True)

        soft_label_flag = False
        if isinstance(sp_lbs[0], (np.ndarray, torch.Tensor)):
            soft_label_flag = True

        if soft_label_flag:
            labels = sp_lbs
        else:
            labels = [[config.lb2idx[lb] for lb in lbs] for lbs in sp_lbs]

        encoded_labels = list()
        token_masks = list()

        logger.info('Aligning labels to encoded text')
        for doc_labels, doc_offset in tqdm(zip(labels, self._encoded_texts.offset_mapping), total=len(labels)):
            # create an empty array of -100
            if soft_label_flag:
                doc_enc_labels = np.ones([len(doc_offset), config.n_lbs]) * -100
            else:
                doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
            arr_offset = np.array(doc_offset)

            # set labels whose first offset position is 0 and the second is not 0
            doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())

            # create an empty array of False
            masks = np.zeros(len(doc_offset), dtype=np.bool)
            masks[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = True
            token_masks.append(masks)
        self._encoded_lbs = encoded_labels
        self._token_masks = token_masks

        return self

    def select(self, ids: Union[List[int], np.ndarray, torch.Tensor]):
        """
        Select a subset of dataset

        Parameters
        ----------
        ids: instance indices to select

        Returns
        -------
        A BertClassificationDataset consists of selected items
        """
        if np.max(ids) >= self.n_insts:
            logger.error("Invalid indices: exceeding the dataset size!")
            raise ValueError('Invalid indices: exceeding the dataset size!')
        text_ = np.asarray(self._text, dtype=object)[ids].tolist()
        lbs_ = np.asarray(self._lbs, dtype=object)[ids].tolist() if self._lbs else None
        logger.warning("Need to run `encode_text` or `encode_text_and_lbs` on the selected subset.")
        return BertNERDataset(text_, lbs_)

    def load_file(self,
                  file_dir: str,
                  config: Optional[BertConfig] = None) -> "BertNERDataset":
        """
        Load data from disk

        Parameters
        ----------
        file_dir: the directory of the file. In JSON or PT
        config: chmm configuration; Optional to make function testing easier.

        Returns
        -------
        self (BERTNERDataset)
        """

        file_dir = os.path.normpath(file_dir)
        logger.info(f'Loading data from {file_dir}')

        if file_dir.endswith('.json'):
            sentence_list, label_list, weak_label_list = load_data_from_json(file_dir, config)
        # for backward compatibility
        elif file_dir.endswith('.pt'):
            sentence_list, label_list, weak_label_list = load_data_from_pt(file_dir, config)
        else:
            logger.error(f"Unsupported data type: {file_dir}")
            raise TypeError(f"Unsupported data type: {file_dir}")

        self._text = sentence_list
        self._lbs = label_list
        logger.info(f'Data loaded from {file_dir}.')

        return self
