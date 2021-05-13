# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """

import logging
import os
import torch
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union, Dict
from tqdm.auto import tqdm

from filelock import FileLock
from transformers import PreTrainedTokenizer

from torch import nn
from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]] = None
    weak_lb_weights: Optional[List[str]] = None


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    weak_lb_weights: Optional[np.ndarray] = None


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class TokenClassificationTask:
    def load_examples(
            self,
            data_dir,
            dataset,
            tokenizer: PreTrainedTokenizer,
            mode: Union[Split, str],
            max_seq_length: Optional[int] = None,
            weak_annos: Optional[np.ndarray] = None
    ) -> List[InputExample]:
        raise NotImplementedError

    def get_labels(self, path: str) -> List[str]:
        raise NotImplementedError

    @staticmethod
    def convert_examples_to_features(
            examples: List[InputExample],
            label_list: List[str],
            max_seq_length: int,
            tokenizer: PreTrainedTokenizer,
            cls_token_at_end=False,
            cls_token="[CLS]",
            cls_token_segment_id=1,
            sep_token="[SEP]",
            sep_token_extra=False,
            pad_on_left=False,
            pad_token=0,
            pad_token_segment_id=0,
            pad_token_label_id=-100,
            sequence_a_segment_id=0,
            mask_padding_with_zero=True,
    ) -> List[InputFeatures]:
        """Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        label_map = {label: i for i, label in enumerate(label_list)}

        logger.info("*** Constructing Dataset ***")
        features = []
        for (ex_index, example) in enumerate(tqdm(examples)):
            words = [' ' + w for w in example.words]
            no_weak_lbs = True if example.weak_lb_weights is None else False

            tokens = []
            label_ids = []
            weak_lb_weights = []
            if no_weak_lbs:
                for word, label in zip(words, example.labels):
                    word_tokens = tokenizer.tokenize(word)

                    if len(word_tokens) > 0:
                        tokens.extend(word_tokens)
                        label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
            else:
                for word, label, weak_lb in zip(words, example.labels, example.weak_lb_weights):
                    word_tokens = tokenizer.tokenize(word)

                    # bert-base-multilingual-cased sometimes output "nothing ([])
                    # when calling tokenize with just a space.
                    if len(word_tokens) > 0:
                        tokens.extend(word_tokens)
                        label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                        if isinstance(weak_lb, str):
                            weak_lb_weight = np.zeros(len(label_list), dtype=np.float32)
                            weak_lb_weight[label_map[weak_lb]] = 1
                        else:
                            weak_lb_weight = weak_lb
                        weak_lb_weights.extend(
                            [weak_lb_weight] + [np.zeros_like(weak_lb_weight)] * (len(word_tokens) - 1)
                        )

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = tokenizer.num_special_tokens_to_add()
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]
                if not no_weak_lbs:
                    weak_lb_weights = weak_lb_weights[: (max_seq_length - special_tokens_count)]

            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            if not no_weak_lbs:
                weak_lb_weights += [np.zeros_like(weak_lb_weights[0])]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [pad_token_label_id]
                if not no_weak_lbs:
                    weak_lb_weights += [np.zeros_like(weak_lb_weights[0])]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
                if not no_weak_lbs:
                    weak_lb_weights += [np.zeros_like(weak_lb_weights[0])]
            else:
                tokens = [cls_token] + tokens
                label_ids = [pad_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids
                if not no_weak_lbs:
                    weak_lb_weights = [np.zeros_like(weak_lb_weights[0])] + weak_lb_weights

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The batch_mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
                if not no_weak_lbs:
                    weak_lb_weights = ([np.zeros_like(weak_lb_weights[0])] * padding_length) + weak_lb_weights
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length
                if not no_weak_lbs:
                    weak_lb_weights += [np.zeros_like(weak_lb_weights[0])] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            if not no_weak_lbs:
                assert len(weak_lb_weights) == max_seq_length

            if "token_type_ids" not in tokenizer.model_input_names:
                segment_ids = None

            if not no_weak_lbs:
                weak_lb_weights = np.stack(weak_lb_weights)
            else:
                weak_lb_weights = None
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    token_type_ids=segment_ids,
                    label_ids=label_ids,
                    weak_lb_weights=weak_lb_weights
                )
            )
        return features


class TokenClassificationDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
            self,
            token_classification_task: TokenClassificationTask,
            data_dir: str,
            dataset: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            weak_src: Optional[str] = None,
            weak_annos: Optional[np.ndarray] = None,
            mode: Split = Split.train,
    ):

        # Load data features from cache or dataset file
        if weak_annos is None:
            weak_src = None
        src_name = weak_src if weak_src else 'true'
        cached_features_file = os.path.join(
            data_dir, "cached_{}_{}_{}_{}".format(dataset, src_name, mode.value, max_seq_length)
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                examples = token_classification_task.load_examples(
                    data_dir=data_dir,
                    dataset=dataset,
                    tokenizer=tokenizer,
                    mode=mode,
                    weak_annos=weak_annos,
                    max_seq_length=max_seq_length
                )
                self.features = token_classification_task.convert_examples_to_features(
                    examples,
                    labels,
                    max_seq_length,
                    tokenizer,
                    cls_token_at_end=bool(model_type in ["xlnet"]),
                    # xlnet has a cls token at the end
                    cls_token=tokenizer.cls_token,
                    cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                    sep_token=tokenizer.sep_token,
                    sep_token_extra=False,
                    # roberta uses an extra separator b/w pairs of sentences, cf.
                    # github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                    pad_on_left=bool(tokenizer.padding_side == "left"),
                    pad_token=tokenizer.pad_token_id,
                    pad_token_segment_id=tokenizer.pad_token_type_id,
                    pad_token_label_id=self.pad_token_label_id,
                )
                logger.info(f"Saving features into cached file {cached_features_file}")
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def data_collator(features: list) -> Dict[str, torch.Tensor]:
    """
    Very simple data collator that:
    - simply collates batches of dict-like objects
    - Performs special handling for potential keys named:
        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object
    - does not do any additional preprocessing

    i.e., Property names of the input object will be used as corresponding inputs to the model.
    See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    if not isinstance(features[0], dict):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.stack([torch.from_numpy(f[k]) for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)

    return batch
