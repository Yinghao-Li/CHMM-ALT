import sys
sys.path.append('../..')

import logging
from typing import List, Optional
from transformers import (BatchEncoding)
from seqlbtoolkit.bert_ner.dataset import BertNERBaseDataset

logger = logging.getLogger(__name__)


class BertNERDataset(BertNERBaseDataset):
    def __init__(self,
                 text: Optional[List[List[str]]] = None,
                 lbs: Optional[List[List[str]]] = None,
                 encoded_texts: Optional[BatchEncoding] = BatchEncoding(),
                 encoded_lbs: Optional[List[List[int]]] = None,
                 mapping_ids: Optional[List[List[int]]] = None,
                 token_masks: Optional[List[List[int]]] = None,
                 ):
        super().__init__(text,
                         lbs,
                         encoded_texts,
                         encoded_lbs,
                         mapping_ids,
                         token_masks)

