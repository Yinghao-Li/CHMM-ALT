import logging
from typing import Optional
from dataclasses import dataclass, field

from src.chmm.args import CHMMArguments, CHMMConfig
from src.bert.args import BertArguments, BertConfig

logger = logging.getLogger(__name__)


@dataclass
class AltArguments(CHMMArguments, BertArguments):
    num_phase2_loop: Optional[int] = field(
        default=10, metadata={'help': 'number of loops in alternate training phase II'}
    )
    num_phase2_em_train_epochs: Optional[int] = field(
        default=20, metadata={'help': "number of training epochs for phase II's BERT"}
    )
    pass_soft_labels: Optional[bool] = field(
        default=False, metadata={'help': "Pass soft labels from label models to end models if possible"}
    )


class AltConfig(CHMMConfig, BertConfig, AltArguments):
    pass
