import sys
sys.path.append('../..')

import logging
from typing import Optional
from dataclasses import dataclass, field

from LabelModel.CHMM.Args import CHMMArguments, CHMMConfig
from EndModel.BERT.Args import BertArguments, BertConfig

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
        default=False, metadata={'help': "whether pass soft labels from label models to end models if possible"}
    )


class AltConfig(CHMMConfig, BertConfig, AltArguments):

    def from_args(self, args: AltArguments) -> "AltConfig":
        """
        Initialize configuration from arguments

        Parameters
        ----------
        args: arguments (parent class)

        Returns
        -------
        self (type: AltConfig)
        """
        logger.info(f'Setting {type(self)} from {type(args)}.')
        arg_elements = {attr: getattr(args, attr) for attr in dir(args) if not callable(getattr(args, attr))
                        and not attr.startswith("__") and not attr.startswith("_")}
        logger.info(f'The following attributes will be changed: {arg_elements.keys()}')
        for attr, value in arg_elements.items():
            try:
                setattr(self, attr, value)
            except AttributeError:
                pass
        return self
