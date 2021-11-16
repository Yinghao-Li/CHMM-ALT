import sys
sys.path.append('../..')

import torch
import logging
from typing import Optional
from dataclasses import dataclass, field
from transformers.file_utils import cached_property, torch_required

from seqlbtoolkit.bert_ner.config import BertBaseConfig

logger = logging.getLogger(__name__)


@dataclass
class BertArguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """
    train_file: Optional[str] = field(
        default='', metadata={'help': 'training data name'}
    )
    valid_file: Optional[str] = field(
        default='', metadata={'help': 'development data name'}
    )
    test_file: Optional[str] = field(
        default='', metadata={'help': 'test data name'}
    )
    output_dir: Optional[str] = field(
        default='.',
        metadata={"help": "The output folder where the model predictions and checkpoints will be written."},
    )
    num_em_train_epochs: Optional[int] = field(
        default=15, metadata={'help': 'number of denoising model training epochs'}
    )
    num_em_valid_tolerance: Optional[int] = field(
        default=10, metadata={"help": "How many tolerance epochs before quiting training"}
    )
    learning_rate: Optional[float] = field(
        default=5e-5, metadata={'help': 'learning rate'}
    )
    warmup_ratio: Optional[int] = field(
        default=0.2, metadata={'help': 'ratio of warmup steps for learning rate scheduler'}
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear", metadata={"help": "Default as `linear`. See the documentation of "
                                            "`transformers.SchedulerType` for all possible values"},
    )
    weight_decay: Optional[float] = field(
        default=0.01, metadata={'help': 'strength of weight decay'}
    )
    em_batch_size: Optional[int] = field(
        default=128, metadata={'help': 'denoising model training batch size'}
    )
    max_length: Optional[int] = field(
        default=512, metadata={'help': 'maximum sequence length'}
    )
    bert_model_name_or_path: Optional[str] = field(
        default='', metadata={"help": "Path to pretrained BERT model or model identifier from huggingface.co/models; "
                                      "Used to construct BERT embeddings if not exist"}
    )
    no_cuda: Optional[bool] = field(default=False, metadata={"help": "Disable CUDA even when it is available"})
    log_dir: Optional[str] = field(
        default=None,
        metadata={"help": "the directory of the log file. Set to '' to disable logging"}
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    batch_gradient_descent: Optional[bool] = field(
        default=False, metadata={'help': 'whether use batch instead of mini-batch for gradient descent.'}
    )
    debug_mode: Optional[bool] = field(
        default=False, metadata={"help": "Debugging mode with fewer training data"}
    )

    # The following three functions are copied from transformers.training_args
    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        else:
            device = torch.device("cuda")
            self._n_gpu = 1

        return device

    @property
    @torch_required
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        return self._setup_devices

    @property
    @torch_required
    def n_gpu(self) -> "int":
        """
        The number of GPUs used by this process.

        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        # Make sure `self._n_gpu` is properly setup.
        _ = self._setup_devices
        return self._n_gpu


@dataclass
class BertConfig(BertArguments, BertBaseConfig):
    pass
