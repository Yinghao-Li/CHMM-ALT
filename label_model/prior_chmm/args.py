import torch
import logging
from typing import Optional, List
from dataclasses import dataclass, field
from transformers.file_utils import cached_property, torch_required

from seqlbtoolkit.chmm.config import CHMMConfig

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
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
    trans_nn_weight: Optional[float] = field(
        default=1.0, metadata={'help': 'the weight of neural part in the transition matrix'}
    )
    emiss_nn_weight: Optional[float] = field(
        default=1.0, metadata={'help': 'the weight of neural part in the emission matrix'}
    )
    dirichlet_concentration_base: Optional[float] = field(
        default=1.0, metadata={'help': 'the basic concentration parameter (lower-bound)'}
    )
    dirichlet_concentration_max: Optional[float] = field(
        default=100.0, metadata={'help': 'the maximum concentration parameter value'}
    )
    diag_exp_t1: Optional[float] = field(
        default=2.0, metadata={'help': 'Tier 1 emission term for scaling up the emission diagonal values, '
                                       'should be >= 1.'}
    )
    diag_exp_t2: Optional[float] = field(
        default=3.0, metadata={'help': 'Tier 2 emission term for scaling up the emission diagonal values, '
                                       'should be >= 1.'}
    )
    nondiag_exp: Optional[float] = field(
        default=3.0, metadata={'help': 'Exponential term that controls how quick the e2o emission prob descents.'}
    )
    num_lm_train_epochs: Optional[int] = field(
        default=15, metadata={'help': 'number of denoising model training epochs'}
    )
    num_lm_nn_pretrain_epochs: Optional[int] = field(
        default=5, metadata={'help': 'number of denoising model pre-training epochs'}
    )
    num_lm_valid_tolerance: Optional[int] = field(
        default=10, metadata={"help": "How many tolerance epochs before quiting training"}
    )
    hmm_lr: Optional[float] = field(
        default=0.01, metadata={'help': 'learning rate of the original hidden markov model transition and emission'}
    )
    nn_lr: Optional[float] = field(
        default=0.001, metadata={'help': 'learning rate of the neural networks in CHMM'}
    )
    lm_batch_size: Optional[int] = field(
        default=128, metadata={'help': 'denoising model training batch size'}
    )
    obs_normalization: Optional[bool] = field(
        default=False, metadata={'help': 'whether normalize observations'}
    )
    use_src_prior: Optional[bool] = field(
        default=False, metadata={'help': 'whether use source priors'}
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
class PriorConfig(Arguments, CHMMConfig):

    @property
    def load_src_metrics(self):
        return True if self.use_src_prior else False
