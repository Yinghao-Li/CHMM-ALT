import torch
import logging
from typing import Optional, List
from dataclasses import dataclass, field
from transformers.file_utils import cached_property, torch_required

logger = logging.getLogger(__name__)


@dataclass
class CHMMArguments:
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
    no_neural_emiss: Optional[bool] = field(
        default=False, metadata={'help': 'Not use neural networks to predict emission probabilities.'}
    )
    emiss_nn_weight: Optional[float] = field(
        default=1.0, metadata={'help': 'the weight of neural part in the emission matrix'}
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
class CHMMConfig(CHMMArguments):
    """
    Conditional HMM training configuration
    """
    sources: Optional[List[str]] = None
    entity_types: Optional[List[str]] = None
    bio_label_types: Optional[List[str]] = None
    src_priors: Optional[dict] = None
    d_emb: Optional[int] = None

    @cached_property
    def _get_hidden_dim(self) -> "int":
        """
        Returns the HMM hidden dimension, AKA, the number of bio labels
        """
        return len(self.bio_label_types) if self.bio_label_types is not None else 0

    @property
    def d_hidden(self) -> "int":
        return self._get_hidden_dim

    @property
    def d_obs(self) -> "int":
        return self._get_hidden_dim

    @property
    def n_lbs(self):
        """
        Returns
        -------
        The number of BIO labels
        """
        return len(self.bio_label_types) if self.bio_label_types is not None else 0

    @property
    def n_src(self) -> "int":
        """
        Returns
        -------
        The number of sources
        """
        return len(self.sources) if self.sources is not None else 0

    def from_args(self, args: CHMMArguments) -> "CHMMConfig":
        """
        Initialize configuration from arguments

        Parameters
        ----------
        args: arguments (parent class)

        Returns
        -------
        self (type: CHMMConfig)
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
