import os
import json
from dataclasses import dataclass, field
from typing import Optional
from Src.Constants import *


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    task_type: Optional[str] = field(
        default="NER", metadata={"help": "Task type to fine tune in training (e.g. NER, POS, etc)"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    dataset_name: str = field(
        metadata={"help": "The name of the dataset."}
    )
    train_name: Optional[str] = field(
        default='', metadata={'help': 'training data name'}
    )
    dev_name: Optional[str] = field(
        default='', metadata={'help': 'development data name'}
    )
    test_name: Optional[str] = field(
        default='', metadata={'help': 'test data name'}
    )
    denoising_model: Optional[str] = field(
        default=None,
        metadata={"help": "From which weak model the scores are generated."}
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: Optional[bool] = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    self_training_start_epoch: Optional[int] = field(
        default=1, metadata={"help": "After how many epochs do we to start self-training"}
    )
    teacher_update_period: Optional[int] = field(
        default=1, metadata={"help": "How many epochs do we update the teacher model"}
    )
    bert_tolerance_epoch: int = field(
        default=60, metadata={"help": "How many tolerance epochs before quiting training"}
    )
    converse_first: bool = field(
        default=False, metadata={'help': 'converse the annotation space before (True) or after training'}
    )
    model_reinit: bool = field(
        default=False, metadata={'help': 're-initialized BERT model before each training stage/loop'}
    )
    update_embeddings: bool = field(
        default=False, metadata={'help': 'whether update embeddings during mixed training process'}
    )
    redistribute_confidence: bool = field(
        default=False, metadata={'help': 'whether to make the CHMM output sharper'}
    )
    phase2_train_epochs: int = field(
        default=20, metadata={'help': 'phase 2 fine-tuning epochs'}
    )
    true_lb_ratio: float = field(
        default=0, metadata={'help': 'What is the ratio of true label to use'}
    )


@dataclass
class CHMMArguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """
    trans_nn_weight: float = field(
        default=1.0, metadata={'help': 'the weight of neural part in the transition matrix'}
    )
    emiss_nn_weight: float = field(
        default=1.0, metadata={'help': 'the weight of neural part in the emission matrix'}
    )
    denoising_epoch: int = field(
        default=15, metadata={'help': 'number of denoising model training epochs'}
    )
    denoising_pretrain_epoch: int = field(
        default=5, metadata={'help': 'number of denoising model pre-train training epochs'}
    )
    chmm_tolerance_epoch: int = field(
        default=10, metadata={"help": "How many tolerance epochs before quiting training"}
    )
    retraining_loops: int = field(
        default=10, metadata={"help": "How many self-training (denoising-training) loops to adopt"}
    )
    hmm_lr: float = field(
        default=0.01, metadata={'help': 'learning rate of the hidden markov part'}
    )
    nn_lr: float = field(
        default=0.001, metadata={'help': 'learning rate of the neural part of the Neural HMM'}
    )
    denoising_batch_size: int = field(
        default=128, metadata={'help': 'denoising model training batch size'}
    )
    obs_normalization: bool = field(
        default=False, metadata={'help': 'whether normalize observations'}
    )
    ontonote_anno_scheme: bool = field(
        default=False, metadata={'help': 'whether to use ontonote annotation scheme'}
    )


def expend_args(training_args, chmm_args, data_args):
    if not data_args.data_dir:
        data_args.data_dir = os.path.join(ROOT_DIR, '../data', data_args.dataset_name)
    else:
        data_args.data_dir = os.path.join(ROOT_DIR, data_args.data_dir, data_args.dataset_name)
    if not data_args.train_name:
        data_args.train_name = data_args.dataset_name + '-linked-train.pt'
    if not data_args.dev_name:
        data_args.dev_name = data_args.dataset_name + '-linked-dev.pt'
    if not data_args.test_name:
        data_args.test_name = data_args.dataset_name + '-linked-test.pt'

    data_args.train_emb = data_args.train_name.replace('linked', 'emb')
    data_args.dev_emb = data_args.dev_name.replace('linked', 'emb')
    data_args.test_emb = data_args.test_name.replace('linked', 'emb')

    if data_args.dataset_name == 'Laptop':
        data_args.lbs = LAPTOP_LABELS
        data_args.bio_lbs = LAPTOP_BIO
        data_args.lbs2idx = LAPTOP_INDICES
        data_args.src = LAPTOP_SOURCE_NAMES
        data_args.src_to_keep = LAPTOP_SOURCES_TO_KEEP
        data_args.src_priors = LAPTOP_SOURCE_PRIORS
    elif data_args.dataset_name == 'NCBI':
        data_args.lbs = NCBI_LABELS
        data_args.bio_lbs = NCBI_BIO
        data_args.lbs2idx = NCBI_INDICES
        data_args.src = NCBI_SOURCE_NAMES
        data_args.src_to_keep = NCBI_SOURCES_TO_KEEP
        data_args.src_priors = NCBI_SOURCE_PRIORS
    elif data_args.dataset_name == 'BC5CDR':
        data_args.lbs = BC5CDR_LABELS
        data_args.bio_lbs = BC5CDR_BIO
        data_args.lbs2idx = BC5CDR_INDICES
        data_args.src = BC5CDR_SOURCE_NAMES
        data_args.src_to_keep = BC5CDR_SOURCES_TO_KEEP
        data_args.src_priors = BC5CDR_SOURCE_PRIORS
    elif data_args.dataset_name == 'Co03':
        data_args.lbs = CoNLL_LABELS
        data_args.bio_lbs = OntoNotes_BIO if not data_args.converse_first else CoNLL_BIO
        data_args.lbs2idx = OntoNotes_INDICES if not data_args.converse_first else CoNLL_INDICES
        data_args.src = CoNLL_SOURCE_NAMES
        data_args.src_to_keep = CoNLL_SOURCE_TO_KEEP
        data_args.src_priors = CoNLL_SOURCE_PRIORS if not data_args.converse_first else CONLL_SRC_PRIORS
        data_args.mappings = CoNLL_MAPPINGS
        data_args.prior_name = 'CoNLL03-init-stat-all.pt'
    else:
        json_name = os.path.join(data_args.data_dir, f'{data_args.dataset_name}-metadata.json')
        with open(json_name, 'r') as f:
            data = json.load(f)
        data_args.lbs = data['labels']
        if 'source-labels' not in data.keys():
            data_args.bio_lbs = ["O"] + ["%s-%s" % (bi, label) for label in data_args.lbs for bi in "BI"]
        else:
            data_args.bio_lbs = ["O"] + ["%s-%s" % (bi, label) for label in data['source-labels'] for bi in "BI"]
        data_args.lbs2idx = {label: i for i, label in enumerate(data_args.bio_lbs)}
        data_args.src = data['sources']
        data_args.src_to_keep = data['sources-to-keep'] if 'sources-to-keep' in data.keys() else data['sources']
        data_args.src_priors = data['priors'] if 'priors' in data.keys() else \
            {src: {lb: (0.8, 0.8) for lb in data_args.lbs} for src in data_args.src_to_keep}
        data_args.mappings = data['mapping'] if 'mapping' in data.keys() else None

    training_args.self_training_start_epoch = data_args.self_training_start_epoch
    training_args.teacher_update_period = data_args.teacher_update_period
    training_args.bert_tolerance_epoch = data_args.bert_tolerance_epoch
    training_args.model_reinit = data_args.model_reinit
    training_args.update_embeddings = data_args.update_embeddings
    training_args.redistribute_confidence = data_args.redistribute_confidence

    chmm_args.device = training_args.device
