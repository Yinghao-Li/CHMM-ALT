import numpy as np
import torch.nn as nn
from importlib import import_module
from typing import Dict
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer
)

from Src.Bert.BertTrain import SoftTrainer
from Src.Bert.BertData import Split, TokenClassificationDataset, TokenClassificationTask, data_collator
from Src.Utils import compute_metrics, one_hot


def prepare_bert_training(model_args,
                          data_args,
                          training_args,
                          training_weak_anno,
                          eval_weak_anno,
                          test_weak_anno,
                          label_denoiser
                          ) -> (SoftTrainer, AutoTokenizer, TokenClassificationDataset):
    module = import_module("Src.Bert.BertNERTask")
    try:
        token_classification_task_clazz = getattr(module, model_args.task_type)
        token_classification_task: TokenClassificationTask = token_classification_task_clazz()
    except AttributeError:
        raise ValueError(
            f"Task {model_args.task_type} needs to be defined as a TokenClassificationTask subclass in {module}. "
            f"Available tasks classes are: {TokenClassificationTask.__subclasses__()}"
        )

    labels = token_classification_task.get_labels(data_args)
    id2label: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    label2id: Dict[str, int] = {label: i for i, label in enumerate(labels)}
    num_labels = len(labels)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        cache_dir=model_args.cache_dir,
        output_hidden_states=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    train_dataset = (
        TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_args.data_dir,
            dataset=data_args.dataset_name,
            weak_src=data_args.denoising_model,
            weak_annos=training_weak_anno,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=True,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_args.data_dir,
            dataset=data_args.dataset_name,
            weak_src=data_args.denoising_model,
            weak_annos=eval_weak_anno,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )
    test_dataset = (
        TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_args.data_dir,
            dataset=data_args.dataset_name,
            weak_src=data_args.denoising_model,
            weak_annos=test_weak_anno,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
        )
        if training_args.do_predict
        else None
    )

    if data_args.true_lb_ratio > 1e-9:
        assert data_args.true_lb_ratio <= 1
        true_label_ids = np.random.choice(len(train_dataset),
                                          round(len(train_dataset) * data_args.true_lb_ratio),
                                          replace=False)
    else:
        true_label_ids = np.array([])
    train_dataset = insert_true_labels(train_dataset, true_label_ids)

    trainer = SoftTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        compute_metrics=lambda p: compute_metrics(p, label_map=id2label),
        data_collator=data_collator,
        label_denoiser=label_denoiser
    )
    return trainer, config, true_label_ids


def reinitial_bert_trainer(model_args, training_args, config,
                           train_dataset, eval_dataset, test_dataset,
                           label_denoiser, true_label_ids):

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    train_dataset = insert_true_labels(train_dataset, true_label_ids)

    trainer = SoftTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        compute_metrics=lambda p: compute_metrics(p, label_map=config.id2label),
        data_collator=data_collator,
        label_denoiser=label_denoiser
    )

    return trainer


def insert_true_labels(dataset, true_label_ids):
    n_class = dataset[0].weak_lb_weights.shape[-1]

    for idx in true_label_ids:
        non_padding_annos = np.array(dataset[idx].label_ids) != nn.CrossEntropyLoss().ignore_index
        one_hot_true_lbs = one_hot(np.asarray(dataset[idx].label_ids)[non_padding_annos], n_class=n_class)
        dataset[idx].weak_lb_weights[non_padding_annos] = one_hot_true_lbs

    return dataset
