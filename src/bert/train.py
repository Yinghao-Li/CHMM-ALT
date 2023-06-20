import sys
sys.path.append('../..')

import os
import logging
import numpy as np
from tqdm.auto import tqdm
from typing import Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForTokenClassification,
    AutoTokenizer,
    AdamW,
    get_scheduler,
    default_data_collator
)
from transformers.trainer_pt_utils import get_parameter_names
from transformers.modeling_outputs import TokenClassifierOutput

from seqlbtoolkit.data import probs_to_lbs, ids_to_lbs
from seqlbtoolkit.training.eval import Metric, get_ner_metrics
from .args import BertConfig
from .dataset import BertNERDataset

logger = logging.getLogger(__name__)


class BertTrainer:
    """
    Bert trainer used for training BERT for token classification (sequence labeling)
    """
    def __init__(self,
                 config: BertConfig,
                 collate_fn: Optional[PreTrainedModel] = default_data_collator,
                 model: Optional[PreTrainedModel] = None,
                 tokenzier: Optional[PreTrainedTokenizer] = None,
                 training_dataset: Optional[BertNERDataset] = None,
                 valid_dataset: Optional[BertNERDataset] = None,
                 test_dataset: Optional[BertNERDataset] = None,
                 optimizer=None,
                 lr_scheduler=None):

        self._model = model
        self._tokenizer = tokenzier
        self._config = config
        self._training_dataset = training_dataset
        self._valid_dataset = valid_dataset
        self._test_dataset = test_dataset
        self._collate_fn = collate_fn
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, x):
        logger.warning("Updating BertTrainer.config")
        self._config = x

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def training_dataset(self):
        return self._training_dataset

    @training_dataset.setter
    def training_dataset(self, dataset):
        logger.warning("BertNERDataset.training_dataset is updated!")
        self._training_dataset = dataset

    @property
    def valid_dataset(self):
        return self._valid_dataset

    @valid_dataset.setter
    def valid_dataset(self, dataset):
        logger.warning("BertNERDataset.valid_dataset is updated!")
        self._valid_dataset = dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    @test_dataset.setter
    def test_dataset(self, dataset):
        logger.warning("BertNERDataset.test_dataset is updated!")
        self._test_dataset = dataset

    def initialize_trainer(self, model=None, tokenizer=None, optimizer=None, lr_scheduler=None):
        """
        Initialize necessary components for training

        Returns
        -------
        the initialized trainer
        """
        self.set_model(model, tokenizer)
        self.set_optimizer_scheduler(optimizer, lr_scheduler)
        return self

    def set_datasets(self, training=None, valid=None, test=None):
        """
        Set bert trainer datasets

        Parameters
        ----------
        training: training dataset
        valid: validation dataset
        test: test dataset

        Returns
        -------
        self
        """
        if training:
            self.training_dataset = training
        if valid:
            self.valid_dataset = valid
        if test:
            self.test_dataset = test
        return self

    def set_model(self, model=None, tokenizer=None):
        """
        Initialize BERT model by given model/default

        Parameters
        ----------
        model: input BERT model
        tokenizer: input tokenizer

        Returns
        -------
        self
        """
        if model is not None:
            if self._model is not None:
                logger.warning(f"The original model {type(self._model)} in {type(self)} is not None. "
                               f"It will be overwritten by input!")
            self._model = model
            assert tokenizer is not None, ValueError("The tokenizer has to be assigned along with the model.")
            self._tokenizer = tokenizer
        else:
            if self._model is not None:
                logger.warning(f"The original model {type(self._model)} in {type(self)} is not None. "
                               f"It will be re-initialized by default!")
            self._model = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=self._config.bert_model_name_or_path,
                num_labels=self._config.n_lbs
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self._config.bert_model_name_or_path)
        return self

    def set_optimizer_scheduler(self, optimizer=None, lr_scheduler=None):
        """
        create optimizer and scheduler

        Parameters
        ----------
        optimizer: input optimizer
        lr_scheduler: input learning rate scheduler

        Returns
        -------
        self (BertTrainer)
        """
        if optimizer is not None:
            if self._optimizer is not None:
                logger.warning(f"The original optimizer {type(self._optimizer)} in {type(self)} is not None. "
                               f"It will be overwritten by input!")
            self._optimizer = optimizer
        else:
            if self._optimizer is not None:
                logger.warning(f"The original optimizer {type(self._optimizer)} in {type(self)} is not None."
                               f"It will be re-initialized by default!")
            # The following codes are modified from transformers.Trainer.create_optimizer_and_scheduler
            decay_parameters = get_parameter_names(self._model, [torch.nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self._model.named_parameters() if n in decay_parameters],
                    "weight_decay": self._config.weight_decay,
                },
                {
                    "params": [p for n, p in self._model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            self._optimizer = AdamW(optimizer_grouped_parameters, lr=self._config.learning_rate)

        if lr_scheduler is not None:
            if self._lr_scheduler is not None:
                logger.warning(f"The original learning rate scheduler {type(self._lr_scheduler)} in {type(self)} "
                               f"is not None. It will be overwritten by input!")
            self._lr_scheduler = lr_scheduler
        else:
            if self._lr_scheduler is not None:
                logger.warning(f"The original learning rate scheduler {type(self._lr_scheduler)} in {type(self)} "
                               f"is not None. It will be re-initialized by default!")
            # The following codes are modified from transformers.Trainer.create_optimizer_and_scheduler
            assert self._training_dataset, AttributeError("Need to define training set to initialize lr scheduler.")
            if not self._config.batch_gradient_descent:
                num_update_steps_per_epoch = int(np.ceil(len(self._training_dataset) / self._config.em_batch_size))
            else:
                num_update_steps_per_epoch = 1

            num_warmup_steps = int(np.ceil(
                num_update_steps_per_epoch * self._config.warmup_ratio * self._config.num_em_train_epochs))
            num_training_steps = int(np.ceil(num_update_steps_per_epoch * self._config.num_em_train_epochs))
            self._lr_scheduler = get_scheduler(
                self._config.lr_scheduler_type,
                self._optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        return self

    def train(self) -> Metric:
        training_dataloader = self.get_dataloader(self._training_dataset, shuffle=True)
        self._model.to(self._config.device)

        valid_results = Metric()
        best_f1 = 0
        tolerance_epoch = 0

        # ----- start training process -----
        logger.info("Start training BERT...")
        for epoch_i in range(self._config.num_em_train_epochs):
            logger.info("------")
            logger.info(f"Epoch {epoch_i + 1} of {self._config.num_em_train_epochs}")

            train_loss = self.training_step(training_dataloader, self._optimizer, self._lr_scheduler)
            logger.info("Training loss: %.4f" % train_loss)

            valid_metrics = self.evaluate(self._valid_dataset)

            logger.info("Validation results:")
            for k, v in valid_metrics.items():
                logger.info(f"\t{k}: {v:.4f}")

            # ----- save model -----
            if valid_metrics['f1'] >= best_f1:
                self.save()
                logger.info("Checkpoint Saved!\n")
                best_f1 = valid_metrics['f1']
                tolerance_epoch = 0
            else:
                tolerance_epoch += 1

            # ----- log history -----
            valid_results.append(valid_metrics)
            if tolerance_epoch > self._config.num_em_valid_tolerance:
                logger.info("Training stopped because of exceeding tolerance")
                break

        # retrieve the best state dict
        self.load()

        return valid_results

    def training_step(self, data_loader, optimizer, lr_scheduler):
        train_loss = 0
        num_samples = 0

        self._model.train()

        optimizer.zero_grad()

        for inputs in tqdm(data_loader):
            # get data
            inputs.pop('token_masks', None)
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self._config.device)
            if inputs['labels'].dtype in (torch.float, torch.float16, torch.float64):
                labels = inputs.pop('labels')
            else:
                labels = None
            batch_size = len(inputs['input_ids'])
            num_samples += batch_size

            # training step
            outputs: TokenClassifierOutput = self._model(**inputs)
            if labels is not None:
                inputs['labels'] = labels
            loss = self.compute_loss(outputs, inputs)
            loss.backward()
            # track loss
            train_loss += loss.item() * batch_size
            if not self._config.batch_gradient_descent:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        if self._config.batch_gradient_descent:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        train_loss /= num_samples

        return train_loss

    def compute_loss(self,
                     model_outputs: TokenClassifierOutput,
                     model_inputs: dict):

        if model_inputs['labels'].dtype in (torch.float, torch.float16, torch.float64):
            labels = model_inputs['labels']
            logits = model_outputs.logits
            loss = self.batch_kld_loss(
                torch.log_softmax(logits, dim=-1), labels, labels >= 0
            )
        elif model_inputs['labels'].dtype in (torch.int, torch.int8, torch.int16, torch.int64):
            loss = model_outputs.loss
        else:
            logger.error("Unknown label type!")
            raise TypeError('Unknown label type!')
        return loss

    def evaluate(self, dataset: BertNERDataset) -> Metric:
        data_loader = self.get_dataloader(dataset)
        self._model.to(self._config.device)
        self._model.eval()

        pred_lbs = list()
        pred_probs = list()
        with torch.no_grad():
            for inputs in tqdm(data_loader):
                # get data
                token_masks = np.expand_dims(inputs.pop('token_masks').numpy(), -1).repeat(self._config.n_lbs, axis=-1)
                if 'labels' in inputs:
                    inputs.pop('labels')
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self._config.device)

                outputs: TokenClassifierOutput = self._model(**inputs)
                logits = outputs.logits

                # discard paddings and predicted probabilities/labels corresponding to the sub-words
                pred_prob_batch = [probs[mask].reshape([-1, self._config.n_lbs]) for probs, mask in
                                   zip(F.softmax(logits, dim=-1).detach().to('cpu').numpy(), token_masks)]
                pred_lb_batch = [probs_to_lbs(probs=probs, label_types=self._config.bio_label_types).tolist()
                                 for probs in pred_prob_batch]
                pred_probs += pred_prob_batch
                pred_lbs += pred_lb_batch

        true_lbs = [ids_to_lbs(lb[mask], label_types=self._config.bio_label_types).tolist()
                    for mask, lb in zip(dataset.token_masks, dataset.encoded_lbs)]
        metric_values = get_ner_metrics(true_lbs, pred_lbs)

        return metric_values

    def predict(self, dataset: BertNERDataset):
        data_loader = self.get_dataloader(dataset)
        self._model.to(self._config.device)
        self._model.eval()

        pred_lbs = list()
        pred_probs = list()
        with torch.no_grad():
            for inputs in tqdm(data_loader):
                # get data
                token_masks = np.expand_dims(inputs.pop('token_masks').numpy(), -1).repeat(self._config.n_lbs, axis=-1)
                if 'labels' in inputs:
                    inputs.pop('labels')
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self._config.device)

                outputs: TokenClassifierOutput = self._model(**inputs)
                logits = outputs.logits

                # discard paddings and predicted probabilities/labels corresponding to the sub-words
                pred_prob_batch = [probs[mask].reshape([-1, self._config.n_lbs]) for probs, mask in
                                   zip(F.softmax(logits, dim=-1).detach().to('cpu').numpy(), token_masks)]
                pred_lb_batch = [probs_to_lbs(probs=probs, label_types=self._config.bio_label_types).tolist()
                                 for probs in pred_prob_batch]
                pred_probs += pred_prob_batch
                pred_lbs += pred_lb_batch

        lb_list = list()
        prob_list = list()
        # glue splitted sentences together
        for mapping_id in dataset.mapping_ids:
            lbs = list()
            probs = None
            for idx in mapping_id:
                lbs += pred_lbs[idx]
                if probs is None:
                    probs = pred_probs[idx]
                else:
                    probs = np.r_[probs, pred_probs[idx]]
            lb_list.append(lbs)
            prob_list.append(probs)

        return lb_list, prob_list

    def test(self) -> Metric:
        self._model.to(self._config.device)
        test_metrics = self.evaluate(self._test_dataset)
        return test_metrics

    def get_dataloader(self, dataset: BertNERDataset, shuffle: Optional[bool] = False):
        if dataset:
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=self._config.em_batch_size,
                collate_fn=self._collate_fn,
                shuffle=shuffle,
                drop_last=False
            )
            return data_loader
        else:
            logger.error('Dataset is not defined!')
            raise ValueError("Dataset is not defined!")

    def save(self, output_dir: Optional[str] = None,
             save_optimizer_and_scheduler: Optional[bool] = False):
        """
        Save model parameters as well as trainer parameters

        Parameters
        ----------
        output_dir: model directory
        save_optimizer_and_scheduler: whether to save optimizer and scheduler

        Returns
        -------
        None
        """
        output_dir = output_dir if output_dir is not None else self._config.output_dir
        logger.info(f"Saving model to {output_dir}")
        self._model.save_pretrained(save_directory=output_dir)
        self._tokenizer.save_pretrained(save_directory=output_dir)
        # Good practice: save your training arguments together with the trained model
        self._config.save(output_dir)
        # save trainer parameters
        if save_optimizer_and_scheduler:
            logger.info("Saving optimizer and scheduler")
            torch.save(self._optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(self._lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    def load(self, input_dir: Optional[str] = None, load_optimizer_and_scheduler: Optional[bool] = False):
        """
        Load model parameters.

        Parameters
        ----------
        input_dir: model directory
        load_optimizer_and_scheduler: whether load other trainer parameters

        Returns
        -------
        self
        """
        input_dir = input_dir if input_dir is not None else self._config.output_dir
        if self._model is not None:
            logger.warning(f"The original model {type(self._model)} in {type(self)} is not None. "
                           f"It will be overwritten by the loaded model!")
        logger.info(f"Loading model from {input_dir}")
        self._model = AutoModelForTokenClassification.from_pretrained(input_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(input_dir)
        if load_optimizer_and_scheduler:
            logger.info("Loading optimizer and scheduler")
            if self._optimizer is None:
                self.set_optimizer_scheduler()
            if os.path.isfile(os.path.join(input_dir, "optimizer.pt")):
                self._optimizer.load_state_dict(
                    torch.load(os.path.join(input_dir, "optimizer.pt"), map_location=self._config.device)
                )
            else:
                logger.warning("Optimizer file does not exist!")
            if os.path.isfile(os.path.join(input_dir, "scheduler.pt")):
                self._lr_scheduler.load_state_dict(torch.load(os.path.join(input_dir, "scheduler.pt")))
            else:
                logger.warning("Learning rate scheduler file does not exist!")
        return self

    @staticmethod
    def batch_kld_loss(batch_log_q, batch_p, batch_mask=None):
        """
        Parameters
        ----------
        batch_log_q: Q(x) in the log domain
        batch_p: P(x)
        batch_mask: select elements to compute loss Log-domain KLD loss

        Returns
        -------
        kld loss
        """
        kld = 0
        for log_q, p, mask in zip(batch_log_q, batch_p, batch_mask):
            kld += torch.sum(p[mask] * (torch.log(p[mask]) - log_q[mask]))
        kld /= len(batch_log_q)

        return kld
