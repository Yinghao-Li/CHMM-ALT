import sys
sys.path.append('../..')

import time
import logging
from tqdm.auto import tqdm

import torch
from torch.nn import functional as F

from seqlbtoolkit.base_model.eval import Metric, get_ner_metrics
from seqlbtoolkit.chmm.dataset import CHMMBaseDataset
from seqlbtoolkit.chmm.train import CHMMBaseTrainer
from .args import CHMMConfig
from .model import CHMM

logger = logging.getLogger(__name__)


class CHMMTrainer(CHMMBaseTrainer):
    def __init__(self,
                 config: CHMMConfig,
                 collate_fn,
                 training_dataset=None,
                 valid_dataset=None,
                 test_dataset=None,
                 pretrain_optimizer=None,
                 optimizer=None):

        super().__init__(
            config, collate_fn, training_dataset, valid_dataset, test_dataset, pretrain_optimizer, optimizer
        )

    @property
    def neural_module(self):
        return self._model.neural_module

    def initialize_trainer(self):
        """
        Initialize necessary components for training, returns self
        """
        CHMMBaseTrainer.initialize_trainer(self)
        return self

    def initialize_model(self):
        self._model = CHMM(
            config=self._config,
            state_prior=self._init_state_prior,
            trans_matrix=self._init_trans_mat,
            emiss_matrix=self._init_emiss_mat
        )
        return self

    def pretrain_step(self, data_loader, optimizer, trans_, emiss_):
        train_loss = 0
        num_samples = 0

        self.neural_module.train()
        if trans_ is not None:
            trans_ = trans_.to(self._config.device)
        if emiss_ is not None:
            emiss_ = emiss_.to(self._config.device)

        for i, batch in enumerate(tqdm(data_loader)):
            emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self._config.device), batch[:3])
            batch_size = len(obs_batch)
            num_samples += batch_size

            optimizer.zero_grad()
            nn_trans, nn_emiss = self.neural_module(embs=emb_batch)
            batch_size, max_seq_len, n_hidden, _ = nn_trans.size()

            loss_mask = torch.zeros([batch_size, max_seq_len], device=self._config.device)
            for n in range(batch_size):
                loss_mask[n, :seq_lens[n]] = 1
            trans_mask = loss_mask.view(batch_size, max_seq_len, 1, 1)
            trans_pred = trans_mask * nn_trans
            trans_true = trans_mask * trans_.view(1, 1, n_hidden, n_hidden).repeat(batch_size, max_seq_len, 1, 1)

            emiss_pred = emiss_true = 0
            if nn_emiss is not None:
                n_obs = nn_emiss.size(-1)
                emiss_mask = loss_mask.view(batch_size, max_seq_len, 1, 1, 1)
                emiss_pred = emiss_mask * nn_emiss
                emiss_true = emiss_mask * emiss_.view(
                    1, 1, self._config.n_src, n_hidden, n_obs
                ).repeat(batch_size, max_seq_len, 1, 1, 1)

            if trans_ is not None:
                l1 = F.mse_loss(trans_pred, trans_true)
            else:
                l1 = 0
            if emiss_ is not None and nn_emiss is not None:
                l2 = F.mse_loss(emiss_pred, emiss_true)
            else:
                l2 = 0
            loss = l1 + l2
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size
        train_loss /= num_samples
        return train_loss

    def training_step(self, data_loader, optimizer):
        train_loss = 0
        num_samples = 0

        self._model.train()

        start_time = None
        if self.config.track_training_time:
            start_time = time.time()

        for i, batch in enumerate(tqdm(data_loader)):
            # get data
            emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self._config.device), batch[:3])
            batch_size = len(obs_batch)
            num_samples += batch_size

            # training step
            optimizer.zero_grad()
            log_probs, _ = self._model(
                emb=emb_batch, obs=obs_batch, seq_lengths=seq_lens,
                normalize_observation=self._config.obs_normalization
            )

            loss = -log_probs.mean()
            loss.backward()
            optimizer.step()

            # track loss
            train_loss += loss.item() * batch_size

        if start_time is not None:
            logger.info(f"Training time for current epoch: {time.time() - start_time} s.")

        train_loss /= num_samples

        return train_loss

    def train(self) -> Metric:
        training_dataloader = self.get_dataloader(self._training_dataset, shuffle=True)

        # ----- pre-train neural module -----
        if self._config.num_lm_nn_pretrain_epochs > 0:
            logger.info(" ----- ")
            logger.info("Pre-training neural module...")
            for epoch_i in range(self._config.num_lm_nn_pretrain_epochs):
                train_loss = self.pretrain_step(
                    training_dataloader, self._pretrain_optimizer, self._init_trans_mat, self._init_emiss_mat
                )
                logger.info(f"Epoch: {epoch_i}, Loss: {train_loss}")
            logger.info("Neural module pretrained!")

        valid_results = Metric()
        best_f1 = 0
        tolerance_epoch = 0

        # ----- start training process -----
        logger.info(" ----- ")
        logger.info("Training CHMM...")
        for epoch_i in range(self._config.num_lm_train_epochs):
            logger.info("------")
            logger.info(f"Epoch {epoch_i + 1} of {self._config.num_lm_train_epochs}")

            train_loss = self.training_step(training_dataloader, self._optimizer)
            valid_metrics = self.evaluate(self._valid_dataset)

            logger.info("Training loss: %.4f" % train_loss)
            logger.info("Validation results:")
            for k, v in valid_metrics.items():
                logger.info(f"  {k}: {v:.4f}")

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
            if tolerance_epoch > self._config.num_lm_valid_tolerance:
                logger.info("Training stopped because of exceeding tolerance")
                break

        # retrieve the best state dict
        self.load()

        return valid_results

    def evaluate(self, dataset: CHMMBaseDataset) -> Metric:

        data_loader = self.get_dataloader(dataset)
        self._model.eval()

        pred_lbs = list()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader)):
                # get data
                emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self._config.device), batch[:3])

                # get prediction
                pred_lb_indices, _ = self._model.viterbi(
                    emb=emb_batch,
                    obs=obs_batch,
                    seq_lengths=seq_lens,
                    normalize_observation=self._config.obs_normalization
                )
                pred_lb_batch = [[self._config.bio_label_types[lb_index] for lb_index in label_indices]
                                 for label_indices in pred_lb_indices]
                pred_lbs += pred_lb_batch

        true_lbs = dataset.lbs
        metric_values = get_ner_metrics(true_lbs, pred_lbs)

        return metric_values

    def predict(self, dataset: CHMMBaseDataset):

        data_loader = self.get_dataloader(dataset)
        self._model.eval()

        pred_lbs = list()
        pred_probs = list()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader)):
                # get data
                emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self._config.device), batch[:3])

                # get prediction
                pred_lb_indices, pred_prob_batch = self._model.viterbi(
                    emb=emb_batch,
                    obs=obs_batch,
                    seq_lengths=seq_lens,
                    normalize_observation=self._config.obs_normalization
                )
                pred_lb_batch = [[self._config.bio_label_types[lb_index] for lb_index in label_indices]
                                 for label_indices in pred_lb_indices]

                pred_probs += pred_prob_batch
                pred_lbs += pred_lb_batch

        return pred_lbs, pred_probs

    def get_trans_and_emiss(self, dataset: CHMMBaseDataset) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        data_loader = self.get_dataloader(dataset)
        self.neural_module.eval()

        transitions = None
        emissions = None
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader)):
                emb_batch = batch[0].to(self.config.device)
                seq_lens = batch[2].to(self.config.device)

                # predict reliability scores
                trans_probs, emiss_probs = self.neural_module(embs=emb_batch)

                if transitions is None:
                    transitions = [trans[:seq_len] for trans, seq_len in zip(trans_probs.detach().cpu(), seq_lens)]
                    emissions = [emiss[:seq_len] for emiss, seq_len in zip(emiss_probs.detach().cpu(), seq_lens)]
                else:
                    transitions += [trans[:seq_len] for trans, seq_len in zip(trans_probs.detach().cpu(), seq_lens)]
                    emissions += [emiss[:seq_len] for emiss, seq_len in zip(emiss_probs.detach().cpu(), seq_lens)]
        return transitions, emissions

    def get_pretrain_optimizer(self):
        pretrain_optimizer = torch.optim.Adam(
            self.neural_module.parameters(),
            lr=5e-4,
            weight_decay=1e-5
        )
        return pretrain_optimizer

    def get_optimizer(self):
        # ----- initialize optimizer -----
        hmm_params = [
            self._model.unnormalized_emiss,
            self._model.unnormalized_trans,
            self._model.state_priors
        ]
        optimizer = torch.optim.Adam(
            [{'params': self.neural_module.parameters(), 'lr': self._config.nn_lr},
             {'params': hmm_params}],
            lr=self._config.hmm_lr,
            weight_decay=1e-5
        )
        return optimizer
