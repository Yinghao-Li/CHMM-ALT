import sys
sys.path.append('../..')

import logging
from tqdm.auto import tqdm

import torch
from torch.nn import functional as F

from seqlbtoolkit.eval import Metric, get_ner_metrics
from seqlbtoolkit.chmm.dataset import CHMMBaseDataset
from seqlbtoolkit.chmm.train import CHMMBaseTrainer

from .args import DirCHMMConfig
from .model import DirCHMM, DirCHMMMetric

logger = logging.getLogger(__name__)


class DirCHMMTrainer(CHMMBaseTrainer):
    def __init__(self,
                 config: DirCHMMConfig,
                 collate_fn,
                 training_dataset=None,
                 valid_dataset=None,
                 test_dataset=None,
                 pretrain_optimizer=None,
                 optimizer=None):

        super().__init__(
            config, collate_fn, training_dataset, valid_dataset, test_dataset, pretrain_optimizer, optimizer
        )

    def initialize_trainer(self):
        """
        Initialize necessary components for training, returns self
        """
        CHMMBaseTrainer.initialize_trainer(self)
        return self

    def initialize_model(self):
        self._model = DirCHMM(
            config=self._config,
            state_prior=self._state_prior,
            trans_matrix=self._trans_mat,
            emiss_matrix=self._emiss_mat
        )
        return self

    def pretrain_step(self, data_loader, optimizer, trans_, emiss_):
        train_loss = 0
        num_samples = 0

        self._model._nn_module.train()
        if trans_ is not None:
            trans_ = trans_.to(self._config.device)
        if emiss_ is not None:
            emiss_ = emiss_.to(self._config.device)

        for i, batch in enumerate(tqdm(data_loader)):
            emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self._config.device), batch[:3])
            batch_size = len(obs_batch)
            num_samples += batch_size

            optimizer.zero_grad()
            nn_trans, nn_emiss, _ = self._model._nn_module(embs=emb_batch)
            batch_size, max_seq_len, n_hidden, _ = nn_trans.size()

            loss_mask = torch.zeros([batch_size, max_seq_len], device=self._config.device)
            for n in range(batch_size):
                loss_mask[n, :seq_lens[n]] = 1
            trans_mask = loss_mask.view(batch_size, max_seq_len, 1, 1)
            trans_pred = trans_mask * nn_trans
            trans_true = trans_mask * trans_.view(1, 1, n_hidden, n_hidden).repeat(batch_size, max_seq_len, 1, 1)

            emiss_pred = emiss_true = 0
            if nn_emiss is not None:
                emiss_pred = nn_emiss
                emiss_true = emiss_.view(
                    1, self._config.n_src, self._config.d_hidden, self._config.d_obs
                ).repeat(batch_size, 1, 1, 1)

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
        train_loss /= num_samples

        return train_loss

    def train(self):
        training_dataloader = self.get_dataloader(self._training_dataset, shuffle=True)

        # ----- pre-train neural module -----
        if self._config.num_lm_nn_pretrain_epochs > 0:
            logger.info(" ----- ")
            logger.info("Pre-training neural module...")
            for epoch_i in range(self._config.num_lm_nn_pretrain_epochs):
                train_loss = self.pretrain_step(
                    training_dataloader, self._pretrain_optimizer, self._trans_mat, self._emiss_mat
                )
                logger.info(f"Epoch: {epoch_i}, Loss: {train_loss}")
            logger.info("Neural module pretrained!")

        valid_results = DirCHMMMetric()
        best_f1 = 0
        tolerance_epoch = 0

        # ----- start training process -----
        logger.info(" ----- ")
        logger.info("Training Dir-CHMM...")
        for epoch_i in range(self._config.num_lm_train_epochs):
            logger.info("------")
            logger.info(f"Epoch {epoch_i + 1} of {self._config.num_lm_train_epochs}")

            train_loss = self.training_step(training_dataloader, self._optimizer)
            valid_metrics = self.evaluate(self._valid_dataset)

            logger.info("Training loss: %.4f" % train_loss)
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
            inter_results = self.model.pop_inter_results()
            valid_results.append(valid_metrics).append(inter_results)
            if tolerance_epoch > self._config.num_lm_valid_tolerance:
                logger.info("Training stopped because of exceeding tolerance")
                break

        # retrieve the best state dict
        self.load()

        return valid_results

    def test(self) -> Metric:
        self._model.to(self._config.device)
        test_metrics = self.evaluate(self._test_dataset)

        logger.info("Test results:")
        for k, v in test_metrics.items():
            logger.info(f"\t{k}: {v:.4f}")
        return test_metrics

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

    def get_pretrain_optimizer(self):
        pretrain_optimizer = torch.optim.Adam(
            self._model._nn_module.parameters(),
            lr=5e-4,
            weight_decay=1e-5
        )
        return pretrain_optimizer

    def get_optimizer(self):
        # ----- initialize optimizer -----
        hmm_params = [
            self._model._unnormalized_emiss,
            self._model._unnormalized_trans,
            self._model._state_priors
        ]
        optimizer = torch.optim.Adam(
            [{'params': self._model._nn_module.parameters(), 'lr': self._config.nn_lr},
             {'params': hmm_params}],
            lr=self._config.hmm_lr,
            weight_decay=1e-5
        )
        return optimizer
