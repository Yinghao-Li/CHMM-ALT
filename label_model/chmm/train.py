import os
import time
import logging
from tqdm.auto import tqdm

import numpy as np
from typing import Optional

import torch
from torch.nn import functional as F

from .model import CHMM
from .dataset import CHMMBaseDataset
from seqlbtoolkit.training.eval import Metric, get_ner_metrics
from seqlbtoolkit.training.train import BaseTrainer


OUT_RECALL = 0.9
OUT_PRECISION = 0.8

logger = logging.getLogger(__name__)


def initialise_transmat(observations,
                        label_set,
                        src_idx=None):
    """
    initialize transition matrix
    :param src_idx: the index of the source of which the transition statistics is computed.
                    If None, use all sources
    :param label_set: a set of all possible label_set
    :param observations: n_instances X seq_len X n_src X d_obs
    :return: initial transition matrix and transition counts
    """

    logger.info("Constructing transition matrix prior...")
    n_src = observations[0].shape[1]
    trans_counts = np.zeros((len(label_set), len(label_set)))

    if src_idx is not None:
        for obs in observations:
            for k in range(0, len(obs) - 1):
                trans_counts[obs[k, src_idx].argmax(), obs[k + 1, src_idx].argmax()] += 1
    else:
        for obs in observations:
            for k in range(0, len(obs) - 1):
                for z in range(n_src):
                    trans_counts[obs[k, z].argmax(), obs[k + 1, z].argmax()] += 1

    # update transition matrix with prior knowledge
    for i, label in enumerate(label_set):
        if label.startswith("B-") or label.startswith("I-"):
            trans_counts[i, label_set.index("I-" + label[2:])] += 1
        elif i == 0 or label.startswith("I-"):
            for j, label2 in enumerate(label_set):
                if j == 0 or label2.startswith("B-"):
                    trans_counts[i, j] += 1

    transmat_prior = trans_counts + 1
    # initialize transition matrix with dirichlet distribution
    transmat_ = np.vstack([np.random.dirichlet(trans_counts2 + 1E-10)
                           for trans_counts2 in trans_counts])
    return transmat_, transmat_prior


def initialise_emissions(observations,
                         label_set,
                         sources,
                         src_priors,
                         strength=1000):
    """
    initialize emission matrices
    :param sources: source names
    :param src_priors: source priors
    :param label_set: a set of all possible label_set
    :param observations: n_instances X seq_len X n_src X d_obs
    :param strength: Don't know what this is for
    :return: initial emission matrices and emission counts?
    """

    logger.info("Constructing emission probabilities...")

    obs_counts = np.zeros((len(sources), len(label_set)), dtype=np.float64)
    # extract the total number of observations for each prior
    for obs in observations:
        obs_counts += obs.sum(axis=0)
    for source_index, source in enumerate(sources):
        # increase p(O)
        obs_counts[source_index, 0] += 1
        # increase the "reasonable" observations
        for pos_index, pos_label in enumerate(label_set[1:]):
            if pos_label[2:] in src_priors[source]:
                obs_counts[source_index, pos_index] += 1
    # construct probability distribution from counts
    obs_probs = obs_counts / (obs_counts.sum(axis=1, keepdims=True) + 1E-3)

    # initialize emission matrix
    matrix = np.zeros((len(sources), len(label_set), len(label_set)))

    for source_index, source in enumerate(sources):
        for pos_index, pos_label in enumerate(label_set):

            # Simple case: set P(O=x|Y=x) to be the recall
            recall = 0
            if pos_index == 0:
                recall = OUT_RECALL
            elif pos_label[2:] in src_priors[source]:
                _, recall = src_priors[source][pos_label[2:]]
            matrix[source_index, pos_index, pos_index] = recall

            for pos_index2, pos_label2 in enumerate(label_set):
                if pos_index2 == pos_index:
                    continue
                elif pos_index2 == 0:
                    precision = OUT_PRECISION
                elif pos_label2[2:] in src_priors[source]:
                    precision, _ = src_priors[source][pos_label2[2:]]
                else:
                    precision = 1.0

                # Otherwise, we set the probability to be inversely proportional to the precision
                # and the (unconditional) probability of the observation
                error_prob = (1 - recall) * (1 - precision) * (0.001 + obs_probs[source_index, pos_index2])

                # We increase the probability for boundary errors (i.e. I-ORG -> B-ORG)
                if pos_index > 0 and pos_index2 > 0 and pos_label[2:] == pos_label2[2:]:
                    error_prob *= 5

                # We increase the probability for errors with same boundary (i.e. I-ORG -> I-GPE)
                if pos_index > 0 and pos_index2 > 0 and pos_label[0] == pos_label2[0]:
                    error_prob *= 2

                matrix[source_index, pos_index, pos_index2] = error_prob

            error_indices = [i for i in range(len(label_set)) if i != pos_index]
            error_sum = matrix[source_index, pos_index, error_indices].sum()
            matrix[source_index, pos_index, error_indices] /= (error_sum / (1 - recall) + 1E-5)

    emission_priors = matrix * strength
    emission_probs = matrix
    return emission_probs, emission_priors


class CHMMTrainer(BaseTrainer):
    def __init__(self,
                 config,
                 collate_fn=None,
                 training_dataset=None,
                 valid_dataset=None,
                 test_dataset=None,
                 pretrain_optimizer=None):

        super().__init__(config, training_dataset, valid_dataset, test_dataset, collate_fn)
        self._model = None
        self._pretrain_optimizer = pretrain_optimizer
        self._init_state_prior = None
        self._init_trans_mat = None
        self._init_emiss_mat = None

    @property
    def neural_module(self):
        return self._model.neural_module

    def initialize_trainer(self):
        """
        Initialize necessary components for training
        Note: Better not change the order

        Returns
        -------
        the initialized trainer
        """
        self.initialize_matrices()
        self.initialize_model()
        self.initialize_optimizers()
        return self

    def initialize_model(self):
        self._model = CHMM(
            config=self._config,
            state_prior=self._init_state_prior,
            trans_matrix=self._init_trans_mat,
            emiss_matrix=self._init_emiss_mat
        )
        return self

    def initialize_matrices(self):
        """
        Initialize <HMM> transition and emission matrices

        Returns
        -------
        self
        """
        assert self._training_dataset and self._valid_dataset
        # inject prior knowledge about transition and emission
        self._init_state_prior = torch.zeros(self._config.d_hidden, device=self._config.device) + 1e-2
        self._init_state_prior[0] += 1 - self._init_state_prior.sum()

        intg_obs = list(map(np.array, self._training_dataset.obs + self._valid_dataset.obs))

        # construct/load initial transition matrix
        dataset_dir = os.path.split(self._config.train_path)[0]
        transmat_path = os.path.join(dataset_dir, "init_transmat.pt")
        if getattr(self._config, "load_init_mat", False):
            if os.path.isfile(transmat_path):
                logger.info("Loading initial transition matrix from disk")
                self._init_trans_mat = torch.load(transmat_path)

                # if the loaded transmat does not have the proper shape, re-calculate it.
                s0_transmat, s1_transmat = self._init_trans_mat.shape
                if not (s0_transmat == s1_transmat == self.config.d_obs):
                    self._init_trans_mat = None

        if self._init_trans_mat is None:
            self._init_trans_mat = torch.tensor(initialise_transmat(
                observations=intg_obs, label_set=self._config.bio_label_types
            )[0], dtype=torch.float)

            if getattr(self._config, "save_init_mat", False):
                logger.info("Saving initial transition matrix")
                torch.save(self._init_trans_mat, transmat_path)

        # construct/load initial emission matrix
        emissmat_path = os.path.join(dataset_dir, "init_emissmat.pt")
        if getattr(self._config, "load_init_mat", False):
            if os.path.isfile(emissmat_path):
                logger.info("Loading initial emission matrix from disk")
                self._init_emiss_mat = torch.load(emissmat_path)

                # if the loaded emissmat does not have the proper shape, re-calculate it.
                s0_emissmat, s1_emissmat, s2_emissmat = self._init_emiss_mat.shape
                if not (s0_emissmat == self.config.n_src) and (s1_emissmat == s2_emissmat == self.config.d_obs):
                    self._init_emiss_mat = None

        if self._init_emiss_mat is None:
            self._init_emiss_mat = torch.tensor(initialise_emissions(
                observations=intg_obs, label_set=self._config.bio_label_types,
                sources=self._config.sources, src_priors=self._config.src_priors
            )[0], dtype=torch.float)

            if getattr(self._config, "save_init_mat", False):
                logger.info("Saving initial emission matrix")
                torch.save(self._init_emiss_mat, emissmat_path)

        return self

    def initialize_optimizers(self, optimizer=None, pretrain_optimizer=None):
        self._optimizer = self.get_optimizer() if optimizer is None else optimizer
        self._pretrain_optimizer = self.get_pretrain_optimizer() if pretrain_optimizer is None else pretrain_optimizer

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

    def training_step(self, data_loader):
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
            self._optimizer.zero_grad()
            log_probs, _ = self._model(
                emb=emb_batch, obs=obs_batch, seq_lengths=seq_lens,
                normalize_observation=self._config.obs_normalization
            )

            loss = -log_probs.mean()
            loss.backward()
            self._optimizer.step()

            # track loss
            train_loss += loss.item() * batch_size

        if start_time is not None:
            logger.info(f"Training time for current epoch: {time.time() - start_time} s.")

        train_loss /= num_samples

        return train_loss

    def train(self) -> Metric:
        training_dataloader = self.get_dataloader(
            self._training_dataset,
            shuffle=True,
            batch_size=self.config.lm_batch_size
        )

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

            train_loss = self.training_step(training_dataloader)
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

        data_loader = self.get_dataloader(dataset, batch_size=self.config.lm_batch_size)
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

    def valid(self) -> Metric:
        self._model.to(self._config.device)
        valid_metrics = self.evaluate(self._valid_dataset)

        logger.info("Validation results:")
        for k, v in valid_metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        return valid_metrics

    def test(self) -> Metric:
        self._model.to(self._config.device)
        test_metrics = self.evaluate(self._test_dataset)

        logger.info("Test results:")
        for k, v in test_metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        return test_metrics

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

    def save(self,
             output_dir: Optional[str] = None,
             save_optimizer: Optional[bool] = False,
             model_name: Optional[str] = 'chmm',
             optimizer_name: Optional[str] = 'chmm-optimizer',
             pretrain_optimizer_name: Optional[str] = 'chmm-pretrain-optimizer',
             **kwargs):
        """
        Save model parameters as well as trainer parameters

        Parameters
        ----------
        output_dir: model directory
        save_optimizer: whether to save optimizer
        model_name: model name (suffix free)
        optimizer_name: optimizer name (suffix free)
        pretrain_optimizer_name: pretrain optimizer name (suffix free)

        Returns
        -------
        None
        """
        super().save(output_dir=output_dir,
                     save_optimizer=save_optimizer,
                     save_scheduler=False,
                     model_name=model_name,
                     optimizer_name=optimizer_name)

        if save_optimizer:
            torch.save(self._pretrain_optimizer.state_dict(),
                       os.path.join(output_dir, f"{pretrain_optimizer_name}.bin"))

        return self

    def load(self,
             input_dir: Optional[str] = None,
             load_optimizer: Optional[bool] = False,
             model_name: Optional[str] = 'chmm',
             optimizer_name: Optional[str] = 'chmm-optimizer',
             pretrain_optimizer_name: Optional[str] = 'chmm-pretrain-optimizer',
             **kwargs):
        """
        Load model parameters.

        Parameters
        ----------
        input_dir: model directory
        load_optimizer: whether load other trainer parameters
        model_name: model name (suffix free)
        optimizer_name: optimizer name (suffix free)
        pretrain_optimizer_name: pretrain optimizer name (suffix free)

        Returns
        -------
        self
        """
        input_dir = input_dir if input_dir is not None else self._config.output_dir
        if self._model is not None:
            logger.warning(f"The original model {type(self._model)} in {type(self)} is not None. "
                           f"It will be overwritten by the loaded model!")
        logger.info(f"Loading model from {input_dir}")
        self.initialize_model()
        self._model.load_state_dict(torch.load(os.path.join(input_dir, f'{model_name}.bin')))
        self._model.to(self.config.device)

        if load_optimizer:
            logger.info("Loading optimizer and scheduler")
            if self._optimizer is None:
                self.initialize_optimizers()
            if os.path.isfile(os.path.join(input_dir, f"{optimizer_name}.bin")):
                self._optimizer.load_state_dict(
                    torch.load(os.path.join(input_dir, f"{optimizer_name}.bin"), map_location=self.config.device)
                )
            else:
                logger.warning("Optimizer file does not exist!")
            if os.path.isfile(os.path.join(input_dir, f"{pretrain_optimizer_name}.bin")):
                self._pretrain_optimizer.load_state_dict(
                    torch.load(os.path.join(input_dir, f"{pretrain_optimizer_name}.bin"))
                )
            else:
                logger.warning("Pretrain optimizer file does not exist!")
        return self

    def save_results(self,
                     output_dir: str,
                     valid_results: Optional[Metric] = None,
                     file_name: Optional[str] = 'results',
                     disable_final_valid: Optional[bool] = False,
                     disable_test: Optional[bool] = False,
                     disable_inter_results: Optional[bool] = False) -> None:
        """
        Save training (validation) results

        Parameters
        ----------
        output_dir: output directory, should be a folder
        valid_results: validation results during the training process
        file_name: file name
        disable_final_valid: disable final validation process (getting validation results of the trained model)
        disable_test: disable test process
        disable_inter_results: do not save inter-results

        Returns
        -------
        None
        """
        if not disable_final_valid:
            logger.info("Getting final validation metrics")
            valid_metrics = self.valid()
        else:
            valid_metrics = None

        if not disable_test:
            logger.info("Getting test metrics.")
            test_metrics = self.test()
        else:
            test_metrics = None

        # write validation and test results
        result_file = os.path.join(output_dir, f'{file_name}.txt')
        logger.info(f"Writing results to {result_file}")
        self.write_result(file_path=result_file,
                          valid_results=valid_results,
                          final_valid_metrics=valid_metrics,
                          test_metrics=test_metrics)

        if not disable_inter_results:
            # save validation inter results
            logger.info(f"Saving inter results")
            inter_result_file = os.path.join(output_dir, f'{file_name}-inter.pt')
            torch.save(valid_results.__dict__, inter_result_file)
        return None

    @staticmethod
    def write_result(file_path: str,
                     valid_results: Optional[Metric] = None,
                     final_valid_metrics: Optional[Metric] = None,
                     test_metrics: Optional[Metric] = None) -> None:
        """
        Support functions for saving training results

        Parameters
        ----------
        file_path: where to save results
        valid_results: validation results during the training process
        final_valid_metrics: validation results of the trained model
        test_metrics

        Returns
        -------

        """
        with open(file_path, 'w') as f:
            if valid_results is not None:
                for i in range(len(valid_results)):
                    f.write(f"[Epoch {i + 1}]\n")
                    for k in ['precision', 'recall', 'f1']:
                        f.write(f"  {k}: {valid_results[k][i]:.4f}")
                    f.write("\n")
            if final_valid_metrics is not None:
                f.write(f"[Best Validation]\n")
                for k in ['precision', 'recall', 'f1']:
                    f.write(f"  {k}: {final_valid_metrics[k]:.4f}")
                f.write("\n")
            if test_metrics is not None:
                f.write(f"[Test]\n")
                for k in ['precision', 'recall', 'f1']:
                    f.write(f"  {k}: {test_metrics[k]:.4f}")
                f.write("\n")
        return None
