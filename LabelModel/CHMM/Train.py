import sys
sys.path.append('../..')

import os
import logging
import numpy as np
from tqdm.auto import tqdm
from typing import List, Optional
from seqeval import metrics
from seqeval.scheme import IOB2
from collections import OrderedDict

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from LabelModel.CHMM.Args import CHMMConfig
from LabelModel.CHMM.Data import MultiSrcNERDataset
from LabelModel.CHMM.Model import CHMM

logger = logging.getLogger(__name__)

OUT_RECALL = 0.9
OUT_PRECISION = 0.8


class CHMMTrainer:
    def __init__(self,
                 config: CHMMConfig,
                 collate_fn,
                 training_dataset=None,
                 valid_dataset=None,
                 test_dataset=None,
                 pretrain_optimizer=None,
                 optimizer=None):

        self._model = None
        self._config = config
        self._training_dataset = training_dataset
        self._valid_dataset = valid_dataset
        self._test_dataset = test_dataset
        self._collate_fn = collate_fn
        self._pretrain_optimizer = pretrain_optimizer
        self._optimizer = optimizer
        self._state_prior = None
        self._trans_mat = None
        self._emiss_mat = None

        self._has_appended_obs = False  # reserved for alternate training

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, x):
        logger.warning("Updating CHMMTrainer.config")
        self._config = x

    def initialize_trainer(self):
        """
        Initialize necessary components for training

        Returns
        -------
        the initialized trainer
        """
        self.initialize_matrices()
        self.initialize_model()
        self.initialize_optimizers()
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
        self._state_prior = torch.zeros(self._config.d_hidden, device=self._config.device) + 1e-2
        self._state_prior[0] += 1 - self._state_prior.sum()

        intg_obs = list(map(np.array, self._training_dataset.obs + self._valid_dataset.obs))
        self._trans_mat = torch.tensor(initialise_transmat(
            observations=intg_obs, label_set=self._config.bio_label_types)[0], dtype=torch.float)
        self._emiss_mat = torch.tensor(initialise_emissions(
            observations=intg_obs, label_set=self._config.bio_label_types,
            sources=self._config.sources, src_priors=self._config.src_priors
        )[0], dtype=torch.float)
        return self

    def initialize_model(self):
        self._model = CHMM(
            config=self._config,
            state_prior=self._state_prior,
            trans_matrix=self._trans_mat,
            emiss_matrix=self._emiss_mat
        )
        return self

    def initialize_optimizers(self, optimizer=None, pretrain_optimizer=None):
        self._optimizer = self.get_optimizer() if optimizer is None else optimizer
        self._pretrain_optimizer = self.get_pretrain_optimizer() if pretrain_optimizer is None else pretrain_optimizer

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
            nn_trans, nn_emiss = self._model._nn_module(embs=emb_batch)
            batch_size, max_seq_len, n_hidden, _ = nn_trans.size()
            n_obs = nn_emiss.size(-1)

            loss_mask = torch.zeros([batch_size, max_seq_len], device=self._config.device)
            for n in range(batch_size):
                loss_mask[n, :seq_lens[n]] = 1
            trans_mask = loss_mask.view(batch_size, max_seq_len, 1, 1)
            trans_pred = trans_mask * nn_trans
            trans_true = trans_mask * trans_.view(1, 1, n_hidden, n_hidden).repeat(batch_size, max_seq_len, 1, 1)

            emiss_mask = loss_mask.view(batch_size, max_seq_len, 1, 1, 1)
            emiss_pred = emiss_mask * nn_emiss
            emiss_true = emiss_mask * emiss_.view(
                1, 1, self._config.n_src, n_hidden, n_obs
            ).repeat(batch_size, max_seq_len, 1, 1, 1)
            if trans_ is not None:
                l1 = F.mse_loss(trans_pred, trans_true)
            else:
                l1 = 0
            if emiss_ is not None:
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
            logger.info(" ----- \nPre-training neural module...")
            for epoch_i in range(self._config.num_lm_nn_pretrain_epochs):
                train_loss = self.pretrain_step(
                    training_dataloader, self._pretrain_optimizer, self._trans_mat, self._emiss_mat
                )
                logger.info(f"Epoch: {epoch_i}, Loss: {train_loss}")
            logger.info("Neural module pretrained!")

        valid_result_list = list()
        best_f1 = 0
        tolerance_epoch = 0

        # ----- start training process -----
        logger.info(" ----- \nStart training CHMM...")
        for epoch_i in range(self._config.num_lm_train_epochs):
            logger.info("------")
            logger.info(f"Epoch {epoch_i + 1} of {self._config.num_lm_train_epochs}")

            train_loss = self.training_step(training_dataloader, self._optimizer)
            valid_results = self.evaluate(self._valid_dataset)

            logger.info("Training loss: %.4f" % train_loss)
            logger.info("Validation results:")
            for k, v in valid_results.items():
                logger.info(f"\t{k}: {v:.4f}")

            # ----- save model -----
            if valid_results['f1'] >= best_f1:
                self.save()
                logger.info("Checkpoint Saved!\n")
                best_f1 = valid_results['f1']
                tolerance_epoch = 0
            else:
                tolerance_epoch += 1

            # ----- log history -----
            valid_result_list.append(valid_results)
            if tolerance_epoch > self._config.num_lm_valid_tolerance:
                logger.info("Training stopped because of exceeding tolerance")
                break

        # retrieve the best state dict
        self.load()

        return valid_result_list

    def test(self):
        self._model.to(self._config.device)
        test_results = self.evaluate(self._test_dataset)

        logger.info("Test results:")
        for k, v in test_results.items():
            logger.info(f"\t{k}: {v:.4f}")
        return test_results

    def evaluate(self, dataset: MultiSrcNERDataset):

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
        metric_values = OrderedDict()
        metric_values['precision'] = metrics.precision_score(true_lbs, pred_lbs, mode='strict', scheme=IOB2)
        metric_values['recall'] = metrics.recall_score(true_lbs, pred_lbs, mode='strict', scheme=IOB2)
        metric_values['f1'] = metrics.f1_score(true_lbs, pred_lbs, mode='strict', scheme=IOB2)

        return metric_values

    def predict(self, dataset: MultiSrcNERDataset):

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

    def get_dataloader(self, dataset, shuffle=False):
        if dataset is not None:
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=self._config.lm_batch_size,
                collate_fn=self._collate_fn,
                shuffle=shuffle,
                drop_last=False
            )
            return dataloader
        else:
            logger.error('Dataset is not defined')
            raise ValueError("Dataset is not defined!")

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

    def save(self, output_dir: Optional[str] = None):
        """
        Save model parameters as well as trainer parameters

        Parameters
        ----------
        output_dir: model directory

        Returns
        -------
        None
        """
        model_state_dict = self._model.state_dict()
        optimizer_state_dict = self._optimizer.state_dict()
        pretrain_optimizer_state_dict = self._pretrain_optimizer.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'optimizer': optimizer_state_dict,
            'pretrain_optimizer': pretrain_optimizer_state_dict,
            'state_prior': self._state_prior,
            'transitions': self._trans_mat,
            'emissions': self._emiss_mat,
            'config': self._config
        }
        output_dir = output_dir if output_dir is not None else self._config.output_dir
        torch.save(checkpoint, os.path.join(output_dir, 'chmm.bin'))

    def load(self, input_dir: Optional[str] = None, load_trainer_params: Optional[bool] = False):
        """
        Load model parameters.

        Parameters
        ----------
        input_dir: model directory
        load_trainer_params: whether load other trainer parameters

        Returns
        -------
        self
        """
        input_dir = input_dir if input_dir is not None else self._config.output_dir
        checkpoint = torch.load(os.path.join(input_dir, 'chmm.bin'))
        self._model.load_state_dict(checkpoint['model'])
        self._config = checkpoint['config']
        if load_trainer_params:
            self._optimizer.load_state_dict([checkpoint['optimizer']])
            self._pretrain_optimizer.load_state_dict([checkpoint['pretrain_optimizer']])
            self._state_prior = checkpoint['state_prior']
            self._trans_mat = checkpoint['transitions']
            self._emiss_mat = checkpoint['emissions']
        return self


def initialise_startprob(observations,
                         label_set,
                         src_idx=None):
    """
    calculate initial hidden states (not used in our setup since our sequences all begin from
    [CLS], which corresponds to hidden state "O".
    :param src_idx: source index
    :param label_set: a set of all possible label_set
    :param observations: n_instances X seq_len X n_src X d_obs
    :return: probabilities for the initial hidden states
    """
    n_src = observations[0].shape[1]
    logger.info("Constructing start distribution prior...")

    init_counts = np.zeros((len(label_set),))

    if src_idx is not None:
        for obs in observations:
            init_counts[obs[0, src_idx].argmax()] += 1
    else:
        for obs in observations:
            for z in range(n_src):
                init_counts[obs[0, z].argmax()] += 1

    for i, label in enumerate(label_set):
        if i == 0 or label.startswith("B-"):
            init_counts[i] += 1

    startprob_prior = init_counts + 1
    startprob_ = np.random.dirichlet(init_counts + 1E-10)
    return startprob_, startprob_prior


# TODO: try to use a more reliable source to start the transition and emission
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
