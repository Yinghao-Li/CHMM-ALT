import os
import logging
import numpy as np
from tqdm.auto import tqdm
from typing import Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from seqlbtoolkit.eval import Metric, get_ner_metrics
from seqlbtoolkit.chmm.dataset import CHMMDataset
from .args import PriorConfig
from .model import PriorCHMM, DirCHMMMetric

logger = logging.getLogger(__name__)

OUT_RECALL = 0.9
OUT_PRECISION = 0.8


class PriorTrainer:
    def __init__(self,
                 config: PriorConfig,
                 collate_fn,
                 training_dataset: Optional[CHMMDataset] = None,
                 valid_dataset: Optional[CHMMDataset] = None,
                 test_dataset: Optional[CHMMDataset] = None,
                 pretrain_optimizer=None,
                 optimizer=None):

        self._model: Optional[PriorCHMM] = None
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

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, x):
        logger.warning("Updating DirCHMMTrainer.config")
        self._config = x

    @property
    def model(self) -> PriorCHMM:
        return self._model

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
        self._model = PriorCHMM(
            config=self._config,
            state_prior=self._state_prior,
            trans_matrix=self._trans_mat,
            emiss_matrix=self._emiss_mat
        )
        return self

    def initialize_optimizers(self, optimizer=None, pretrain_optimizer=None):
        self._optimizer = self.get_optimizer() if optimizer is None else optimizer
        self._pretrain_optimizer = self.get_pretrain_optimizer() if pretrain_optimizer is None else pretrain_optimizer

    def pretrain_step(self,
                      data_loader,
                      optimizer,
                      trans_,
                      emiss_: Optional[torch.Tensor] = None,
                      src_priors: Optional[torch.Tensor] = None):
        """
        One pre-training epoch of CHMM
        """
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
            nn_trans, nn_emiss, (conc_o2o, conc_e2e_scale) = self._model._nn_module(embs=emb_batch)
            batch_size, max_seq_len, n_hidden, _ = nn_trans.size()

            loss_mask = torch.zeros([batch_size, max_seq_len], device=self._config.device)
            for n in range(batch_size):
                loss_mask[n, :seq_lens[n]] = 1
            trans_mask = loss_mask.view(batch_size, max_seq_len, 1, 1)
            trans_pred = trans_mask * nn_trans
            trans_true = trans_mask * trans_.view(1, 1, n_hidden, n_hidden).repeat(batch_size, max_seq_len, 1, 1)

            if trans_ is not None:
                l1 = F.mse_loss(trans_pred, trans_true)
            else:
                l1 = 0

            if nn_emiss is not None and (src_priors is None or not self.config.use_src_prior):
                emiss_pred = nn_emiss
                emiss_true = emiss_.view(
                    1, self._config.n_src, self._config.d_hidden, self._config.d_obs
                ).repeat(batch_size, 1, 1, 1)
                l2 = F.mse_loss(emiss_pred, emiss_true)
            elif src_priors is not None and self.config.use_src_prior:
                rel_pred = torch.concat((conc_o2o.unsqueeze(-1), conc_e2e_scale), dim=-1)
                rel_pred.retain_grad()
                l2 = F.mse_loss(rel_pred, src_priors.unsqueeze(0).expand_as(rel_pred))
            else:
                l2 = 0

            loss = l1 + l2
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size
        train_loss /= num_samples
        return train_loss

    def training_step(self, data_loader, optimizer, src_priors: Optional[torch.Tensor] = None):
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
            log_probs = self._model(
                emb=emb_batch, obs=obs_batch, seq_lengths=seq_lens,
                normalize_observation=self._config.obs_normalization
            )
            loss = -log_probs.mean()

            conc = self._model.inter_results.pop_attr("conc_batch")
            if self.config.use_src_prior and src_priors is not None:
                loss += 10 / (i+1) * F.mse_loss(conc[:, :, 1:], src_priors.unsqueeze(0).expand_as(conc)[:, :, 1:])

            loss.backward()
            optimizer.step()

            # track loss
            train_loss += loss.item() * batch_size
        train_loss /= num_samples

        return train_loss

    def train(self):
        training_dataloader = self.get_dataloader(self._training_dataset, shuffle=True)
        src_priors = self.compile_src_reliability()

        # ----- pre-train neural module -----
        if self._config.num_lm_nn_pretrain_epochs > 0:
            logger.info(" ----- ")
            logger.info("Pre-training neural module...")

            for epoch_i in range(self._config.num_lm_nn_pretrain_epochs):
                train_loss = self.pretrain_step(
                    training_dataloader, self._pretrain_optimizer, self._trans_mat, self._emiss_mat, src_priors
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

            train_loss = self.training_step(
                training_dataloader,
                self._optimizer,
                src_priors=src_priors
            )
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

    def evaluate(self, dataset: CHMMDataset) -> Metric:

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

    def predict(self, dataset: CHMMDataset):

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

    def compile_src_reliability(self, dataset: Optional[CHMMDataset] = None) -> torch.Tensor:
        """
        Get source reliability in the similar format as Neural module's output

        Parameters
        ----------
        dataset: the dataset that contains the desired source metrics

        Returns
        -------
        source reliabilities that can be used to regularize training
        """
        if dataset is None:
            dataset = self._test_dataset if self._test_dataset is not None else self._training_dataset
        src_metrics = dataset.src_metrics
        src_reliabilities = list()
        for src in self.config.sources:
            reliability = [0.8]
            for ent in self.config.entity_types:
                reliability.append(src_metrics[src][ent]['f1'])
            src_reliabilities.append(reliability)
        return torch.tensor(src_reliabilities, device=self.config.device)

    def save(self,
             output_dir: Optional[str] = None,
             save_optimizer: Optional[bool] = False):
        """
        Save model parameters as well as trainer parameters

        Parameters
        ----------
        output_dir: model directory
        save_optimizer: whether to save optimizer

        Returns
        -------
        None
        """
        output_dir = output_dir if output_dir is not None else self._config.output_dir
        logger.info(f"Saving model to {output_dir}")

        model_state_dict = self._model.state_dict()
        torch.save(model_state_dict, os.path.join(output_dir, 'chmm.bin'))

        self._config.save(output_dir)

        if save_optimizer:
            logger.info("Saving optimizer and scheduler")
            torch.save(self._optimizer.state_dict(), os.path.join(output_dir, "chmm-optimizer.pt"))
            torch.save(self._pretrain_optimizer.state_dict(), os.path.join(output_dir, "chmm-pretrain-optimizer.pt"))

    def load(self, input_dir: Optional[str] = None, load_optimizer: Optional[bool] = False):
        """
        Load model parameters.

        Parameters
        ----------
        input_dir: model directory
        load_optimizer: whether load other trainer parameters

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
        self._model.load_state_dict(torch.load(os.path.join(input_dir, 'chmm.bin')))
        self._model.to(self.config.device)

        if load_optimizer:
            logger.info("Loading optimizer and scheduler")
            if self._optimizer is None:
                self.initialize_optimizers()
            if os.path.isfile(os.path.join(input_dir, "chmm-optimizer.pt")):
                self._optimizer.load_state_dict(
                    torch.load(os.path.join(input_dir, "chmm-optimizer.pt"), map_location=self.config.device)
                )
            else:
                logger.warning("Optimizer file does not exist!")
            if os.path.isfile(os.path.join(input_dir, "chmm-pretrain-optimizer.pt")):
                self._pretrain_optimizer.load_state_dict(
                    torch.load(os.path.join(input_dir, "chmm-pretrain-optimizer.pt"))
                )
            else:
                logger.warning("Pretrain optimizer file does not exist!")
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
