import logging
import torch
import copy
import os
import numpy as np
from tqdm.auto import tqdm
from typing import List
from torch.nn import functional as F
from torch.utils.data import DataLoader

from Src.Utils import get_results, anno_space_map
from Src.CHMM.CHMMData import Dataset
from Src.DataAssist import initialise_transmat, initialise_emissions
from Src.CHMM.CHMMModel import ConditionalHMM

logger = logging.getLogger(__name__)


class CHMMTrainer:
    def __init__(self, training_args, data_args,
                 train_dataset, eval_dataset, test_dataset,
                 collate_fn, pretrain_optimizer=None, optimizer=None):

        self.model = None
        self.data_args = data_args
        self.training_args = training_args
        self.device = training_args.device
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.collate_fn = collate_fn
        self.pretrain_optimizer = pretrain_optimizer
        self.optimizer = optimizer
        self.state_prior = None
        self.trans_mat = None
        self.emiss_mat = None

        self.has_appended_obs = False

        self.initialize_matrices()
        self.initialize_model()
        self.initialize_optimizers()

    def initialize_matrices(self):
        assert self.train_dataset and self.eval_dataset
        # inject prior knowledge about transition and emission
        self.state_prior = torch.zeros(self.training_args.n_hidden, device=self.device) + 1e-2
        self.state_prior[0] += 1 - self.state_prior.sum()

        intg_obs = list(map(np.array, self.train_dataset.obs + self.eval_dataset.obs))
        self.trans_mat = torch.tensor(initialise_transmat(
            observations=intg_obs, label_set=self.data_args.bio_lbs)[0], dtype=torch.float)
        self.emiss_mat = torch.tensor(initialise_emissions(
            observations=intg_obs, label_set=self.data_args.bio_lbs,
            sources=self.data_args.src_to_keep, src_priors=self.data_args.src_priors
        )[0], dtype=torch.float)

    def initialize_model(self):
        self.model = ConditionalHMM(
            args=self.training_args,
            state_prior=self.state_prior,
            trans_matrix=self.trans_mat,
            emiss_matrix=self.emiss_mat,
            device=self.device
        )

    def initialize_optimizers(self, optimizer=None, pretrain_optimizer=None):
        self.optimizer = self.get_optimizer() if optimizer is None else optimizer
        self.pretrain_optimizer = self.get_pretrain_optimizer() if pretrain_optimizer is None else pretrain_optimizer

    def pretrain_step(self, data_loader, optimizer, trans_, emiss_):
        train_loss = 0
        num_samples = 0

        self.model.nn_module.train()
        if trans_ is not None:
            trans_ = trans_.to(self.device)
        if emiss_ is not None:
            emiss_ = emiss_.to(self.device)

        for i, batch in enumerate(tqdm(data_loader)):
            emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self.device), batch[:3])
            batch_size = len(obs_batch)
            num_samples += batch_size

            optimizer.zero_grad()
            nn_trans, nn_emiss = self.model.nn_module(embs=emb_batch)
            batch_size, max_seq_len, n_hidden, _ = nn_trans.size()
            n_obs = nn_emiss.size(-1)

            loss_mask = torch.zeros([batch_size, max_seq_len], device=self.device)
            for n in range(batch_size):
                loss_mask[n, :seq_lens[n]] = 1
            trans_mask = loss_mask.view(batch_size, max_seq_len, 1, 1)
            trans_pred = trans_mask * nn_trans
            trans_true = trans_mask * trans_.view(1, 1, n_hidden, n_hidden).repeat(batch_size, max_seq_len, 1, 1)

            emiss_mask = loss_mask.view(batch_size, max_seq_len, 1, 1, 1)
            emiss_pred = emiss_mask * nn_emiss
            emiss_true = emiss_mask * emiss_.view(
                1, 1, self.training_args.n_src, n_hidden, n_obs
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

        self.model.train()

        for i, batch in enumerate(tqdm(data_loader)):
            # get data
            emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self.device), batch[:3])
            batch_size = len(obs_batch)
            num_samples += batch_size

            # training step
            optimizer.zero_grad()
            log_probs, _ = self.model(
                emb=emb_batch, obs=obs_batch, seq_lengths=seq_lens,
                normalize_observation=self.training_args.obs_normalization
            )

            loss = -log_probs.mean()
            loss.backward()
            optimizer.step()

            # track loss
            train_loss += loss.item() * batch_size
        train_loss /= num_samples
        # print(train_loss)

        return train_loss

    def train(self):
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader()

        # ----- pre-train neural module -----
        if self.training_args.denoising_pretrain_epoch > 0:
            logger.info("[Neural HMM] pre-training neural module")
            for epoch_i in range(self.training_args.denoising_pretrain_epoch):
                train_loss = self.pretrain_step(
                    train_dataloader, self.pretrain_optimizer, self.trans_mat, self.emiss_mat
                )
                logger.info(f"Epoch: {epoch_i}, Loss: {train_loss}")

        micro_results = list()
        best_f1 = 0
        tolerance_epoch = 0
        best_state_dict = None

        # ----- start training process -----
        for epoch_i in range(self.training_args.denoising_epoch):
            logger.info("========= Epoch %d of %d =========" % (epoch_i + 1, self.training_args.denoising_epoch))

            train_loss = self.training_step(train_dataloader, self.optimizer)
            results = self.evaluate(eval_dataloader)

            logger.info("========= Results: epoch %d of %d =========" %
                        (epoch_i + 1, self.training_args.denoising_epoch))
            logger.info("[INFO] train loss: %.4f" % train_loss)
            logger.info("[INFO] validation results:")
            for k, v in results['micro'].items():
                if 'entity' in k:
                    logger.info(f"{k} = {v}")

            # ----- save model -----
            if results['micro']['entity_f1'] >= best_f1:
                best_state_dict = copy.deepcopy(self.model.state_dict())
                logger.info("[INFO] Checkpoint Saved!\n")
                best_f1 = results['micro']['entity_f1']
                tolerance_epoch = 0
            else:
                tolerance_epoch += 1

            # ----- log history -----
            micro_results.append(results['micro'])
            if tolerance_epoch > self.training_args.chmm_tolerance_epoch:
                logger.info("Training stopped because of exceeded tolerance")
                break

        # retrieve the best state dict
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        return micro_results

    def test(self):
        test_dataloader = self.get_test_dataloader()
        results = self.evaluate(test_dataloader)
        return results['micro']

    def evaluate(self, data_loader):

        self.model.eval()
        batch_pred_span = list()
        batch_true_span = list()
        batch_sent = list()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader)):
                # get data
                emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self.device), batch[:3])

                # get prediction
                pred_span, _ = self.model.annotate(
                    emb=emb_batch, obs=obs_batch, seq_lengths=seq_lens, label_set=self.data_args.bio_lbs,
                    normalize_observation=self.training_args.obs_normalization
                )

                if hasattr(self.data_args, 'mappings'):
                    if self.data_args.mappings is not None:
                        pred_span = [anno_space_map(ps, self.data_args.mappings, self.data_args.lbs)
                                     for ps in pred_span]

                batch_pred_span += pred_span

                # Save source text and spans
                batch_sent += batch[-2]
                batch_true_span += batch[-1]
            results = get_results(batch_pred_span, batch_true_span, batch_sent, all_labels=self.data_args.lbs)
        return results

    def annotate_data(self, partition):
        if partition == 'train':
            data_loader = self.get_train_dataloader(shuffle=False)
        elif partition == 'eval':
            data_loader = self.get_eval_dataloader()
        elif partition == 'test':
            data_loader = self.get_test_dataloader()
        else:
            raise ValueError("[CHMM] invalid data partition")

        score_list = list()
        span_list = list()
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader)):
                # get data
                emb_batch, obs_batch, seq_lens = map(lambda x: x.to(self.training_args.device), batch[:3])
                # get prediction
                # the scores are shifted back, i.e., len = len(emb)-1 = len(sentence)
                _, (scored_spans, scores) = self.model.annotate(
                    emb=emb_batch, obs=obs_batch, seq_lengths=seq_lens, label_set=self.data_args.bio_lbs,
                    normalize_observation=self.training_args.obs_normalization
                )
                score_list += scores
                span_list += scored_spans
        return span_list, score_list

    def get_train_dataloader(self, shuffle=True):
        if self.train_dataset:
            train_loader = torch.utils.data.DataLoader(
                dataset=self.train_dataset,
                batch_size=self.training_args.denoising_batch_size,
                collate_fn=self.collate_fn,
                shuffle=shuffle,
                drop_last=False
            )
            return train_loader
        else:
            raise ValueError("[CHMM] Training dataset is not defined!")

    def get_eval_dataloader(self):
        if self.eval_dataset:
            eval_loader = torch.utils.data.DataLoader(
                dataset=self.eval_dataset,
                batch_size=self.training_args.denoising_batch_size,
                collate_fn=self.collate_fn,
                shuffle=False,
                drop_last=False
            )
            return eval_loader
        else:
            raise ValueError("[CHMM] Evaluation dataset is not defined!")

    def get_test_dataloader(self):
        if self.test_dataset:
            test_loader = torch.utils.data.DataLoader(
                dataset=self.test_dataset,
                batch_size=self.training_args.denoising_batch_size,
                collate_fn=self.collate_fn,
                shuffle=False,
                drop_last=False
            )
            return test_loader
        else:
            raise ValueError("[CHMM] Test dataset is not defined!")

    def get_pretrain_optimizer(self):
        pretrain_optimizer = torch.optim.Adam(
            self.model.nn_module.parameters(),
            lr=5e-4,
            weight_decay=1e-5
        )
        return pretrain_optimizer

    def get_optimizer(self):
        # ----- initialize optimizer -----
        hmm_params = [
            self.model.unnormalized_emiss,
            self.model.unnormalized_trans,
            self.model.state_priors
        ]
        optimizer = torch.optim.Adam(
            [{'params': self.model.nn_module.parameters(), 'lr': self.training_args.nn_lr},
             {'params': hmm_params}],
            lr=self.training_args.hmm_lr,
            weight_decay=1e-5
        )
        return optimizer

    def save_model(self):
        model_state_dict = self.model.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'optimizer': optimizer_state_dict,
            'settings': self.training_args
        }
        torch.save(checkpoint, os.path.join(self.training_args.output_dir, 'chmm_model.bin'))

    def load_model(self, path=None):
        if path is None:
            checkpoint = torch.load(os.path.join(self.training_args.output_dir, 'chmm_model.bin'))
        else:
            checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict([checkpoint['optimizer']])
        return checkpoint['settings']

    def append_dataset_obs(self, dataset: Dataset, obs: List[np.ndarray]):
        # --- update dataset ---
        ori_obs = dataset.obs
        assert len(ori_obs) == len(obs)

        new_obs = list()
        for ori_ob, ob in zip(ori_obs, obs):
            assert len(ori_ob) == len(ob)

            # make sure bert observation does not accumulate
            if not self.has_appended_obs:
                new_ob = torch.cat((ori_ob, torch.tensor(ob).unsqueeze(1)), dim=1)
            else:
                ori_ob[:, -1, :] = torch.tensor(ob)
                new_ob = ori_ob
            new_obs.append(new_ob)

        dataset.obs = new_obs

        # --- update arguments ---
        self.training_args.n_src = dataset.obs[0].size(1)

        if 'added_bert' not in self.data_args.src_to_keep:
            self.data_args.src_to_keep += ['added_bert']
            self.data_args.src_priors['added_bert'] = {lb: (0.9, 0.9) for lb in self.data_args.lbs}

    @staticmethod
    def update_embs(dataset: Dataset, embs: List[np.ndarray]):
        # --- update dataset ---
        ori_embs = dataset.embs
        assert len(ori_embs) == len(embs)

        new_embs = list()
        for ori_emb, emb in zip(ori_embs, embs):
            assert ori_emb.shape[0] == emb.shape[0] + 1
            assert ori_emb.shape[1] == emb.shape[1]
            new_emb = ori_emb.clone()
            new_emb[1:, :] = torch.from_numpy(emb)
            new_embs.append(new_emb)

        dataset.embs = new_embs
