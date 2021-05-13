import numpy as np
import torch
import torch.nn as nn
from Src.Utils import log_matmul, log_maxmul, validate_prob, logsumexp
from Src.DataAssist import label_to_span
from typing import Optional


class NeuralModule(nn.Module):
    def __init__(self,
                 d_emb,
                 n_hidden,
                 n_src,
                 n_obs):
        super(NeuralModule, self).__init__()

        self.n_hidden = n_hidden
        self.n_src = n_src
        self.n_obs = n_obs
        self.neural_transition = nn.Linear(d_emb, self.n_hidden * self.n_hidden)
        self.neural_emissions = nn.ModuleList([
            nn.Linear(d_emb, self.n_hidden * self.n_obs) for _ in range(self.n_src)
        ])

        self._init_parameters()

    def forward(self,
                embs: torch.Tensor,
                temperature: Optional[int] = 1.0):
        batch_size, max_seq_length, _ = embs.size()
        trans_temp = self.neural_transition(embs).view(
            batch_size, max_seq_length, self.n_hidden, self.n_hidden
        )
        nn_trans = torch.softmax(trans_temp / temperature, dim=-1)

        nn_emiss = torch.stack([torch.softmax(emiss(embs).view(
                batch_size, max_seq_length, self.n_hidden, self.n_obs
            ) / temperature, dim=-1) for emiss in self.neural_emissions]).permute(1, 2, 0, 3, 4)
        return nn_trans, nn_emiss

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.neural_transition.weight.data, gain=nn.init.calculate_gain('relu'))
        for emiss in self.neural_emissions:
            nn.init.xavier_uniform_(emiss.weight.data, gain=nn.init.calculate_gain('relu'))


class ConditionalHMM(nn.Module):

    def __init__(self,
                 args,
                 state_prior=None,
                 trans_matrix=None,
                 emiss_matrix=None,
                 device='cpu'):
        super(ConditionalHMM, self).__init__()

        self.d_emb = args.d_emb  # embedding dimension
        self.n_src = args.n_src
        self.n_obs = args.n_obs  # number of possible obs_set
        self.n_hidden = args.n_hidden  # number of states

        self.trans_weight = args.trans_nn_weight
        self.emiss_weight = args.emiss_nn_weight

        self.device = device

        self.nn_module = NeuralModule(d_emb=self.d_emb, n_hidden=self.n_hidden, n_src=self.n_src, n_obs=self.n_obs)

        # initialize unnormalized state-prior, transition and emission matrices
        self._initialize_model(
            state_prior=state_prior, trans_matrix=trans_matrix, emiss_matrix=emiss_matrix
        )
        self.to(self.device)

    def _initialize_model(self,
                          state_prior: torch.Tensor,
                          trans_matrix: torch.Tensor,
                          emiss_matrix: torch.Tensor
                          ):

        if state_prior is None:
            priors = torch.zeros(self.n_hidden, device=self.device) + 1E-3
            priors[0] = 1
            self.state_priors = nn.Parameter(torch.log(priors))
        else:
            state_prior.to(self.device)
            priors = validate_prob(state_prior, dim=0)
            self.state_priors = nn.Parameter(torch.log(priors))

        if trans_matrix is None:
            self.unnormalized_trans = nn.Parameter(torch.randn(self.n_hidden, self.n_hidden, device=self.device))
        else:
            trans_matrix.to(self.device)
            trans_matrix = validate_prob(trans_matrix)
            # We may want to use softmax later, so we put here a log to counteract the effact
            self.unnormalized_trans = nn.Parameter(torch.log(trans_matrix))

        if emiss_matrix is None:
            self.unnormalized_emiss = nn.Parameter(
                torch.zeros(self.n_src, self.n_hidden, self.n_obs, device=self.device)
            )
        else:
            emiss_matrix.to(self.device)
            emiss_matrix = validate_prob(emiss_matrix)
            # We may want to use softmax later, so we put here a log to counteract the effact
            self.unnormalized_emiss = nn.Parameter(torch.log(emiss_matrix))

        print("[INFO] model initialized!")

        return None

    def _initialize_states(self,
                           embs: torch.Tensor,
                           obs: torch.Tensor,
                           temperature: Optional[int] = 1.0,
                           normalize_observation: Optional[bool] = True):
        # normalize and put the probabilities into the log domain
        batch_size, max_seq_length, n_src, _ = obs.size()
        self.log_state_priors = torch.log_softmax(self.state_priors / temperature, dim=-1)
        trans = torch.softmax(self.unnormalized_trans / temperature, dim=-1)
        emiss = torch.softmax(self.unnormalized_emiss / temperature, dim=-1)

        # get neural transition and emission matrices
        nn_trans, nn_emiss = self.nn_module(embs)

        self.log_trans = torch.log((1-self.trans_weight) * trans + self.trans_weight * nn_trans)
        self.log_emiss = torch.log((1-self.emiss_weight) * emiss + self.emiss_weight * nn_emiss)

        # if at least one source observes an entity at a position, set the probabilities of other sources to
        # the mean value (so that they will not affect the prediction)
        # maybe we can also set them all to 0?
        # [10/20/2020] The current version works fine. No need to change for now.
        # [10/20/2020] Pack this process into an if branch
        if normalize_observation:
            lbs = obs.argmax(dim=-1)
            # at least one source observes an entity
            entity_idx = lbs.sum(dim=-1) > 1E-6
            # the sources that do not observe any entity
            no_entity_idx = lbs <= 1E-6
            no_obs_src_idx = entity_idx.unsqueeze(-1) * no_entity_idx
            subsitute_prob = torch.zeros_like(obs[0, 0, 0])
            subsitute_prob[0] = 0.01
            subsitute_prob[1:] = 0.99 / self.n_obs
            obs[no_obs_src_idx] = subsitute_prob

        # Calculate the emission probabilities in one time, so that we don't have to compute this repeatedly
        # log-domain subtract is regular-domain divide
        self.log_emiss_probs = log_matmul(
            self.log_emiss, torch.log(obs).unsqueeze(-1)
        ).squeeze(-1).sum(dim=-2)

        self.log_alpha = torch.zeros([batch_size, max_seq_length, self.n_hidden], device=self.device)
        self.log_beta = torch.zeros([batch_size, max_seq_length, self.n_hidden], device=self.device)
        # Gamma can be readily computed and need no initialization
        self.log_gamma = None
        # only values in 1:max_seq_length are valid. The first state is a dummy
        self.log_xi = torch.zeros([batch_size, max_seq_length, self.n_hidden, self.n_hidden], device=self.device)
        return None

    def _forward_step(self, t):
        # initial alpha state
        if t == 0:
            log_alpha_t = self.log_state_priors + self.log_emiss_probs[:, t, :]
        # do the forward step
        else:
            log_alpha_t = self.log_emiss_probs[:, t, :] + \
                          log_matmul(self.log_alpha[:, t - 1, :].unsqueeze(1), self.log_trans[:, t, :, :]).squeeze(1)

        # normalize the result
        normalized_log_alpha_t = log_alpha_t - log_alpha_t.logsumexp(dim=-1, keepdim=True)
        return normalized_log_alpha_t

    def _backward_step(self, t):
        # do the backward step
        # beta is not a distribution, so we do not need to normalize it
        log_beta_t = log_matmul(
            self.log_trans[:, t, :, :],
            (self.log_emiss_probs[:, t, :] + self.log_beta[:, t + 1, :]).unsqueeze(-1)
        ).squeeze(-1)
        return log_beta_t

    def _forward_backward(self, seq_lengths):
        max_seq_length = seq_lengths.max().item()
        # calculate log alpha
        for t in range(0, max_seq_length):
            self.log_alpha[:, t, :] = self._forward_step(t)

        # calculate log beta
        # The last beta state beta[:, -1, :] = log1 = 0, so no need to re-assign the value
        for t in range(max_seq_length - 2, -1, -1):
            self.log_beta[:, t, :] = self._backward_step(t)
        # shift the output (since beta is calculated in backward direction,
        # we need to shift each instance in the batch according to its length)
        shift_distances = seq_lengths - max_seq_length
        self.log_beta = torch.stack(
            [torch.roll(beta, s.item(), 0) for beta, s in zip(self.log_beta, shift_distances)]
        )
        return None

    def _compute_xi(self, t):
        temp_1 = self.log_emiss_probs[:, t, :] + self.log_beta[:, t, :]
        temp_2 = log_matmul(self.log_alpha[:, t-1, :].unsqueeze(-1), temp_1.unsqueeze(1))
        log_xi_t = self.log_trans[:, t, :, :] + temp_2
        return log_xi_t

    def _expected_complete_log_likelihood(self, seq_lengths):
        batch_size = len(seq_lengths)
        max_seq_length = seq_lengths.max().item()

        # calculate expected sufficient statistics: gamma_t(j) = P(z_t = j|x_{1:T})
        self.log_gamma = self.log_alpha + self.log_beta
        # normalize as gamma is a distribution
        log_gamma = self.log_gamma - self.log_gamma.logsumexp(dim=-1, keepdim=True)

        # calculate expected sufficient statistics: psi_t(i, j) = P(z_{t-1}=i, z_t=j|x_{1:T})
        for t in range(1, max_seq_length):
            self.log_xi[:, t, :, :] = self._compute_xi(t)
        stabled_norm_term = logsumexp(self.log_xi[:, 1:, :, :].view(batch_size, max_seq_length-1, -1), dim=-1)\
            .view(batch_size, max_seq_length-1, 1, 1)
        log_xi = self.log_xi[:, 1:, :, :] - stabled_norm_term

        # calculate the expected complete data log likelihood
        log_prior = torch.sum(torch.exp(log_gamma[:, 0, :]) * self.log_state_priors, dim=-1)
        log_prior = log_prior.mean()
        # sum over j, k
        log_tran = torch.sum(torch.exp(log_xi) * self.log_trans[:, 1:, :, :], dim=[-2, -1])
        # sum over valid time steps, and then average over batch. Note this starts from t=2
        log_tran = torch.mean(torch.stack([inst[:length].sum() for inst, length in zip(log_tran, seq_lengths-1)]))
        # same as above
        log_emis = torch.sum(torch.exp(log_gamma) * self.log_emiss_probs, dim=-1)
        log_emis = torch.mean(torch.stack([inst[:length].sum() for inst, length in zip(log_emis, seq_lengths)]))
        log_likelihood = log_prior + log_tran + log_emis

        return log_likelihood

    def forward(self, emb, obs, seq_lengths, normalize_observation=True):
        """
        """
        # the row of obs should be one-hot or at least sum to 1
        # assert (obs.sum(dim=-1) == 1).all()

        batch_size, max_seq_length, n_src, n_obs = obs.size()
        assert n_obs == self.n_obs
        assert n_src == self.n_src

        # Initialize alpha, beta and xi
        self._initialize_states(embs=emb, obs=obs, normalize_observation=normalize_observation)
        self._forward_backward(seq_lengths=seq_lengths)
        log_likelihood = self._expected_complete_log_likelihood(seq_lengths=seq_lengths)
        return log_likelihood, (self.log_trans, self.log_emiss)

    def viterbi(self, emb, obs, seq_lengths, normalize_observation=True):
        """
        Find argmax_z log p(z|obs) for each (obs) in the batch.
        """
        batch_size = len(seq_lengths)
        max_seq_length = seq_lengths.max().item()

        # initialize states
        self._initialize_states(embs=emb, obs=obs, normalize_observation=normalize_observation)
        # maximum probabilities
        log_delta = torch.zeros([batch_size, max_seq_length, self.n_hidden], device=self.device)
        # most likely previous state on the most probable path to z_t = j. a[0] is undefined.
        pre_states = torch.zeros([batch_size, max_seq_length, self.n_hidden], dtype=torch.long, device=self.device)

        # the initial delta state
        log_delta[:, 0, :] = self.log_state_priors + self.log_emiss_probs[:, 0, :]
        for t in range(1, max_seq_length):
            # udpate delta and a. It does not matter where we put the emission probabilities
            max_log_prob, argmax_val = log_maxmul(
                log_delta[:, t-1, :].unsqueeze(1),
                self.log_trans[:, t, :, :] + self.log_emiss_probs[:, t, :].unsqueeze(1)
            )
            log_delta[:, t, :] = max_log_prob.squeeze(1)
            pre_states[:, t, :] = argmax_val.squeeze(1)

        # The terminal state
        batch_max_log_prob = list()
        batch_z_t_star = list()

        for l_delta, length in zip(log_delta, seq_lengths):
            max_log_prob, z_t_star = l_delta[length-1, :].max(dim=-1)
            batch_max_log_prob.append(max_log_prob)
            batch_z_t_star.append(z_t_star)

        # Trace back
        batch_z_star = [[z_t_star.item()] for z_t_star in batch_z_t_star]
        for p_states, z_star, length in zip(pre_states, batch_z_star, seq_lengths):
            for t in range(length-2, -1, -1):
                z_t = p_states[t+1, z_star[0]].item()
                z_star.insert(0, z_t)

        # compute the smoothed marginal p(z_t = j | obs_{1:T})
        self._forward_backward(seq_lengths)
        log_marginals = self.log_alpha + self.log_beta
        norm_marginals = torch.exp(log_marginals - logsumexp(log_marginals, dim=-1, keepdim=True))
        batch_marginals = list()
        for marginal, length in zip(norm_marginals, seq_lengths):
            mgn_list = marginal[1:length].detach().cpu().numpy()
            batch_marginals.append(mgn_list)

        return batch_z_star, batch_marginals

    def annotate(self, emb, obs, seq_lengths, label_set, normalize_observation=True):
        batch_label_indices, batch_probs = self.viterbi(
            emb, obs, seq_lengths, normalize_observation=normalize_observation
        )
        batch_labels = [[label_set[lb_index] for lb_index in label_indices]
                        for label_indices in batch_label_indices]

        # For batch_spans, we are going to compare them with the true spans,
        # and the true spans is already shifted, so we do not need to shift predicted spans back
        batch_spans = list()
        batch_scored_spans = list()
        for labels, probs, indices in zip(batch_labels, batch_probs, batch_label_indices):
            spans = label_to_span(labels)
            batch_spans.append(spans)

            ps = [p[s] for p, s in zip(probs, indices[1:])]
            scored_spans = dict()
            for k, v in spans.items():
                if k == (0, 1):
                    continue
                start = k[0] - 1 if k[0] > 0 else 0
                end = k[1] - 1
                score = np.mean(ps[start:end])
                scored_spans[(start, end)] = [(v, score)]
            batch_scored_spans.append(scored_spans)

        return batch_spans, (batch_scored_spans, batch_probs)
