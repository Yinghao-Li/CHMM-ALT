import itertools
import hmmlearn
import hmmlearn.hmm
import numpy as np

from tqdm.auto import tqdm
from Core.Annotate import UnifiedAnnotator
from Core.Constants import *
from Core.IO import docbin_reader
from Core.Data import extract_sequence
from numba import njit, prange


# noinspection PyProtectedMember,PyArgumentList,PyTypeChecker
class HMMAnnotator(hmmlearn.hmm._BaseHMM, UnifiedAnnotator):

    def __init__(self, sources_to_keep=None, source_name="HMM", informative_priors=True):
        hmmlearn.hmm._BaseHMM.__init__(self, len(POSITIONED_LABELS_BIO), verbose=True, n_iter=10)
        UnifiedAnnotator.__init__(self, sources_to_keep=sources_to_keep, source_name=source_name)
        self.informative_priors = informative_priors

    def train(self, docbin_file, cutoff=None):
        """Train the HMM annotator based on the docbin file"""

        spacy_docs = docbin_reader(docbin_file, cutoff=cutoff)
        x_stream = (extract_sequence(doc) for doc in spacy_docs)
        streams = itertools.tee(x_stream, 3)  # 应该是搞 data parallel 的
        self._initialise_startprob(streams[0])
        self._initialise_transmat(streams[1])
        self._initialise_emissions(streams[2])
        self._check()

        self.monitor_._reset()
        for i in range(self.n_iter):
            print("Starting iteration", (i + 1))
            stats = self._initialize_sufficient_statistics()
            curr_logprob = 0

            nb_docs = 0
            for doc in tqdm(docbin_reader(docbin_file, cutoff=cutoff)):
                x = self.extract_sequence(doc)
                # TODO: This function is re-implemented by the author
                # Actually this could be moved outside the loop (NO, unless we get all x in advance)
                framelogprob = self._compute_log_likelihood(x)
                if framelogprob.max(axis=1).min() < -100000:
                    print("problem found!")
                    return framelogprob
                # forward-backward training part
                # TODO: this is a hmmlearn function
                logprob, fwdlattice = self._do_forward_pass(framelogprob)
                curr_logprob += logprob
                # TODO: these are hmmlearn functions
                bwdlattice = self._do_backward_pass(framelogprob)
                posteriors = self._compute_posteriors(fwdlattice, bwdlattice)
                self._accumulate_sufficient_statistics(
                    stats, x, framelogprob, posteriors, fwdlattice,
                    bwdlattice)
                nb_docs += 1

            print("Finished E-step with %i documents" % nb_docs)

            # XXX must be before convergence check, because otherwise
            #     there won't be any updates for the case ``n_iter=1``.
            self._do_mstep(stats)

            self.monitor_.report(curr_logprob)
            if self.monitor_.converged:
                break

        return self

    def label(self, doc):
        """Makes a list of predicted labels (using Viterbi) for each token, along with
        the associated probability according to the HMM model."""

        if not hasattr(self, "emission_probs"):
            raise RuntimeError("Model is not yet trained")

        doc.user_data["annotations"][self.source_name] = {}

        sequence = self.extract_sequence(doc)
        framelogprob = self._compute_log_likelihood(sequence)
        logprob, predicted = self._do_viterbi_pass(framelogprob)
        self.check_outputs(predicted)

        labels = [POSITIONED_LABELS_BIO[x] for x in predicted]

        predicted_proba = np.exp(framelogprob)
        predicted_proba = predicted_proba / predicted_proba.sum(axis=1)[:, np.newaxis]

        confidences = np.array([probs[x] for (probs, x) in zip(predicted_proba, predicted)])
        return labels, confidences

    def _initialise_startprob(self, x_stream):

        print("Constructing start distribution prior...")

        init_counts = np.zeros((len(POSITIONED_LABELS_BIO),))

        if self.informative_priors:
            source_with_best_coverage = sorted(SOURCE_NAMES, key=lambda x: len(SOURCE_PRIORS[x]))[-1]
            print("Using source", source_with_best_coverage, "to estimate start probability priors")
            source_index = SOURCE_NAMES.index(source_with_best_coverage)
            for X in x_stream:
                init_counts[X[0, source_index].argmax()] += 1

        for i, label in enumerate(POSITIONED_LABELS_BIO):
            if i == 0 or label.startswith("B-"):
                init_counts[i] += 1

        self.startprob_prior = init_counts + 1
        self.startprob_ = np.random.dirichlet(init_counts + 1E-10)

    def _initialise_transmat(self, x_stream):

        print("Constructing transition matrix prior...")
        trans_counts = np.zeros((len(POSITIONED_LABELS_BIO), len(POSITIONED_LABELS_BIO)))

        # initialize transition matrix by the labeling source with the largest coverage
        if self.informative_priors:
            source_with_best_coverage = sorted(SOURCE_NAMES, key=lambda v: len(SOURCE_PRIORS[v]))[-1]
            source_index = SOURCE_NAMES.index(source_with_best_coverage)
            for x in x_stream:
                for k in range(0, len(x) - 1):
                    trans_counts[x[k, source_index].argmax(), x[k + 1, source_index].argmax()] += 1

        # update transition matrix with prior knowledge
        for i, label in enumerate(POSITIONED_LABELS_BIO):
            if label.startswith("B-") or label.startswith("I-"):
                trans_counts[i, POSITIONED_LABELS_BIO.index("I-" + label[2:])] += 1
            elif i == 0 or label.startswith("I-"):
                for j, label2 in enumerate(POSITIONED_LABELS_BIO):
                    if j == 0 or label2.startswith("B-"):
                        trans_counts[i, j] += 1

        self.transmat_prior = trans_counts + 1
        # initialize transition matrix with dirichlet distribution
        self.transmat_ = np.vstack([np.random.dirichlet(trans_counts2 + 1E-10)
                                    for trans_counts2 in trans_counts])

    def _initialise_emissions(self, x_stream, strength=1000):

        print("Constructing emission probabilities...")

        obs_counts = np.zeros((len(SOURCE_NAMES), len(POSITIONED_LABELS_BIO)), dtype=np.float64)
        # extract the total number of observations for each prior
        if self.informative_priors:
            for x in x_stream:
                obs_counts += x.sum(axis=0)
        for source_index, source in enumerate(SOURCE_NAMES):
            # increase p(O)
            obs_counts[source_index, 0] += 1
            # increase the "reasonable" observations
            for pos_index, pos_label in enumerate(POSITIONED_LABELS_BIO[1:]):
                if pos_label[2:] in SOURCE_PRIORS[source]:
                    obs_counts[source_index, pos_index] += 1
        # construct probability distribution from counts
        obs_probs = obs_counts / obs_counts.sum(axis=1, keepdims=True)

        # initialize emission matrix
        matrix = np.zeros((len(SOURCE_NAMES), len(POSITIONED_LABELS_BIO), len(POSITIONED_LABELS_BIO)))

        for source_index, source in enumerate(SOURCE_NAMES):
            for pos_index, pos_label in enumerate(POSITIONED_LABELS_BIO):

                # Simple case: set P(O=x|Y=x) to be the recall
                recall = 0
                if pos_index == 0 or not self.informative_priors:
                    recall = OUT_RECALL
                elif pos_label[2:] in SOURCE_PRIORS[source]:
                    _, recall = SOURCE_PRIORS[source][pos_label[2:]]
                matrix[source_index, pos_index, pos_index] = recall

                for pos_index2, pos_label2 in enumerate(POSITIONED_LABELS_BIO):
                    if pos_index2 == pos_index:
                        continue
                    elif pos_index2 == 0 or not self.informative_priors:
                        precision = OUT_PRECISION
                    elif pos_label2[2:] in SOURCE_PRIORS[source]:
                        precision, _ = SOURCE_PRIORS[source][pos_label2[2:]]
                    else:
                        precision = 1.0

                    # Otherwise, we set the probability to be inversely proportional to the precision
                    # and the (unconditional) probability of the observation
                    error_prob = (1 - recall) * (1 - precision) * (0.001 + obs_probs[source_index, pos_index2])

                    # We increase the probability for boundary errors (i.e. I-ORG -> B-ORG)
                    if self.informative_priors and pos_index > 0 and pos_index2 > 0 and pos_label[2:] == pos_label2[2:]:
                        error_prob *= 5

                    # We increase the probability for errors with same boundary (i.e. I-ORG -> I-GPE)
                    if self.informative_priors and pos_index > 0 and pos_index2 > 0 and pos_label[0] == pos_label2[0]:
                        error_prob *= 2

                    matrix[source_index, pos_index, pos_index2] = error_prob

                error_indices = [i for i in range(len(POSITIONED_LABELS_BIO)) if i != pos_index]
                error_sum = matrix[source_index, pos_index, error_indices].sum()
                matrix[source_index, pos_index, error_indices] /= (error_sum / (1 - recall))

        self.emission_priors = matrix * strength
        self.emission_probs = matrix

    def generate_sample_from_state(self, state):
        result = np.zeros((len(SOURCE_NAMES), len(POSITIONED_LABELS_BIO)), dtype=bool)
        for i in range(len(SOURCE_NAMES)):
            choice = np.random.choice(self.emission_probs.shape[2],
                                      p=self.emission_probs[i, state])
            result[i, choice] = True
        return result

    def _compute_log_likelihood(self, x):

        logsum = np.zeros((len(x), len(POSITIONED_LABELS_BIO)))
        for source_index in range(len(SOURCE_NAMES)):
            if source_index not in self.source_indices_to_keep:
                continue
            probs = np.dot(x[:, source_index, :], self.emission_probs[source_index, :, :].T)
            logsum += np.ma.log(probs).filled(-np.inf)

        # We also add a constraint that the probability of a state is zero is no labelling functions observes it
        # TODO: If no labelling functions observes it, set the emission prob to 0
        # TODO: What is the influence of this trick? It seems infeasible when back-propagation is adopted
        x_all_obs = x.sum(axis=1).astype(bool)
        logsum = np.where(x_all_obs, logsum, -np.inf)

        return logsum

    def _initialize_sufficient_statistics(self) -> dict:
        stats = super(HMMAnnotator, self)._initialize_sufficient_statistics()
        stats['obs'] = np.zeros(self.emission_probs.shape)
        return stats

    def _accumulate_sufficient_statistics(self, stats, x, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(HMMAnnotator, self)._accumulate_sufficient_statistics(
            stats, x, framelogprob, posteriors, fwdlattice, bwdlattice)
        if 'e' in self.params:
            self.sum_posteriors(stats["obs"], x, posteriors)

    def _do_mstep(self, stats):
        super(HMMAnnotator, self)._do_mstep(stats)
        if 'e' in self.params:
            emission_counts = self.emission_priors + stats['obs']
            emission_probs = emission_counts / (emission_counts + 1E-100).sum(axis=2)[:, :, np.newaxis]
            self.emission_probs = np.where(self.emission_probs > 0, emission_probs, 0)

    @staticmethod
    @njit(parallel=True)
    def sum_posteriors(stats, x, posteriors):
        for i in prange(x.shape[0]):
            for source_index in range(x.shape[1]):
                for j in range(x.shape[2]):
                    obs = x[i, source_index, j]
                    if obs > 0:
                        stats[source_index, :, j] += (obs * posteriors[i])

    @staticmethod
    def check_outputs(predictions):
        """Checks whether the output is consistent"""
        prev_bio_label = "O"
        for i in range(len(predictions)):
            bio_label = POSITIONED_LABELS_BIO[predictions[i]]
            if prev_bio_label[0] == "O" and bio_label[0] == "I":
                print("inconsistent start of NER at pos %i:" % i, bio_label, "after", prev_bio_label)
            elif prev_bio_label[0] in {"B", "I"}:
                if bio_label[0] not in {"I", "O"}:
                    print("inconsistent continuation of NER at pos %i:" % i, bio_label, "after", prev_bio_label)
                if bio_label[0] == "I" and bio_label[2:] != prev_bio_label[2:]:
                    print("inconsistent continuation of NER at pos %i:" % i, bio_label, "after", prev_bio_label)
            prev_bio_label = bio_label
