import torch
import random
import json
import numpy as np
import torch.nn as nn
from scipy.special import softmax
from typing import List, Dict
from Src.Constants import OntoNotes_BIO
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import EvalPrediction


def one_hot(x, n_class=None):
    """
    x : LongTensor of shape (batch size, sequence max_seq_length)
    n_class : integer

    Convert batch of integer letter indices to one-hot vectors of dimension S (# of possible x).
    """

    if n_class is None:
        n_class = np.max(x) + 1
    one_hot_vec = np.zeros([int(np.prod(x.shape)), n_class])
    indices = x.reshape([-1])
    one_hot_vec[np.arange(len(indices)), indices] = 1.0
    one_hot_vec = one_hot_vec.reshape(list(x.shape) + [n_class])
    return one_hot_vec


def one_hot_to_string(x):
    """
    obs : Tensor of shape (lengths, |obs_set|)
    S : list of characters (alphabet, obs_set or Sy)
    """

    return [c for c in x.max(dim=1)[1]]


def first_nonzero_idx(x, dim=-1):
    indices = (x == 0).sum(dim=dim)
    return indices


# noinspection PyUnresolvedReferences
def set_seed_everywhere(seed, cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    return None


def log_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    a : m obs n
    b : n obs p

    output : m obs p matrix

    Normally, a matrix multiplication
    computes out_{i,j} = sum_k A_{i,k} obs B_{k,j}

    A log domain matrix multiplication
    computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{k,j}

    This is needed for numerical stability when A and B are probability matrices.
    """
    a1 = a.unsqueeze(-1)
    b1 = b.unsqueeze(-3)
    return (a1 + b1).logsumexp(-2)


def log_maxmul(a, b):
    a1 = a.unsqueeze(-1)
    b1 = b.unsqueeze(-3)
    return (a1 + b1).max(-2)


def validate_prob(x, dim=-1):
    if (x <= 0).any():
        prob = normalize(x, dim=dim)
    elif (x.sum(dim=dim) != 1).any():
        prob = x / x.sum(dim=dim, keepdim=True)
    else:
        prob = x
    return prob


def normalize(x, dim=-1, epsilon=1e-6):
    result = x - x.min(dim=dim, keepdim=True)[0] + epsilon
    result = result / result.sum(dim=dim, keepdim=True)
    return result


# noinspection PyTypeChecker
def logsumexp(x, dim=None, keepdim=False):
    if dim is None:
        x, dim = x.view(-1), 0
    xm, _ = torch.max(x, dim=dim, keepdim=True)
    x = torch.where(
        (xm == np.inf) | (xm == -np.inf),
        xm,
        xm + torch.logsumexp(x - xm, dim=dim, keepdim=True)
    )
    return x if keepdim else x.squeeze(dim)


def construct_length_mask(seq_lengths):
    """
    construct sequence length mask to rule out padding terms.
    :param seq_lengths: a list or 1D Tensor containing the lengths of the sequences in a batch
    :return: a 2D Tensor containing the mask
    """

    max_sequence_length = max(seq_lengths)
    mask = torch.zeros([len(seq_lengths), max_sequence_length]).bool()
    for line, length in zip(mask, seq_lengths):
        line[: length] = True
    return mask


def load_labels(file_name):
    with open(file_name, 'r') as f:
        labels = json.load(f)
    return labels


def load_conll_2003_data(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()

    all_sentence = list()
    sentence = list()
    all_labels = list()
    labels = list()
    for line in lines:
        try:
            token, _, _, ner_label = line.strip().split()
            sentence.append(token)
            labels.append(ner_label)
        except ValueError:
            all_sentence.append(sentence)
            all_labels.append(labels)
            sentence = list()
            labels = list()

    for sentence, labels in zip(all_sentence, all_labels):
        assert len(sentence) == len(labels)

    return all_sentence, all_labels


def check_outputs(predictions):
    """Checks whether the output is consistent"""
    prev_bio_label = "O"
    for i in range(len(predictions)):
        bio_label = OntoNotes_BIO[predictions[i]]
        if prev_bio_label[0] == "O" and bio_label[0] == "I":
            print("inconsistent start of NER at pos %i:" % i, bio_label, "after", prev_bio_label)
        elif prev_bio_label[0] in {"B", "I"}:
            if bio_label[0] not in {"I", "O"}:
                print("inconsistent continuation of NER at pos %i:" % i, bio_label, "after", prev_bio_label)
            if bio_label[0] == "I" and bio_label[2:] != prev_bio_label[2:]:
                print("inconsistent continuation of NER at pos %i:" % i, bio_label, "after", prev_bio_label)
        prev_bio_label = bio_label


def get_results(pred_spans: List[dict],
                true_spans: List[dict],
                sents: List[List[str]],
                all_labels: list):
    """Computes the usual metrics (precision, recall, F1, cross-entropy) on the dataset, using the spacy entities
    in each document as gold standard, and the annotations of a given lb_source as the predicted values"""

    # We start by computing the TP, FP and FN values
    tok_tp = dict()
    tok_fp = dict()
    tok_fn = dict()

    ent_tp = dict()
    ent_fp = dict()
    ent_fn = dict()
    ent_support = dict()
    tok_support = dict()

    for pred_span, true_span, sent in zip(pred_spans, true_spans, sents):

        # We may need to do some mapping / filtering on the entities (eg. mapping PERSON to PER),
        # depending on the corpus we are dealing with

        for label in all_labels:
            lb_true_spans = {tuple(k) for k, v in true_span.items() if v == label}
            lb_pred_spans = {tuple(k) for k, v in pred_span.items() if v == label}

            # Normalisation of dates (with or without prepositions / articles)
            if label == "DATE":
                lb_true_spans = {
                    (start + 1 if sent[start].lower() in {"in", "on", "a", "the", "for", "an", "at"} else start, end)
                    for (start, end) in lb_true_spans
                }
                lb_pred_spans = {
                    (start + 1 if sent[start].lower() in {"in", "on", "a", "the", "for", "an", "at"} else start, end)
                    for (start, end) in lb_pred_spans
                }

            ent_tp[label] = ent_tp.get(label, 0) + len(lb_true_spans.intersection(lb_pred_spans))
            ent_fp[label] = ent_fp.get(label, 0) + len(lb_pred_spans - lb_true_spans)
            ent_fn[label] = ent_fn.get(label, 0) + len(lb_true_spans - lb_pred_spans)
            ent_support[label] = ent_support.get(label, 0) + len(lb_true_spans)

            true_tok_labels = {i for start, end in lb_true_spans for i in range(start, end)}
            pred_tok_labels = {i for start, end in lb_pred_spans for i in range(start, end)}
            tok_tp[label] = tok_tp.get(label, 0) + len(true_tok_labels.intersection(pred_tok_labels))
            tok_fp[label] = tok_fp.get(label, 0) + len(pred_tok_labels - true_tok_labels)
            tok_fn[label] = tok_fn.get(label, 0) + len(true_tok_labels - pred_tok_labels)
            tok_support[label] = tok_support.get(label, 0) + len(true_tok_labels)

    # We then compute the metrics themselves
    results = {}
    for label in ent_support:
        ent_pred = ent_tp[label] + ent_fp[label] + 1E-10
        ent_true = ent_tp[label] + ent_fn[label] + 1E-10
        tok_pred = tok_tp[label] + tok_fp[label] + 1E-10
        tok_true = tok_tp[label] + tok_fn[label] + 1E-10
        results[label] = {}
        results[label]["entity_precision"] = round(ent_tp[label] / ent_pred, 4)
        results[label]["entity_recall"] = round(ent_tp[label] / ent_true, 4)
        results[label]["token_precision"] = round(tok_tp[label] / tok_pred, 4)
        results[label]["token_recall"] = round(tok_tp[label] / tok_true, 4)

        ent_f1_numerator = (results[label]["entity_precision"] * results[label]["entity_recall"])
        ent_f1_denominator = (results[label]["entity_precision"] + results[label]["entity_recall"]) + 1E-10
        results[label]["entity_f1"] = 2 * round(ent_f1_numerator / ent_f1_denominator, 4)

        tok_f1_numerator = (results[label]["token_precision"] * results[label]["token_recall"])
        tok_f1_denominator = (results[label]["token_precision"] + results[label]["token_recall"]) + 1E-10
        results[label]["token_f1"] = 2 * round(tok_f1_numerator / tok_f1_denominator, 4)

    results["macro"] = {"entity_precision": np.round(np.mean([results[lb]["entity_precision"] for lb in ent_support]),
                                                     4),
                        "entity_recall": np.round(np.mean([results[lb]["entity_recall"] for lb in ent_support]), 4),
                        "token_precision": np.round(np.mean([results[lb]["token_precision"] for lb in ent_support]), 4),
                        "token_recall": np.round(np.mean([results[lb]["token_recall"] for lb in ent_support]), 4)}

    label_weights = {lb: ent_support[lb] / (sum(ent_support.values()) + 1E-10) for lb in ent_support}
    results["label_weights"] = label_weights
    results["weighted"] = {"entity_precision": np.round(np.sum([results[lb]["entity_precision"] * label_weights[lb]
                                                                for lb in ent_support]), 4),
                           "entity_recall": np.round(np.sum([results[lb]["entity_recall"] * label_weights[lb]
                                                             for lb in ent_support]), 4),
                           "token_precision": np.round(np.sum([results[lb]["token_precision"] * label_weights[lb]
                                                               for lb in ent_support]), 4),
                           "token_recall": np.round(np.sum([results[lb]["token_recall"] * label_weights[lb]
                                                            for lb in ent_support]), 4)}

    ent_pred = sum([ent_tp[lb] for lb in ent_support]) + sum([ent_fp[lb] for lb in ent_support]) + 1E-10
    ent_true = sum([ent_tp[lb] for lb in ent_support]) + sum([ent_fn[lb] for lb in ent_support]) + 1E-10
    tok_pred = sum([tok_tp[lb] for lb in ent_support]) + sum([tok_fp[lb] for lb in ent_support]) + 1E-10
    tok_true = sum([tok_tp[lb] for lb in ent_support]) + sum([tok_fn[lb] for lb in ent_support]) + 1E-10
    results["micro"] = {"entity_precision": round(sum([ent_tp[lb] for lb in ent_support]) / ent_pred, 4),
                        "entity_recall": round(sum([ent_tp[lb] for lb in ent_support]) / ent_true, 4),
                        "token_precision": round(sum([tok_tp[lb] for lb in ent_support]) / tok_pred, 4),
                        "token_recall": round(sum([tok_tp[lb] for lb in ent_support]) / tok_true, 4)}

    for metric in ["macro", "weighted", "micro"]:
        ent_f1_numerator = (results[metric]["entity_precision"] * results[metric]["entity_recall"])
        ent_f1_denominator = (results[metric]["entity_precision"] + results[metric]["entity_recall"]) + 1E-10
        results[metric]["entity_f1"] = 2 * round(ent_f1_numerator / ent_f1_denominator, 4)

        tok_f1_numerator = (results[metric]["token_precision"] * results[metric]["token_recall"])
        tok_f1_denominator = (results[metric]["token_precision"] + results[metric]["token_recall"]) + 1E-10
        results[metric]["token_f1"] = 2 * round(tok_f1_numerator / tok_f1_denominator, 4)

    return results


def anno_space_map(spans, mappings, tgt_space):
    norm_spans = dict()
    for span, v in spans.items():
        if isinstance(v, str):
            norm_v = mappings.get(v, v)
            if norm_v in tgt_space:
                norm_spans[span] = norm_v
        elif isinstance(v, tuple) or isinstance(v, list):
            norm_v = mappings.get(v[0][0], v[0][0])
            if norm_v in tgt_space:
                norm_spans[span] = ((norm_v, v[0][1]),)
    return norm_spans


def align_predictions(predictions: np.ndarray,
                      label_ids: np.ndarray,
                      label_map: dict):
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list, out_label_list


def compute_metrics(p: EvalPrediction, label_map: Dict[int, str]) -> Dict:
    preds_list, out_label_list = align_predictions(p.predictions, p.label_ids, label_map=label_map)
    return {
        "accuracy_score": accuracy_score(out_label_list, preds_list),
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }


def soft_frequency(logits, power=2, probs=False):
    """
    Unsupervised Deep Embedding for Clustering Analysis
    https://arxiv.org/abs/1511.06335
    """
    if not probs:
        p = softmax(logits, axis=-1)
        p[p != p] = 0
    else:
        p = logits
    f = np.sum(p, axis=0, keepdims=True)
    p = p ** power / f
    p = p / (np.sum(p, axis=-1, keepdims=True) + 1e-9)
    p[p != p] = 0

    return p
