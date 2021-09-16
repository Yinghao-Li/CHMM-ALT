import numpy as np
import torch

from typing import List, Dict, Tuple, Optional, Union
from tokenizations import get_alignments, get_original_spans


def respan(src_tokens: List[str],
           tgt_tokens: List[str],
           src_span: List[tuple]):
    """
    transfer original spans to target spans
    :param src_tokens: source tokens
    :param tgt_tokens: target tokens
    :param src_span: a list of span tuples. The first element in the tuple
    should be the start index and the second should be the end index
    :return: a list of transferred span tuples.
    """
    s2t, _ = get_alignments(src_tokens, tgt_tokens)
    tgt_spans = list()
    for spans in src_span:
        start = s2t[spans[0]][0]
        if spans[1] < len(s2t):
            end = s2t[spans[1]][-1]
        else:
            end = s2t[-1][-1]
        if end == start:
            end += 1
        tgt_spans.append((start, end))

    return tgt_spans


def txt_to_token_span(tokens: List[str],
                      text: str,
                      txt_spans):
    """
    Transfer text-domain spans to token-domain spans
    :param tokens: tokens
    :param text: text
    :param txt_spans: text spans tuples: (start, end, ...)
    :return: a list of transferred span tuples.
    """
    token_indices = get_original_spans(tokens, text)
    if isinstance(txt_spans, list):
        tgt_spans = list()
        for txt_span in txt_spans:
            txt_start = txt_span[0]
            txt_end = txt_span[1]
            start = None
            end = None
            for i, (s, e) in enumerate(token_indices):
                if s <= txt_start < e:
                    start = i
                if s <= txt_end <= e:
                    end = i + 1
                if (start is not None) and (end is not None):
                    break
            assert (start is not None) and (end is not None), ValueError("input spans out of scope")
            tgt_spans.append((start, end))
    elif isinstance(txt_spans, dict):
        tgt_spans = dict()
        for txt_span, v in txt_spans.items():
            txt_start = txt_span[0]
            txt_end = txt_span[1]
            start = None
            end = None
            for i, (s, e) in enumerate(token_indices):
                if s <= txt_start < e:
                    start = i
                if txt_start == e:
                    start = i + 1
                if s <= txt_end <= e:
                    end = i + 1
                if (start is not None) and (end is not None):
                    break
            assert (start is not None) and (end is not None), ValueError("input spans out of scope")
            tgt_spans[(start, end)] = v
    else:
        raise NotImplementedError
    return tgt_spans


def token_to_txt_span(tokens: List[str],
                      text: str,
                      token_spans):
    """
    Transfer text-domain spans to token-domain spans
    :param tokens: tokens
    :param text: text
    :param token_spans: text spans tuples: (start, end, ...)
    :return: a list of transferred span tuples.
    """
    token_indices = get_original_spans(tokens, text)
    tgt_spans = dict()
    for token_span, value in token_spans.items():
        txt_start = token_indices[token_span[0]][0]
        txt_end = token_indices[token_span[1]-1][1]
        tgt_spans[(txt_start, txt_end)] = value
    return tgt_spans


def span_to_label(labeled_spans: Dict[Tuple[int, int], str],
                  tokens: List[str]) -> List[str]:
    """
    Convert entity spans to labels

    Parameters
    ----------
    labeled_spans: labeled span dictionary: {(start, end): label}
    tokens: a list of tokens, used to check if the spans are valid.

    Returns
    -------
    a list of string labels
    """
    if labeled_spans:
        assert list(labeled_spans.keys())[-1][1] <= len(tokens), ValueError("label spans out of scope!")

    labels = ['O'] * len(tokens)
    for (start, end), label in labeled_spans.items():
        if type(label) == list or type(label) == tuple:
            lb = label[0][0]
        else:
            lb = label
        labels[start] = 'B-' + lb
        if end - start > 1:
            labels[start + 1: end] = ['I-' + lb] * (end - start - 1)

    return labels


def label_to_span(labels: List[str],
                  scheme: Optional[str] = 'BIO') -> dict:
    """
    convert labels to spans
    :param labels: a list of labels
    :param scheme: labeling scheme, in ['BIO', 'BILOU'].
    :return: labeled spans, a list of tuples (start_idx, end_idx, label)
    """
    assert scheme in ['BIO', 'BILOU'], ValueError("unknown labeling scheme")

    labeled_spans = dict()
    i = 0
    while i < len(labels):
        if labels[i] == 'O':
            i += 1
            continue
        else:
            if scheme == 'BIO':
                if labels[i][0] == 'B':
                    start = i
                    lb = labels[i][2:]
                    i += 1
                    try:
                        while labels[i][0] == 'I':
                            i += 1
                        end = i
                        labeled_spans[(start, end)] = lb
                    except IndexError:
                        end = i
                        labeled_spans[(start, end)] = lb
                        i += 1
                # this should not happen
                elif labels[i][0] == 'I':
                    i += 1
            elif scheme == 'BILOU':
                if labels[i][0] == 'U':
                    start = i
                    end = i + 1
                    lb = labels[i][2:]
                    labeled_spans[(start, end)] = lb
                    i += 1
                elif labels[i][0] == 'B':
                    start = i
                    lb = labels[i][2:]
                    i += 1
                    try:
                        while labels[i][0] != 'L':
                            i += 1
                        end = i
                        labeled_spans[(start, end)] = lb
                    except IndexError:
                        end = i
                        labeled_spans[(start, end)] = lb
                        break
                    i += 1
                else:
                    i += 1

    return labeled_spans


def span_dict_to_list(span_dict: Dict[Tuple[int], str]):
    """
    convert entity label span dictionaries to span list

    Parameters
    ----------
    span_dict

    Returns
    -------
    span_list
    """
    span_list = list()
    for (s, e), v in span_dict.items():
        span_list.append([s, e, v])
    return span_list


def span_list_to_dict(span_list: List[list]):
    """
    convert entity label span list to span dictionaries

    Parameters
    ----------
    span_list

    Returns
    -------
    span_dict
    """
    span_dict = dict()
    for span in span_list:
        span_dict[(span[0], span[1])] = span[2]
    return span_dict


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


def probs_to_ids(probs: Union[torch.Tensor, np.ndarray]):
    """
    Convert label probability labels to index

    Parameters
    ----------
    probs: label probabilities

    Returns
    -------
    label indices (shape = one_hot_lbs.shape[:-1])
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    lb_ids = probs.argmax(axis=-1)
    return lb_ids


def ids_to_lbs(ids: Union[torch.Tensor, np.ndarray], label_types: List[str]):
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().numpy()
    np_map = np.vectorize(lambda lb: label_types[lb])
    return np_map(ids)


def probs_to_lbs(probs: Union[torch.Tensor, np.ndarray], label_types: List[str]):
    """
    Convert label probability labels to index

    Parameters
    ----------
    probs: label probabilities
    label_types: label types, size = probs.shape[-1]

    Returns
    -------
    labels (shape = one_hot_lbs.shape[:-1])
    """
    np_map = np.vectorize(lambda lb: label_types[lb])
    lb_ids = probs_to_ids(probs)
    return np_map(lb_ids)


def entity_to_bio_labels(entities: List[str]):
    bio_labels = ["O"] + ["%s-%s" % (bi, label) for label in entities for bi in "BI"]
    return bio_labels
