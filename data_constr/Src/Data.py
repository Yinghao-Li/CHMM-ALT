import sys
sys.path.append('../..')

import numpy as np
import torch
import copy
import random
import logging
from typing import List, Optional
from tokenizations import get_alignments, get_original_spans

from data_constr.Src.Constants import (
    CoNLL_SOURCE_NAMES,
    OntoNotes_INDICES,
    CoNLL_SOURCE_PRIORS,
    OntoNotes_BIO,
    OUT_RECALL,
    OUT_PRECISION
)
from tqdm.auto import tqdm
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)


def binary_search(start, end, intervals):
    """Performs a binary search"""
    start_search = 0
    end_search = len(intervals)
    while start_search < (end_search - 1):
        mid = start_search + (end_search - start_search) // 2
        (interval_start, interval_end) = intervals[mid]

        if interval_end <= start:
            start_search = mid
        elif interval_start >= end:
            end_search = mid
        else:
            break
    return start_search, end_search


def get_overlaps(start, end, annotations, sources=None):
    """Returns a list of overlaps (as (start, end, value) between the provided span
    and the existing annotations for the sources"""

    overlaps = []
    for source in (sources if sources is not None else annotations.keys()):
        intervals = list(annotations[source].keys())

        start_search, end_search = binary_search(start, end, intervals)

        for interval_start, interval_end in intervals[start_search:end_search]:
            if start < interval_end and end > interval_start:
                interval_value = annotations[source][(interval_start, interval_end)]
                overlaps.append((interval_start, interval_end, interval_value))

    return overlaps


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
    for idx, token_indice in enumerate(token_indices):
        if token_indice is None:
            # handle none error
            if idx == 0:
                token_indices[idx] = (0, token_indices[idx+1][0] - 1)
            elif idx == len(token_indices) - 1:
                token_indices[idx] = (token_indices[idx-1][1]+1, len(text))
            else:
                token_indices[idx] = (token_indices[idx-1][1]+1, token_indices[idx+1][0]-1)
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


def span_to_label(tokens: List[str],
                  labeled_spans: dict,
                  scheme: Optional[str] = 'BIO') -> List[str]:
    """
    Convert spans to label
    :param tokens: a list of tokens
    :param labeled_spans: a list of tuples (start_idx, end_idx, label)
    :param scheme: labeling scheme, in ['BIO', 'BILOU'].
    :return: a list of string labels
    """
    assert scheme in ['BIO', 'BILOU'], ValueError("unknown labeling scheme")
    if labeled_spans:
        assert list(labeled_spans.keys())[-1][1] <= len(tokens), ValueError("label spans out of scope!")

    labels = ['O'] * len(tokens)
    for (start, end), label in labeled_spans.items():
        if scheme == 'BIO':
            labels[start] = 'B-' + label
            if end - start > 1:
                labels[start + 1: end] = ['I-' + label] * (end - start - 1)
        elif scheme == 'BILOU':
            if end - start == 1:
                labels[start] = 'U-' + label
            else:
                labels[start] = 'B-' + label
                labels[end - 1] = 'L-' + label
                if end - start > 2:
                    labels[start + 1: end - 1] = ['I-' + label] * (end - start - 2)

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
        if labels[i] == 'O' or labels[i] == 'ABS':
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


def annotate_doc_with_spacy(sents, spacy_doc):
    sent_level_annos = list()
    for i, (src_sent, spacy_sent) in enumerate(zip(sents, spacy_doc.sents)):

        spacy_tokens = [t.text for t in spacy_sent]
        sent_level_anno = dict()

        sent_start_idx = spacy_sent[0].i
        sent_end_idx = spacy_sent[-1].i

        for source in spacy_doc.spans:
            sent_level_anno[source] = dict()

            for ent in spacy_doc.spans[source]:
                start = ent.start
                end = ent.end
                lb = ent.label_
                # convert document-level annotation to sentence-level
                if start >= sent_start_idx and end <= sent_end_idx:
                    tgt_start = start - sent_start_idx
                    tgt_end = end - sent_start_idx

                    src_span = respan(spacy_tokens, src_sent, [(tgt_start, tgt_end)])
                    sent_level_anno[source][src_span[0]] = lb
        sent_level_annos.append(sent_level_anno)
    return sent_level_annos


def annotate_sent_with_spacy(sent, spacy_doc):

    spacy_tokens = [t.text for t in spacy_doc]
    sent_level_anno = dict()

    sent_start_idx = spacy_doc[0].i
    sent_end_idx = spacy_doc[-1].i

    for source in spacy_doc.spans:
        sent_level_anno[source] = dict()

        for ent in spacy_doc.spans[source]:
            start = ent.start
            end = ent.end
            lb = ent.label_
            # convert document-level annotation to sentence-level
            if start >= sent_start_idx and end <= sent_end_idx:
                tgt_start = start - sent_start_idx
                tgt_end = end - sent_start_idx

                src_span = respan(spacy_tokens, sent, [(tgt_start, tgt_end)])
                sent_level_anno[source][src_span[0]] = lb

    return sent_level_anno


# noinspection PyTypeChecker
def annotate_sent_with_wiser_allennlp(src_sents, allen_sents, token_suffix: str = ''):
    assert len(src_sents) == len(allen_sents)

    src_token_list = list()
    src_anno_list = list()
    tagging_anno_list = list()
    linking_anno_list = list()

    mapping_dict = {0: 'O', 1: 'I'}

    for src_txt, allen_annos in tqdm(zip(src_sents, allen_sents), total=len(src_sents)):

        # handle the data read from the source text
        src_tokens = word_tokenize(src_txt)
        for i in range(len(src_tokens)):
            if src_tokens[i] == r'``' or src_tokens[i] == r"''":
                src_tokens[i] = r'"'

        allen_tokens = list(map(str, allen_annos['tokens']))

        invalid_token_ids = list()
        for i, t in enumerate(allen_tokens):
            if t.strip() == '':
                invalid_token_ids.append(i)

        allen_tokens = [t for i, t in enumerate(allen_tokens) if i not in invalid_token_ids]

        src_labels = [t for i, t in enumerate(allen_annos['tags']) if i not in invalid_token_ids]
        src_labels = formalize_bio(src_labels, suffix=token_suffix)
        src_spans = label_to_span(src_labels)
        src_spans_ = respan(allen_tokens, src_tokens, src_spans)

        src_annos = dict()
        for span, lb in zip(src_spans_, src_spans.values()):
            src_annos[span] = lb

        src_token_list.append(src_tokens)
        src_anno_list.append(src_annos)

        # handle the data constructed using Allennlp
        weak_anno = dict()

        for src, entity_lbs in allen_annos['WISER_LABELS'].items():
            std_lbs = [t for i, t in enumerate(entity_lbs) if i not in invalid_token_ids]
            std_lbs = formalize_bio(std_lbs, suffix=token_suffix)
            weak_span = label_to_span(std_lbs)

            src_weak_span = respan(allen_tokens, src_tokens, weak_span)
            src_weak_anno = dict()
            for span, lb in zip(src_weak_span, weak_span.values()):
                src_weak_anno[span] = lb

            weak_anno[src] = src_weak_anno
        tagging_anno_list.append(weak_anno)

        linked_dict = dict()
        for src, entity_lbs in allen_annos['WISER_LINKS'].items():
            entity_lbs = [mapping_dict[lb] for lb in entity_lbs]
            std_lbs = [t for i, t in enumerate(entity_lbs) if i not in invalid_token_ids]
            std_lbs = formalize_bio(std_lbs)
            entity_spans = label_to_span(std_lbs)

            complete_span = dict()
            for (start, end), lb in entity_spans.items():
                if start != 0:
                    start = start - 1
                complete_span[(start, end)] = lb
            src_link_span = respan(allen_tokens, src_tokens, complete_span)
            linked_dict[src] = src_link_span
        linking_anno_list.append(linked_dict)

    return src_token_list, src_anno_list, tagging_anno_list, linking_anno_list


def linking_to_tagging_annos(tagging_anno_list, linking_anno_list):

    # convert link annotations to tag annotations
    updated_link_anno_list = list()
    for tag_anno, link_anno in zip(tagging_anno_list, linking_anno_list):
        tag_spans = list()
        for src, spans in tag_anno.items():
            for k, v in spans.items():
                tag_spans.append((set(range(k[0], k[1])), v))

        link_entities = dict()
        for src, spans in link_anno.items():
            valid_spans = dict()
            for span in spans:

                span_set = set(range(span[0], span[1]))
                for tag_span, lb in tag_spans:
                    if span_set.intersection(tag_span):
                        if span in valid_spans.keys():
                            if lb not in valid_spans[span]:
                                valid_spans[span].append(lb)
                        else:
                            valid_spans[span] = [lb]
            valid_anno = dict()
            for sp, lbs in valid_spans.items():
                prob = 1 / len(lbs)
                valid_anno[sp] = [(lb, prob) for lb in lbs]
            link_entities[src] = valid_anno
        updated_link_anno_list.append(link_entities)

    # normalize link annotations
    normalized_link_anno_list = list()

    for updated_link_anno in updated_link_anno_list:
        src_anno_dict = dict()
        for src, annos in updated_link_anno.items():
            anno_dict = dict()
            for k, v in annos.items():
                if len(v) > 1:
                    anno_dict[k] = random.choice(v)[0]
                else:
                    anno_dict[k] = v[0][0]
                assert isinstance(anno_dict[k], str)
            src_anno_dict[src] = anno_dict
        normalized_link_anno_list.append(src_anno_dict)

    return normalized_link_anno_list


def build_bert_emb(sents: List[str],
                   tokenizer,
                   model,
                   device: str):
    bert_embs = list()
    for i, sent in enumerate(sents):

        joint_sent = ' '.join(sent)
        bert_tokens = tokenizer.tokenize(joint_sent)

        input_ids = torch.tensor([tokenizer.encode(joint_sent, add_special_tokens=True)], device=device)
        # calculate BERT last layer embeddings
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0].squeeze(0).to('cpu')
            trunc_hidden_states = last_hidden_states[1:-1, :]

        ori2bert, bert2ori = get_alignments(sent, bert_tokens)

        emb_list = list()
        for idx in ori2bert:
            emb = trunc_hidden_states[idx, :]
            emb_list.append(emb.mean(dim=0))

        # TODO: using the embedding of [CLS] may not be the best idea
        # It does not matter since that embedding is not used in the training
        emb_list = [last_hidden_states[0, :]] + emb_list
        bert_emb = torch.stack(emb_list)
        bert_embs.append(bert_emb)
    return bert_embs


def specialise_annotations(annotations, sources=None):
    """
    Replace generic ENT or MISC values with the most likely labels from other annotators
    """
    if sources is None:
        source_indices_to_keep = {i for i in range(len(CoNLL_SOURCE_NAMES))}
    else:
        source_indices_to_keep = {CoNLL_SOURCE_NAMES.index(s) for s in sources}
    to_add = []

    for source in annotations:

        other_sources = [s for s in annotations if "HMM" not in s
                         and s != "gold" and s in CoNLL_SOURCE_NAMES
                         and s != source and "proper" not in s and "nnp_" not in s
                         and "SEC" not in s
                         and "compound" not in s and "BTC" not in s
                         and CoNLL_SOURCE_NAMES.index(s) in source_indices_to_keep]

        for (start, end), vals in annotations[source].items():
            for label, conf in vals:
                if label in {"ENT", "MISC"}:

                    label_counts = {}
                    for other_source in other_sources:
                        overlaps = get_overlaps(start, end, annotations, [other_source])
                        for (start2, end2, vals2) in overlaps:
                            for label2, conf2 in vals2:
                                if label2 not in {"ENT", "MISC"}:
                                    # simple heuristic for the confidence of the label specialisation
                                    conf2 = conf2 if (start2 == start and end2 == end) else 0.3 * conf2
                                    conf2 = conf2 * CoNLL_SOURCE_PRIORS[other_source][label2][0]
                                    label_counts[label2] = label_counts.get(label2, 0) + conf * conf2
                    vals = tuple((lb, CoNLL_SOURCE_PRIORS[source][lb][0] * conf2 / sum(label_counts.values()))
                                 for lb, conf2 in label_counts.items())
                    to_add.append((source, start, end, vals))

    for source, start, end, vals in to_add:
        annotations[source][(start, end)] = vals

    return annotations


def extract_sequence(sent,
                     annotations,
                     sources=CoNLL_SOURCE_NAMES,
                     label_indices=OntoNotes_INDICES,
                     ontonote_anno_scheme=True):
    """
    Convert the annotations of a spacy document into an array of observations of shape
    (nb_sources, nb_bio_labels)
    """
    if ontonote_anno_scheme:
        annotations = specialise_annotations(annotations)
    sequence = torch.zeros([len(sent), len(sources), len(label_indices)], dtype=torch.float)
    for i, source in enumerate(sources):
        sequence[:, i, 0] = 1.0
        assert source in annotations, logger.error(f"source name {source} is not included in the data")
        for (start, end), vals in annotations[source].items():
            for label, conf in vals:
                # Such condition should not exist
                if label in {"MISC", "ENT"} and ontonote_anno_scheme:
                    continue
                elif start >= len(sent):
                    logger.warning("Encountered incorrect annotation boundary")
                    continue
                elif end > len(sent):
                    logger.warning("Encountered incorrect annotation boundary")
                    end = len(sent)
                sequence[start:end, i, 0] = 0.0

                sequence[start, i, label_indices["B-%s" % label]] = conf
                if end - start > 1:
                    sequence[start + 1: end, i, label_indices["I-%s" % label]] = conf

    return sequence


def initialise_startprob(observations,
                         label_set=OntoNotes_BIO,
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
                        label_set=OntoNotes_BIO,
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


def converse_ontonote_to_conll(args, src_annotations):
    anno_list = list()
    for annotations in src_annotations:
        anno_dict = dict()
        for src, annos in annotations.items():
            if src not in anno_dict:
                anno_dict[src] = dict()
            for span, values in annos.items():
                anno_dict[src][span] = tuple()
                for lb, conf in values:
                    norm_lb = args.mappings.get(lb, lb)
                    if norm_lb in args.lbs:
                        anno_dict[src][span] += ((norm_lb, conf),)
        anno_list.append(anno_dict)
    return anno_list


def formalize_bio(labels: List[str], suffix: str = ''):
    lbs = copy.deepcopy(labels)
    illegal_flag = False
    previous_label = 'O'
    for i, label in enumerate(lbs):
        label = lbs[i]
        if label.startswith('I') and (previous_label == 'O' or previous_label == 'ABS'):
            lbs[i] = label.replace('I', 'B') + suffix
        elif label.startswith('I') and previous_label.startswith('B'):
            curr_type = label[2:]
            try:
                prev_type = previous_label[2:]
            except ValueError:
                prev_type = 'O'
            if not curr_type:
                lbs[i] = label + suffix
            elif curr_type != prev_type:  # this should never happen in normal cases
                illegal_flag = True
                lbs[i] = label.replace('I', 'B') + suffix
            else:
                lbs[i] = label + suffix
        elif label.startswith('I'):
            lbs[i] = label + suffix
        elif label.startswith('B'):
            lbs[i] = label + suffix
        previous_label = lbs[i]
    if illegal_flag:
        logger.warning(f"Illegal input labels: {labels}")
    return lbs
