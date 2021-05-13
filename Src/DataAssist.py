import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
from tokenizations import get_alignments, get_original_spans
from Src.Constants import CoNLL_SOURCE_NAMES, OntoNotes_INDICES, CoNLL_SOURCE_PRIORS, OntoNotes_BIO,\
    OUT_RECALL, OUT_PRECISION
from Src.CHMM.CHMMData import Dataset, collate_fn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader


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
                  labeled_spans: Dict[Tuple[int, int], Any]) -> List[str]:
    """
    Convert label spans to
    :param tokens: a list of tokens
    :param labeled_spans: a list of tuples (start_idx, end_idx, label)
    :return: a list of string labels
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


def annotate_doc_with_spacy(sents, spacy_doc):
    sent_level_annos = list()
    for i, (src_sent, spacy_sent) in enumerate(zip(sents, spacy_doc.sents)):

        spacy_tokens = [t.text for t in spacy_sent]
        sent_level_anno = dict()

        sent_start_idx = spacy_sent[0].i
        sent_end_idx = spacy_sent[-1].i

        for source in CoNLL_SOURCE_NAMES:
            sent_level_anno[source] = dict()

            for (start, end), vals in spacy_doc.user_data['annotations'][source].items():
                # convert document-level annotation to sentence-level
                if start >= sent_start_idx and end <= sent_end_idx:
                    tgt_start = start - sent_start_idx
                    tgt_end = end - sent_start_idx

                    src_span = respan(spacy_tokens, src_sent, [(tgt_start, tgt_end)])
                    sent_level_anno[source][src_span[0]] = vals
        sent_level_annos.append(sent_level_anno)
    return sent_level_annos


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
        assert source in annotations, print(f"[ERROR] source name {source} is not included in the data")
        for (start, end), vals in annotations[source].items():
            for label, conf in vals:
                # Such condition should not exist
                if label in {"MISC", "ENT"} and ontonote_anno_scheme:
                    continue
                elif start >= len(sent):
                    print("[Warning] Encountered incorrect annotation boundary")
                    continue
                elif end > len(sent):
                    print("[Warning] Encountered incorrect annotation boundary")
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
    print("Constructing start distribution prior...")

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

    print("Constructing transition matrix prior...")
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

    print("Constructing emission probabilities...")

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


def annotate_data(model, text, embs, obs, lbs, args):
    dataset = Dataset(text=text, embs=embs, obs=obs, lbs=lbs)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=args.pin_memory,
        drop_last=False
    )

    model.eval()
    score_list = list()
    span_list = list()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            # get data
            emb_batch, obs_batch, seq_lens = map(lambda x: x.to(args.device), batch[:3])
            # get prediction
            # the scores are shifted back, i.e., len = len(emb)-1 = len(sentence)
            _, (scored_spans, scores) = model.annotate(
                emb=emb_batch, obs=obs_batch, seq_lengths=seq_lens, label_set=args.bio_lbs,
                normalize_observation=args.obs_normalization
            )
            score_list += scores
            span_list += scored_spans
    return span_list, score_list
