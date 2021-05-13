import torch
from typing import List, Optional, Tuple
from tokenizations import get_alignments, get_original_spans
from Core.Constants import *
from Core.Util import get_overlaps


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
                      txt_spans: List[tuple]):
    """
    Transfer text-domain spans to token-domain spans
    :param tokens: tokens
    :param text: text
    :param txt_spans: text spans tuples: (start, end, ...)
    :return: a list of transferred span tuples.
    """
    token_indices = get_original_spans(tokens, text)
    tgt_spans = list()
    for txt_span in txt_spans:
        spacy_start = txt_span[0]
        spacy_end = txt_span[1]
        start = None
        end = None
        for i, (s, e) in enumerate(token_indices):
            if s <= spacy_start < e:
                start = i
            if s <= spacy_end <= e:
                end = i + 1
            if (start is not None) and (end is not None):
                break
        assert (start is not None) and (end is not None), ValueError("input spans out of scope")
        tgt_spans.append((start, end))
    return tgt_spans


def span_to_label(tokens: List[str],
                  labeled_spans: List[Tuple[int, int, str]],
                  scheme: Optional[str] = 'BIO') -> List[str]:
    """
    Convert label spans to
    :param tokens: a list of tokens
    :param labeled_spans: a list of tuples (start_idx, end_idx, label)
    :param scheme: labeling scheme, in ['BIO', 'BILOU'].
    :return: a list of string labels
    """
    assert scheme in ['BIO', 'BILOU'], ValueError("unknown labeling scheme")
    assert labeled_spans[-1][1] <= len(tokens), ValueError("label spans out of scope!")

    labels = ['O'] * len(tokens)
    for start, end, label in labeled_spans:
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

        for source in SOURCE_NAMES:
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

        # TODO: using the embedding of [CLS] may not be the best idea
        # It does not matter since that embedding is not used in the training
        emb_list = [last_hidden_states[0, :]] + emb_list
        bert_emb = torch.stack(emb_list)
        bert_embs.append(bert_emb)
    return bert_embs


def specialise_annotations(annotations, source_indices_to_keep=SOURCE_NAMES):
    """
    Replace generic ENT or MISC values with the most likely labels from other annotators
    """

    to_add = []

    for source in annotations:

        other_sources = [s for s in annotations if "HMM" not in s
                         and s != "gold" and s in SOURCE_NAMES
                         and s != source and "proper" not in s and "nnp_" not in s
                         and "SEC" not in s
                         and "compound" not in s and "BTC" not in s
                         and SOURCE_NAMES.index(s) in source_indices_to_keep]

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
                                    conf2 = conf2 * SOURCE_PRIORS[other_source][label2][0]
                                    label_counts[label2] = label_counts.get(label2, 0) + conf * conf2
                    vals = tuple((lb, SOURCE_PRIORS[source][lb][0] * conf2 / sum(label_counts.values()))
                                 for lb, conf2 in label_counts.items())
                    to_add.append((source, start, end, vals))

    for source, start, end, vals in to_add:
        annotations[source][(start, end)] = vals

    return annotations


def extract_sequence(sent, annotations, source_indices_to_keep=SOURCE_NAMES):
    """
    Convert the annotations of a spacy document into an array of observations of shape
    (nb_sources, nb_bio_labels)
    """
    annotations = specialise_annotations(annotations)
    sequence = torch.zeros([len(sent), len(SOURCE_NAMES), len(POSITIONED_LABELS_BIO)], dtype=torch.float)
    for i, source in enumerate(SOURCE_NAMES):
        sequence[:, i, 0] = 1.0
        if source not in annotations or i not in source_indices_to_keep:
            continue
        for (start, end), vals in annotations[source].items():
            for label, conf in vals:
                # Such condition should not exist
                if label in {"MISC", "ENT"}:
                    continue
                elif start >= len(sent):
                    print("[Warning] Encountered incorrect annotation boundary")
                    continue
                elif end > len(sent):
                    print("[Warning] Encountered incorrect annotation boundary")
                    end = len(sent)
                sequence[start:end, i, 0] = 0.0

                sequence[start, i, LABEL_INDICES["B-%s" % label]] = conf
                if end - start > 1:
                    sequence[start + 1: end, i, LABEL_INDICES["I-%s" % label]] = conf

    return sequence
