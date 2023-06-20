import os
import regex
import json
import logging
from typing import Optional
from string import printable

import torch
import numpy as np

from seqlbtoolkit.data import (
    span_to_label,
    span_list_to_dict,
    entity_to_bio_labels,
    one_hot,
)

logger = logging.getLogger(__name__)


def load_data_from_json(file_dir: str, config: Optional = None):
    """
    Load data stored in the current data format.


    Parameters
    ----------
    file_dir: file directory
    config: configuration

    """
    with open(file_dir, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)

    # Load meta if exist
    file_loc = os.path.split(file_dir)[0]
    meta_dir = os.path.join(file_loc, 'meta.json')

    if not os.path.isfile(meta_dir):
        raise FileNotFoundError('Meta file does not exist!')

    with open(meta_dir, 'r', encoding='utf-8') as f:
        meta_dict = json.load(f)

    bio_labels = entity_to_bio_labels(meta_dict['entity_types'])
    label_to_id = {lb: i for i, lb in enumerate(bio_labels)}
    np_map = np.vectorize(lambda lb: label_to_id[lb])

    load_all_sources = config is not None and getattr(config, "load_all_sources", False)
    if 'lf_rec' in meta_dict.keys() and not load_all_sources:
        lf_rec_ids = [meta_dict['lf'].index(lf) for lf in meta_dict['lf_rec']]
    else:
        lf_rec_ids = list(range(meta_dict['num_lf']))

    sentence_list = list()
    lbs_list = list()
    w_lbs_list = list()

    for i in range(len(data_dict)):
        data = data_dict[str(i)]
        # get tokens
        tks = [regex.sub("[^{}]+".format(printable), "", tk) for tk in data['data']['text']]
        sent_tks = ['[UNK]' if not tk else tk for tk in tks]
        sentence_list.append(sent_tks)
        # get true labels
        lbs = span_to_label(span_list_to_dict(data['label']), sent_tks)
        lbs_list.append(lbs)
        # get lf annotations (weak labels) in one-hot format tensor
        w_lbs = [span_to_label(span_list_to_dict(data['weak_labels'][lf_idx]), sent_tks) for lf_idx in lf_rec_ids]
        w_lbs = np_map(np.asarray(w_lbs).T)
        w_lbs_one_hot = one_hot(w_lbs, n_class=len(bio_labels))
        w_lbs_list.append(torch.from_numpy(w_lbs_one_hot).to(dtype=torch.float))

    # update config
    if config:
        config.sources = meta_dict['lf_rec'] \
            if 'lf_rec' in meta_dict.keys() and not load_all_sources else meta_dict['lf']
        config.entity_types = meta_dict['entity_types']
        config.bio_label_types = bio_labels
        if 'priors' in meta_dict.keys():
            config.src_priors = meta_dict['priors']
        else:
            config.src_priors = {src: {lb: (0.7, 0.7) for lb in config.entity_types} for src in config.sources}

    if config and getattr(config, 'debug_mode', False):
        return sentence_list[:100], lbs_list[:100], w_lbs_list[:100]
    return sentence_list, lbs_list, w_lbs_list


def load_data_from_pt(file_dir: str, config: Optional = None):
    """
    Load data that are stored as the previous data format.
    For backward compatibility, should not be used in Wrench

    Parameters
    ----------
    file_dir: file directory
    config: configuration

    """
    data_dict = torch.load(file_dir)

    file_loc, file_name = os.path.split(file_dir)
    meta_dir = os.path.join(file_loc, f'{file_name.split("-")[0]}-metadata.json')

    if not os.path.isfile(meta_dir):
        logger.error('Meta file does not exist!')
        raise FileNotFoundError('Meta file does not exist!')

    with open(meta_dir, 'r', encoding='utf-8') as f:
        meta_dict = json.load(f)

    bio_labels = entity_to_bio_labels(meta_dict['labels'])
    label_to_id = {lb: i for i, lb in enumerate(bio_labels)}

    sentence_list = data_dict['sentences']
    annotation_list = data_dict['annotations']
    label_list = [span_to_label(l, s) for s, l in zip(sentence_list, data_dict['labels'])]

    if 'mapping' in meta_dict.keys():
        annotation_list = convert_conll(annotation_list, meta_dict['labels'], meta_dict['mapping'])

    # meta_dict['sources'] equals "source_to_keep" in the macro file

    load_all_sources = config is not None and getattr(config, "load_all_sources", False)
    sources = list(annotation_list[0].keys()) if load_all_sources else meta_dict['sources']
    weak_label_list = [extract_sequence(
        s, a, sources=sources, label_indices=label_to_id
    ) for s, a in zip(sentence_list, annotation_list)]

    # update config
    if config:
        config.sources = sources
        config.entity_types = meta_dict['labels']
        config.bio_label_types = bio_labels
        if 'mapping' not in meta_dict.keys():
            config.src_priors = meta_dict['priors'] if 'priors' in meta_dict.keys() else None
        else:
            mappings = meta_dict['mapping']
            priors = meta_dict['priors']
            entity_types = meta_dict['labels']

            for src, prs in priors.items():
                entity_dict = {e: list() for e in entity_types}
                for src_ent, pr in prs.items():
                    tgt_ent = mappings.get(src_ent, src_ent)
                    if tgt_ent in entity_types:
                        entity_dict[tgt_ent].append(pr)
                for k, v in entity_dict.items():
                    if v:
                        v = (np.asarray(v)).mean(axis=0)
                        entity_dict[k] = [round(v[0], 2), round(v[1], 2)]
                    else:
                        entity_dict[k] = [1E-2, 1E-2]
                priors[src] = entity_dict
            config.src_priors = priors

    if config and getattr(config, 'debug_mode', False):
        return sentence_list[:100], label_list[:100], weak_label_list[:100]
    return sentence_list, label_list, weak_label_list


def convert_conll(src_annotations, label_types, mappings):
    anno_list = list()
    for annotations in src_annotations:
        anno_dict = dict()
        for src, annos in annotations.items():
            if src not in anno_dict:
                anno_dict[src] = dict()
            for span, values in annos.items():
                anno_dict[src][span] = tuple()
                for lb, conf in values:
                    norm_lb = mappings.get(lb, lb)
                    if norm_lb in label_types:
                        anno_dict[src][span] += ((norm_lb, conf),)
        anno_list.append(anno_dict)
    return anno_list


def extract_sequence(sent,
                     annotations,
                     sources,
                     label_indices):
    """
    Convert the annotations of a spacy document into an array of observations of shape
    (nb_sources, nb_bio_labels)
    """
    sequence = torch.zeros([len(sent), len(sources), len(label_indices)], dtype=torch.float)
    for i, source in enumerate(sources):
        sequence[:, i, 0] = 1.0
        assert source in annotations, logger.error(f"source name {source} is not included in the data")
        for (start, end), vals in annotations[source].items():
            for label, conf in vals:
                if start >= len(sent):
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
