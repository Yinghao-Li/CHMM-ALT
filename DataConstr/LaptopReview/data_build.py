import sys
sys.path.append('../..')

import os
import argparse
import json
import logging
import numpy as np
from datetime import datetime

from DataConstr.Src.wiser.data.dataset_readers import LaptopsDatasetReader
from DataConstr.Src.Data import (
    span_to_label,
    annotate_sent_with_wiser_allennlp,
    linking_to_tagging_annos
)
from DataConstr.Src.WiserAnnotator import laptop_annotators
from DataConstr.Src.Util import set_seed_everywhere
from DataConstr.Src.IO import set_logging

from xml.etree import ElementTree


logger = logging.getLogger(__name__)
_time = datetime.now().strftime("%m.%d.%y-%H.%M")
_current_file_name = os.path.basename(__file__)
if _current_file_name.endswith('.py'):
    _current_file_name = _current_file_name[:-3]


def parse_args():
    """
    Wrapper function of argument parsing process.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--partition', type=str, default='train',
        help='train/valid/test.'
    )
    parser.add_argument(
        '--save_loc', type=str, default='.',
        help='where to save results'
    )
    parser.add_argument(
        '--log_dir', type=str, default=os.path.join('logs', f'{_current_file_name}.{_time}.log'),
        help='the directory of the log file'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='the random seed used to split the training/validation sets'
    )

    args = parser.parse_args()

    return args


def main(args):
    set_seed_everywhere(args.seed)

    set_logging(args.log_dir)
    logger.setLevel(logging.INFO)
    logger.info(f"Parameters: {args}")

    logger.info('Reading data...')
    reader = LaptopsDatasetReader()

    data_path = {
        'train': './data/Laptop_Train_v2.xml',
        'test': './data/Laptops_Test_Data_phaseB.xml'
    }

    src_path = data_path[args.partition]
    src_data = reader.read(src_path)
    docs = list(src_data)

    logger.info("Annotating data...")
    docs = laptop_annotators(docs=docs)

    root = ElementTree.parse(src_path).getroot()
    xml_sents = root.findall("./sentence")

    sentences = list()
    for xml_sent in xml_sents:
        text = xml_sent.find("text").text
        sentences.append(text)

    results = annotate_sent_with_wiser_allennlp(sentences, docs, token_suffix='-TERM')
    src_token_list, src_anno_list, tagging_anno_list, linking_anno_list = results

    normalized_link_anno_list = linking_to_tagging_annos(tagging_anno_list, linking_anno_list)

    # combine link annotations and tag annotations
    combined_anno_list = list()
    for tag_anno, link_anno in zip(tagging_anno_list, normalized_link_anno_list):
        comb_anno = dict()
        for k, v in tag_anno.items():
            comb_anno[f'tag-{k}'] = v
        for k, v in link_anno.items():
            comb_anno[f'link-{k}'] = v
        combined_anno_list.append(comb_anno)

    if args.partition == 'train':
        indices = np.arange(len(src_token_list))
        np.random.shuffle(indices)
        train_partition = len(src_token_list) * 4 // 5

        label_srcs = list(combined_anno_list[0].keys())
        data_list = list()

        train_token_list = list()
        train_anno_list = list()
        train_lb_list = list()
        for i in indices[:train_partition]:
            train_token_list.append(src_token_list[i])
            train_anno_list.append(combined_anno_list[i])
            train_lb_list.append(src_anno_list[i])

        for sent, true_spans, anno_spans in zip(train_token_list, train_lb_list, train_anno_list):

            data_inst = dict()
            anno_list = list()
            lbs = span_to_label(sent, true_spans)
            for src in label_srcs:
                anno_list.append(span_to_label(sent, anno_spans[src]))
            data_inst['label'] = lbs
            data_inst['data'] = {'text': sent}
            data_inst['weak_labels'] = np.asarray(anno_list).T.tolist()

            data_list.append(data_inst)

        data_dict = {i: inst for i, inst in enumerate(data_list)}

        logger.info(['Saving training results...'])
        with open(os.path.join(args.save_loc, "train.json"), 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2)

        label_srcs = list(combined_anno_list[0].keys())
        data_list = list()

        valid_token_list = list()
        valid_anno_list = list()
        valid_lb_list = list()
        for i in indices[train_partition:]:
            valid_token_list.append(src_token_list[i])
            valid_anno_list.append(combined_anno_list[i])
            valid_lb_list.append(src_anno_list[i])

        for sent, true_spans, anno_spans in zip(valid_token_list, valid_lb_list, valid_anno_list):

            data_inst = dict()
            anno_list = list()
            lbs = span_to_label(sent, true_spans)
            for src in label_srcs:
                anno_list.append(span_to_label(sent, anno_spans[src]))
            data_inst['label'] = lbs
            data_inst['data'] = {'text': sent}
            data_inst['weak_labels'] = np.asarray(anno_list).T.tolist()

            data_list.append(data_inst)

        data_dict = {i: inst for i, inst in enumerate(data_list)}

        logger.info('Saving validation results...')
        with open(os.path.join(args.save_loc, f"valid.json"), 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2)

    else:

        label_srcs = list(combined_anno_list[0].keys())
        data_list = list()

        for sent, true_spans, anno_spans in zip(src_token_list, src_anno_list, combined_anno_list):

            data_inst = dict()
            anno_list = list()
            lbs = span_to_label(sent, true_spans)
            for src in label_srcs:
                anno_list.append(span_to_label(sent, anno_spans[src]))
            data_inst['label'] = lbs
            data_inst['data'] = {'text': sent}
            data_inst['weak_labels'] = np.asarray(anno_list).T.tolist()

            data_list.append(data_inst)

        data_dict = {i: inst for i, inst in enumerate(data_list)}

        logger.info('Saving test results...')
        with open(os.path.join(args.save_loc, f"{args.partition}.json"), 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2)

    meta_dict = dict()
    meta_dict['lf'] = label_srcs
    meta_dict['entity_types'] = ['TERM']

    with open(os.path.join(args.save_loc, "meta.json"), 'w', encoding='utf-8') as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=2)

    logger.info('Exit with no error')


if __name__ == '__main__':
    argument = parse_args()
    main(argument)
