import sys
sys.path.append('../..')

import os
import json
import logging
import argparse
import numpy as np
from datetime import datetime

from data_constr.src.wiser.data.dataset_readers import CDRCombinedDatasetReader
from data_constr.src.data import (
    span_to_label,
    annotate_sent_with_wiser_allennlp,
    linking_to_tagging_annos
)
from data_constr.src.wiser_annotator import bc5cdr_annotators
from data_constr.src.io import load_bc5cdr_sentences, set_logging

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

    args = parser.parse_args()

    return args


# noinspection PyTypeChecker
def main(args):

    set_logging(args.log_dir)
    logger.setLevel(logging.INFO)
    logger.info(f"Parameters: {args}")

    logger.info('Reading data...')
    cdr_reader = CDRCombinedDatasetReader()

    data_path = {
        'train': './data/CDR_TrainingSet.BioC.xml',
        'valid': './data/CDR_DevelopmentSet.BioC.xml',
        'test': './data/CDR_TestSet.BioC.xml'
    }

    src_path = data_path[args.partition]
    src_data = cdr_reader.read(src_path)
    cdr_docs = list(src_data)

    logger.info("Annotating data...")
    cdr_docs = bc5cdr_annotators(docs=cdr_docs, reader=cdr_reader)

    xml_sents = load_bc5cdr_sentences(src_path)

    results = annotate_sent_with_wiser_allennlp(xml_sents, cdr_docs)
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

    meta_dict = dict()
    meta_dict['lf'] = label_srcs
    meta_dict['entity_types'] = ['Chemical', 'Disease']

    logger.info('Saving results...')
    with open(os.path.join(args.save_loc, f"{args.partition}.json"), 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.save_loc, "meta.json"), 'w', encoding='utf-8') as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=2)

    logger.info('Exit with no error')


if __name__ == '__main__':
    argument = parse_args()
    main(argument)
