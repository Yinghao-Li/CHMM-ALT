import sys
sys.path.append('../..')

import os
import spacy
import json
import logging
import argparse
import numpy as np
from datetime import datetime

from tqdm.auto import tqdm
from data_constr.Src.IO import load_conll_2003_data, set_logging
from data_constr.Src.SkweakAnnotator import construct_spacy_doc, CoNLL2003Annotator, ConLL2003Standardiser
from data_constr.Src.Data import annotate_doc_with_spacy, span_to_label


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


def main(args):
    set_logging(args.log_dir)
    logger.setLevel(logging.INFO)
    logger.info(f"Parameters: {args}")

    logger.info('Reading data...')

    data_dir = r'./data/'
    data_name = f'{args.partition}.txt'
    articles = load_conll_2003_data(os.path.join(data_dir, data_name))

    nlp = spacy.load('en_core_web_md')
    docs = []
    for article in tqdm(articles):
        sents = article['sent_list']
        doc = construct_spacy_doc(sents, nlp)
        docs.append(doc)

    united_annotator = CoNLL2003Annotator().add_all()

    logger.info('labeling articles...')
    docs = list(united_annotator.pipe(docs))

    standarizer = ConLL2003Standardiser()
    docs = [standarizer(doc) for doc in docs]

    label_srcs = [k for k in docs[0].spans.keys()]
    label_srcs.sort()

    data_list = list()

    logger.info('converting labels...')
    for article, doc in tqdm(zip(articles, docs), total=len(articles)):
        sent_list = article['sent_list']
        lbs_list = article['labels_list']
        sent_annos = annotate_doc_with_spacy(sent_list, doc)
        assert len(sent_list) == len(lbs_list) == len(sent_annos)

        for sent, lbs, annos in zip(sent_list, lbs_list, sent_annos):
            data_inst = dict()
            anno_list = list()
            for src in label_srcs:
                anno_list.append(span_to_label(sent, annos[src]))
            data_inst['label'] = lbs
            data_inst['data'] = {'text': sent}
            data_inst['weak_labels'] = np.asarray(anno_list).T.tolist()

            data_list.append(data_inst)

    data_dict = {i: inst for i, inst in enumerate(data_list)}

    meta_dict = dict()
    meta_dict['lf'] = label_srcs
    meta_dict['entity_types'] = ['PER', 'LOC', 'ORG', 'MISC']

    logger.info('Saving results...')
    with open(os.path.join(args.save_loc, f"{args.partition}.json"), 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.save_loc, "meta.json"), 'w', encoding='utf-8') as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=2)

    logger.info('Exit with no error')


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
