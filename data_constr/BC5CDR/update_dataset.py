import sys
sys.path.append('../..')

import os
import json
import logging
import argparse
import numpy as np
from datetime import datetime
from seqeval import metrics
from seqeval.scheme import IOB2

from seqlbtoolkit.io import set_logging, logging_args
from seqlbtoolkit.data import label_to_span, span_dict_to_list

from data_constr.src.io import prettify_json

logger = logging.getLogger(__name__)


def parse_args():
    """
    Wrapper function of argument parsing process.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_loc', type=str, default='.',
        help='where to save results'
    )
    parser.add_argument(
        '--log_dir', type=str, default=None,
        help='the directory of the log file'
    )

    args = parser.parse_args()

    return args


def main(args):
    set_logging(args.log_dir)
    logger.setLevel(logging.INFO)
    logger.info(f"Parameters: {args}")

    logger.info('Reading data...')
    with open(os.path.join(args.save_loc, f"train.json"), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(os.path.join(args.save_loc, f"valid.json"), 'r', encoding='utf-8') as f:
        valid_data = json.load(f)
    with open(os.path.join(args.save_loc, f"test.json"), 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    logger.info('Reading metadata...')
    with open(os.path.join(args.save_loc, "meta.json"), 'r', encoding='utf-8') as f:
        meta = json.load(f)

    logger.info('Getting new metadata')

    max_length = 0
    avg_length = 0
    for data in [train_data, valid_data, test_data]:
        for k, v in data.items():
            l_sent = len(v['data']['text'])
            avg_length += l_sent
            if l_sent > max_length:
                max_length = l_sent
    avg_length /= (len(train_data) + len(valid_data) + len(test_data))

    meta['train_size'] = len(train_data)
    meta['valid_size'] = len(valid_data)
    meta['test_size '] = len(test_data)

    meta['max_length'] = max_length
    meta['avg_length'] = avg_length
    meta['num_lf'] = len(meta['lf'])
    meta['num_labels'] = 2 * len(meta['entity_types']) + 1

    # get the performance of each source
    t_lbs = list()
    w_lbs = [[] for _ in range(meta['num_lf'])]
    for k, v in train_data.items():
        t_lbs.append(v['label'])
        for i, w_lb in enumerate(np.asarray(v['weak_labels']).T):
            w_lbs[i].append(w_lb.tolist())

    for k, v in valid_data.items():
        t_lbs.append(v['label'])
        for i, w_lb in enumerate(np.asarray(v['weak_labels']).T):
            w_lbs[i].append(w_lb.tolist())

    for k, v in test_data.items():
        t_lbs.append(v['label'])
        for i, w_lb in enumerate(np.asarray(v['weak_labels']).T):
            w_lbs[i].append(w_lb.tolist())

    rec_src = list()
    src_f1 = dict()
    logger.info(f'Source performance (F1 score)')
    for i, src_name in enumerate(meta['lf']):
        f1 = metrics.f1_score(t_lbs, w_lbs[i], mode='strict', scheme=IOB2)
        logger.info(f'{src_name}: {f1}')
        if f1 > 0.05:
            rec_src.append(src_name)

    meta['lf_f1'] = src_f1

    logger.info(f'The following sources are recommended for model evaluation:\n'
                f'\t{rec_src}')

    meta['lf_rec'] = rec_src
    meta['num_lf_rec'] = len(rec_src)

    logger.info('Saving updated meta...')

    with open(os.path.join(args.save_loc, "meta.json"), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info('Updating dataset label formats')

    for k in train_data:
        lbs = train_data[k]['label']
        wlbs = train_data[k]['weak_labels']

        train_data[k]['label'] = span_dict_to_list(label_to_span(lbs))
        train_data[k]['weak_labels'] = [span_dict_to_list(label_to_span(wlb)) for wlb in np.array(wlbs).T]

    with open(os.path.join(args.save_loc, "train.json"), 'w', encoding='utf-8') as f:
        f.write(prettify_json(json.dumps(train_data, ensure_ascii=False, indent=2)))

    for k in valid_data:
        lbs = valid_data[k]['label']
        wlbs = valid_data[k]['weak_labels']

        valid_data[k]['label'] = span_dict_to_list(label_to_span(lbs))
        valid_data[k]['weak_labels'] = [span_dict_to_list(label_to_span(wlb)) for wlb in np.array(wlbs).T]

    with open(os.path.join(args.save_loc, "valid.json"), 'w', encoding='utf-8') as f:
        f.write(prettify_json(json.dumps(valid_data, ensure_ascii=False, indent=2)))

    for k in test_data:
        lbs = test_data[k]['label']
        wlbs = test_data[k]['weak_labels']

        test_data[k]['label'] = span_dict_to_list(label_to_span(lbs))
        test_data[k]['weak_labels'] = [span_dict_to_list(label_to_span(wlb)) for wlb in np.array(wlbs).T]

    with open(os.path.join(args.save_loc, "test.json"), 'w', encoding='utf-8') as f:
        f.write(prettify_json(json.dumps(test_data, ensure_ascii=False, indent=2)))

    logger.info('Program successfully finished')


if __name__ == '__main__':

    argument = parse_args()

    _time = datetime.now().strftime("%m.%d.%y-%H.%M")
    _current_file_name = os.path.basename(__file__)
    if _current_file_name.endswith('.py'):
        _current_file_name = _current_file_name[:-3]

    # Setup logging
    if argument.log_dir is None:
        argument.log_dir = os.path.join('logs', f'{_current_file_name}', f'{_time}.log')

    set_logging(log_dir=argument.log_dir)
    logging_args(argument)

    main(argument)
