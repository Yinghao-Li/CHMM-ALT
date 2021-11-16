# coding=utf-8
""" Train the conditional hidden Markov model """

import sys
sys.path.append('..')

import logging
import os
import sys
import gc
import torch
from datetime import datetime

from transformers import (
    HfArgumentParser,
    set_seed,
)

from seqlbtoolkit.io import set_logging, logging_args
from seqlbtoolkit.chmm.dataset import CHMMBaseDataset, collate_fn

from label_model.chmm.train import CHMMTrainer
from label_model.chmm.args import CHMMArguments, CHMMConfig

logger = logging.getLogger(__name__)


def chmm_train(args: CHMMArguments):
    # setup logging
    set_logging(log_dir=args.log_dir)

    logging_args(args)
    set_seed(args.seed)
    config = CHMMConfig().from_args(args)

    training_dataset = valid_dataset = test_dataset = None
    if args.train_file:
        logger.info('Loading training dataset...')
        training_dataset = CHMMBaseDataset().load_file(
            file_path=args.train_file,
            config=config
        )
    if args.valid_file:
        logger.info('Loading validation dataset...')
        valid_dataset = CHMMBaseDataset().load_file(
            file_path=args.valid_file,
            config=config
        )
    if args.test_file:
        logger.info('Loading test dataset...')
        test_dataset = CHMMBaseDataset().load_file(
            file_path=args.test_file,
            config=config
        )

    # create output dir if it does not exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(os.path.abspath(args.output_dir))

    chmm_trainer = CHMMTrainer(
        config=config,
        collate_fn=collate_fn,
        training_dataset=training_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
    ).initialize_trainer()

    if args.train_file:
        logger.info("Start training CHMM.")
        valid_results = chmm_trainer.train()
    else:
        chmm_trainer.load(os.path.join(args.output_dir, 'chmm.bin'), load_optimizer_and_scheduler=True)
        valid_results = None

    if args.test_file:
        logger.info("Start testing CHMM.")
        test_metrics = chmm_trainer.test()
    else:
        test_metrics = None

    result_file = os.path.join(args.output_dir, 'chmm-results.txt')
    logger.info(f"Writing results to {result_file}")
    with open(result_file, 'w') as f:
        if valid_results is not None:
            for i in range(len(valid_results)):
                f.write(f"[Epoch {i + 1}]\n")
                for k, v in valid_results.items(i):
                    f.write(f"  {k}: {v:.4f}")
                f.write("\n")
        if test_metrics is not None:
            f.write(f"[Test]\n")
            for k, v in test_metrics.items():
                f.write(f"  {k}: {v:.4f}")
            f.write("\n")

    logger.info("Collecting garbage.")
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Process finished!")


if __name__ == '__main__':

    _time = datetime.now().strftime("%m.%d.%y-%H.%M")
    _current_file_name = os.path.basename(__file__)
    if _current_file_name.endswith('.py'):
        _current_file_name = _current_file_name[:-3]

    # --- set up arguments ---
    parser = HfArgumentParser(CHMMArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        chmm_args, = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        chmm_args, = parser.parse_args_into_dataclasses()

    if chmm_args.log_dir is None:
        chmm_args.log_dir = os.path.join('logs', f'{_current_file_name}.{_time}.log')

    chmm_train(args=chmm_args)
