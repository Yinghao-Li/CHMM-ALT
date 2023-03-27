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
    set_seed(args.seed)
    config = CHMMConfig().from_args(args)

    # create output dir if it does not exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(os.path.abspath(args.output_dir))

    # load dataset
    training_dataset = valid_dataset = test_dataset = None
    if args.load_preprocessed_dataset:
        logger.info('Loading pre-processed datasets...')
        file_dir = os.path.split(args.train_path)[0]
        try:
            training_dataset = CHMMBaseDataset().load(
                file_dir=file_dir,
                dataset_type='train',
                config=config
            )
            valid_dataset = CHMMBaseDataset().load(
                file_dir=file_dir,
                dataset_type='valid',
                config=config
            )
            test_dataset = CHMMBaseDataset().load(
                file_dir=file_dir,
                dataset_type='test',
                config=config
            )
        except Exception as err:
            logger.exception(f"Encountered error {err} while loading the pre-processed datasets")
            training_dataset = valid_dataset = test_dataset = None

    if training_dataset is None:
        if args.train_path:
            logger.info('Loading training dataset...')
            training_dataset = CHMMBaseDataset().load_file(
                file_path=args.train_path,
                config=config
            )
        if args.valid_path:
            logger.info('Loading validation dataset...')
            valid_dataset = CHMMBaseDataset().load_file(
                file_path=args.valid_path,
                config=config
            )
        if args.test_path:
            logger.info('Loading test dataset...')
            test_dataset = CHMMBaseDataset().load_file(
                file_path=args.test_path,
                config=config
            )

        if config.save_dataset:
            logger.info(f"Saving datasets")
            output_dir = os.path.split(config.train_path)[0] if config.save_dataset_to_data_dir else args.output_dir

            training_dataset.save(output_dir, 'train', config)
            valid_dataset.save(output_dir, 'valid', config)
            test_dataset.save(output_dir, 'test', config)

    chmm_trainer = CHMMTrainer(
        config=config,
        collate_fn=collate_fn,
        training_dataset=training_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
    ).initialize_trainer()

    if args.train_path:
        logger.info("Start training CHMM.")
        valid_results = chmm_trainer.train()
    else:
        chmm_trainer.load(os.path.join(args.output_dir, 'chmm.bin'), load_optimizer_and_scheduler=True)
        valid_results = None

    if args.test_path:
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

    # Setup logging
    if chmm_args.log_dir is None:
        chmm_args.log_dir = os.path.join('logs', f'{_current_file_name}', f'{_time}.log')

    set_logging(log_dir=chmm_args.log_dir)
    logging_args(chmm_args)

    try:
        chmm_train(args=chmm_args)
    except Exception as e:
        logger.exception(e)
        raise e
