# coding=utf-8

import logging
import os
import sys
import copy
import gc
import torch

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from Src.Args import ModelArguments, DataTrainingArguments, CHMMArguments, expend_args
from Src.Bert.BertTrainingPreparation import prepare_bert_training, reinitial_bert_trainer
from Src.CHMM.CHMMTrainingPreparation import prepare_chmm_training

logger = logging.getLogger(__name__)


def main():
    # --- set up arguments ---
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CHMMArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, chmm_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, chmm_args, training_args = parser.parse_args_into_dataclasses()
    expend_args(training_args=training_args, chmm_args=chmm_args, data_args=data_args)

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    # --- setup logging ---
    logging.basicConfig(
        format="[%(levelname)s - %(name)s] - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # --- initialize result-storing file ---
    denoising_model = data_args.denoising_model if data_args.denoising_model else 'true'
    bert_output = os.path.join(
        training_args.output_dir, f"{data_args.dataset_name}-{denoising_model}-{training_args.seed}-bert_results"
    )
    chmm_output = os.path.join(
        training_args.output_dir, f"{data_args.dataset_name}-{denoising_model}-{training_args.seed}_results"
    )
    if os.path.exists(bert_output):
        os.remove(bert_output)

    # --- Set seed ---
    set_seed(training_args.seed)

    # --- setup Neural HMM training functions ---
    chmm_trainer = prepare_chmm_training(
        chmm_args=chmm_args, data_args=data_args, training_args=training_args
    )

    # --- Phase I ---
    # --- train Neural HMM ---
    if training_args.do_train:
        logger.info(" --- starting Neural HMM training process --- ")

        micro_results = chmm_trainer.train()

        results = chmm_trainer.test()
        logger.info("[INFO] test results:")
        for k, v in results.items():
            if 'entity' in k:
                logger.info("  %s = %s", k, v)

        logger.info(" --- Neural HMM training is successfully finished --- ")
        logger.info(f" --- Writing results to {chmm_output} ---")

        with open(chmm_output + '-1.txt', 'w') as f:
            for i, micro_result in enumerate(micro_results):
                f.write(f"[Epoch] {i + 1}\n")
                for k, v in micro_result.items():
                    if 'entity' in k:
                        f.write("%s = %s\n" % (k, v))
            f.write(f"[Test]\n")
            for k, v in results.items():
                if 'entity' in k:
                    f.write("%s = %s\n" % (k, v))

        logger.info(f" --- Results written --- ")
    else:
        logger.info(" --- loading pretrained model... --- ")
        chmm_trainer.load_model()
        logger.info(" --- model loaded --- ")

    # --- get soft labels ---
    logger.info(" --- Constructing soft labels for training set... ---")
    train_annos = chmm_trainer.annotate_data(partition='train')[1]
    eval_annos = chmm_trainer.annotate_data(partition='eval')[1]
    test_annos = chmm_trainer.annotate_data(partition='test')[1]

    # --- setup BERT training functions ---
    bert_trainer, config, true_label_ids = prepare_bert_training(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        training_weak_anno=train_annos,
        eval_weak_anno=eval_annos,
        test_weak_anno=test_annos,
        label_denoiser=chmm_trainer
    )

    if os.path.isfile(bert_output + '-bert-state-dict.pt'):
        logger.warning("Find existing checkpoint, deleting...")
        os.remove(bert_output + '-bert-state-dict.pt')
    best_f1, chmm_trainer = bert_trainer.train(
        model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None,
        log_name=bert_output + '-1',
        model_name=bert_output,
        batch_gd=False
    )

    gc.collect()
    torch.cuda.empty_cache()

    # --- Phase II ---
    # use smaller learning rate and less epochs for phase II
    if not data_args.model_reinit:
        training_args.learning_rate /= 2
        training_args.num_train_epochs = data_args.phase2_train_epochs
        batch_gd = True
    else:
        batch_gd = False

    for n_loop in range(chmm_args.retraining_loops):

        # --- re-initialize Neural hidden Markov model ---
        chmm_trainer.initialize_matrices()
        chmm_trainer.initialize_model()
        chmm_trainer.initialize_optimizers()

        # --- re-train neural hmm model ---
        micro_results = chmm_trainer.train()

        results = chmm_trainer.test()
        logger.info("[INFO] test results:")
        for k, v in results.items():
            if 'entity' in k:
                logger.info("  %s = %s", k, v)

        logger.info(" --- Neural HMM training is successfully finished --- ")
        logger.info(f" --- Writing results to {chmm_output} ---")

        with open(chmm_output + f'-2-{n_loop}.txt', 'w') as f:
            for i, micro_result in enumerate(micro_results):
                f.write(f"[Epoch] {i + 1}\n")
                for k, v in micro_result.items():
                    if 'entity' in k:
                        f.write("%s = %s\n" % (k, v))
            f.write(f"[Test]\n")
            for k, v in results.items():
                if 'entity' in k:
                    f.write("%s = %s\n" % (k, v))

        # --- update BERT training dataset ---
        # get updated denoised labels
        train_annos = chmm_trainer.annotate_data(partition='train')[1]

        if data_args.redistribute_confidence:
            for train_anno in train_annos:
                max_annos = train_anno.max(axis=-1, keepdims=True)
                one_hot_anno = train_anno.copy()
                one_hot_anno[train_anno == max_annos] = 1
                one_hot_anno[train_anno != max_annos] = 0
                train_anno *= 0.5
                train_anno += 0.5 * one_hot_anno

        bert_train_dataset = copy.deepcopy(bert_trainer.update_training_dataset(train_annos))
        bert_eval_dataset = copy.deepcopy(bert_trainer.eval_dataset)
        bert_test_dataset = copy.deepcopy(bert_trainer.test_dataset)

        # garbage collection in case run out of GPU memory
        bert_trainer.model = bert_trainer.model.to('cpu')
        bert_trainer.optimizer_to(torch.device('cpu'))
        del bert_trainer.model
        del bert_trainer.optimizer
        del bert_trainer
        gc.collect()
        torch.cuda.empty_cache()

        # re-initialize bert model
        bert_trainer = reinitial_bert_trainer(
            model_args=model_args,
            training_args=training_args,
            config=config,
            train_dataset=bert_train_dataset,
            eval_dataset=bert_eval_dataset,
            test_dataset=bert_test_dataset,
            label_denoiser=chmm_trainer,
            true_label_ids=true_label_ids,
        )

        best_f1, chmm_trainer = bert_trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None,
            log_name=bert_output + f'-2-{n_loop}',
            model_name=bert_output,
            batch_gd=batch_gd
        )

        gc.collect()
        torch.cuda.empty_cache()


def _mp_fn(_):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
