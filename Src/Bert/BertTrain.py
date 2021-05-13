from Src.Bert.BertImport import *

logger = logging.getLogger(__name__)


# noinspection PyTypeChecker,PyUnresolvedReferences
class SoftTrainer(Trainer):

    def __init__(
            self,
            model: Union[PreTrainedModel, torch.nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            test_dataset: Optional[Dataset] = None,
            tokenizer: Optional["PreTrainedTokenizerBase"] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            label_denoiser: Union[CHMMTrainer] = None
    ):
        super().__init__(model, args,
                         data_collator,
                         train_dataset,
                         eval_dataset,
                         tokenizer,
                         model_init,
                         compute_metrics,
                         callbacks,
                         optimizers)
        self.label_denoiser = label_denoiser
        self.test_dataset = test_dataset

    def train(self,
              model_path: Optional[str] = None,
              trial: Union["optuna.Trial", Dict[str, Any]] = None,
              log_name: Optional[str] = 'log',
              model_name: Optional[str] = 'model',
              batch_gd: Optional[bool] = False):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            log_name:  where to save file
            batch_gd:
            model_name:
        """
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)

            model = self.call_model_init(trial)

            self.model = model.to(self.args.device)

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)
        if not self.args.model_reinit:
            try:
                logger.info("Loading state dict...")
                self.model.load_state_dict(torch.load(model_name + '-bert-state-dict.pt'))
            except:
                logger.error("State dict does not exist! using reinitialized bert...")

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        sequential_train_dataloader = self.get_sequential_train_dataloader()
        eval_dataloader = self.get_eval_dataloader()
        test_dataloader = self.get_test_dataloader(self.test_dataset) \
            if self.test_dataset is not None else None

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                    self.args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = self.args.max_steps
            num_train_epochs = 1

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self.state = TrainerState()

        # Check if saved optimizer or scheduler states exist
        if (model_path is not None
                and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
                and os.path.isfile(os.path.join(model_path, "scheduler.pt"))):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            with warnings.catch_warnings(record=True) as caught_warnings:
                self.lr_scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))
            reissue_pt_warnings(caught_warnings)

        model = self.model

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=(
                    not getattr(model.config, "gradient_checkpointing", False)
                    if isinstance(model, PreTrainedModel)
                    else True
                ),
            )

        logger.info("***** Running training *****")
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", max_steps)

        self.state.epoch = 0

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        self._logging_loss_scalar = 0
        self._total_flos = self.state.total_flos

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # --- phase I training: use the prediction of the CHMM model to train BERT model
        logger.info(" --- Start Phase I training --- ")

        best_f1 = self.train_epochs(
            train_dataloader,
            num_train_epochs,
            train_dataset_is_sized,
            log_name,
            model_name=model_name,
            batch_gd=batch_gd
        )

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed.\n\n")

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)

        self.update_chmm(sequential_train_dataloader, eval_dataloader, test_dataloader)

        return best_f1, self.label_denoiser

    def train_epochs(self,
                     train_dataloader,
                     num_train_epochs,
                     train_dataset_is_sized,
                     log_name,
                     model_name,
                     batch_gd=False):

        model = self.model

        best_f1 = 0.0
        tolerance_epoch = 0
        result_lines = ''
        model.zero_grad()

        # --- training epoch ---
        for epoch in range(num_train_epochs):
            logger.info("  ===== Start Epoch %d =====  ", epoch)
            # logger.info("  ---------- Start Training ----------  ")

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_process_zero())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_process_zero())

            # Reset the past memory state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = len(epoch_iterator) if train_dataset_is_sized else self.args.max_steps
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            # --- training step ---
            for step, inputs in enumerate(epoch_iterator):

                # --- start of regular training step ---
                self.regular_training_step(inputs, step, epoch, steps_in_epoch, batch_gd=batch_gd)
                # --- end of regular training step ---

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            if batch_gd:
                if is_torch_tpu_available():
                    xm.optimizer_step(self.optimizer)
                elif self.args.fp16 and _use_native_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.lr_scheduler.step()
                model.zero_grad()

            # --- end of a training epoch ---
            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)

            # evaluation session
            if self.args.do_eval:
                logger.info("  ===== Evaluation =====  ")
                eval_results = self.evaluate()
                f1 = eval_results['eval_f1']

                logger.info("***** Eval results *****")
                if self.is_world_process_zero():
                    result_lines += f" ----- Epoch = {epoch + 1} ----- \n"
                    for key, value in eval_results.items():
                        if key == 'epoch':
                            continue
                        logger.info("  %s = %s", key, value)
                        result_lines += f"{key} = {value}\n"

                    if f1 >= best_f1:
                        best_f1 = f1
                        torch.save(self.model.state_dict(), model_name + '-bert-state-dict.pt')
                        logger.info("  checkpoint updated!  ")
                        result_lines += "  checkpoint updated!  \n"
                        tolerance_epoch = 0
                    else:
                        tolerance_epoch += 1
            if tolerance_epoch > self.args.bert_tolerance_epoch:
                break

        # load the best model parameters after training
        if best_f1 > 0:
            logger.info(
                f"Loading the best model..."
            )
            self.model.load_state_dict(torch.load(model_name + '-bert-state-dict.pt'))

        if self.args.do_predict:
            logger.info("  ===== Test =====  ")
            test_results = self.test()

            logger.info("*** Test results ***")
            if self.is_world_process_zero():
                result_lines += f" ----- Test ----- \n"
                for key, value in test_results.items():
                    if key == 'epoch':
                        continue
                    logger.info("  %s = %s", key, value)
                    result_lines += f"{key} = {value}\n"

        with open(log_name + f'.txt', 'w') as f:
            f.write(result_lines)

        return best_f1

    def regular_training_step(self, inputs, step, epoch, steps_in_epoch, batch_gd):
        model = self.model

        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

        if (
                ((step + 1) % self.args.gradient_accumulation_steps != 0)
                and self.args.local_rank != -1
                and _use_ddp_no_sync
        ):
            with model.no_sync():
                step_loss = self.training_step(model, inputs)
        else:
            step_loss = self.training_step(model, inputs)
        self._total_flos += self.floating_point_ops(inputs)

        if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                self.args.gradient_accumulation_steps >= steps_in_epoch == (step + 1)
        ):
            if self.args.fp16 and _use_native_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
            elif self.args.fp16 and _use_apex:
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

            if not batch_gd:
                if is_torch_tpu_available():
                    xm.optimizer_step(self.optimizer)
                elif self.args.fp16 and _use_native_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.lr_scheduler.step()
                model.zero_grad()
            self.state.global_step += 1
            self.state.epoch = epoch + (step + 1) / steps_in_epoch
            self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

        return step_loss

    def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        labels = inputs['labels']
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        outputs = model(labels=labels, input_ids=input_ids, attention_mask=attention_mask)

        # model outputs are always tuple in transformers (see doc)
        loss, logits, _ = outputs

        try:
            weak_lb_weights = inputs['weak_lb_weights']

            if weak_lb_weights.dtype == torch.int64:
                weak_lb_weights = weak_lb_weights.view(-1)
                logits = logits.view(weak_lb_weights.size(0), -1)
                loss = F.cross_entropy(logits, weak_lb_weights)
            else:
                loss = self.batch_kld_loss(
                    torch.log_softmax(logits, dim=-1), weak_lb_weights, labels != labels[0, 0]
                )
        except KeyError:
            pass

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16 and _use_native_amp:
            self.scaler.scale(loss).backward()
        elif self.args.fp16 and _use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.detach()

    def prediction_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():

            labels = inputs['labels']
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            outputs = model(labels=labels, input_ids=input_ids, attention_mask=attention_mask)[:2]
            if has_labels:
                loss = outputs[0].mean().detach()
                logits = outputs[1:]
            else:
                loss = None
                # Slicing so we get a tuple even if `outputs` is a `ModelOutput`.
                logits = outputs[:]
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]
                # Remove the past from the logits.
                logits = logits[: self.args.past_index - 1] + logits[self.args.past_index:]

        if prediction_loss_only:
            return loss, None, None

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        return loss, logits, labels

    def test(self):
        """
        Run test and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        if self.test_dataset is None:
            raise ValueError("test_dataset must be implemented")

        test_dataloader = self.get_eval_dataloader(self.test_dataset)

        output = self.prediction_loop(test_dataloader, description="Test")

        self.log(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output.metrics

    @staticmethod
    def batch_kld_loss(batch_log_q, batch_p, batch_mask=None):
        """
        :param batch_log_q: Q(x) in log domain
        :param batch_p: P(x)
        :param batch_mask: select elements to compute loss
        Log-domain KLD loss
        :return: kld loss
        """
        kld = 0
        for log_q, p, mask in zip(batch_log_q, batch_p, batch_mask):
            kld += torch.sum(p[mask] * (torch.log(p[mask]) - log_q[mask]))
        kld /= len(batch_log_q)

        return kld

    def _get_sequential_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if is_torch_tpu_available():
            return SequentialDistributedSampler(
                self.train_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif self.args.local_rank != -1:
            return SequentialDistributedSampler(self.train_dataset)
        else:
            return SequentialSampler(self.train_dataset)

    def get_sequential_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("CHMMTrainer: training requires a train_dataset.")
        train_sampler = self._get_sequential_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_bert_embeddings(self, dataloader) -> List[torch.tensor]:
        self.model.eval()

        embs = list()
        for inputs in dataloader:
            with torch.no_grad():
                inputs = self._prepare_inputs(inputs)

                labels = inputs['labels']
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']

                outputs = self.model(labels=labels, input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_states = outputs[2][-1].detach().cpu().numpy()

                embs += [last_hidden_state for last_hidden_state in last_hidden_states]

        return embs

    def get_bert_predictions(self, train_dataloader, eval_dataloader, test_dataloader):

        # --- get bert predictions for training and evaluation set ---
        logger.info("getting training predictions")
        train_preds, _, _ = self.prediction_loop(
            train_dataloader, description="bert-model-prediction-train"
        )
        logger.info("getting evaluation predictions")
        eval_preds, _, _ = self.prediction_loop(
            eval_dataloader, description="bert-model-prediction-eval"
        )
        logger.info("getting test predictions")
        test_preds, _, _ = self.prediction_loop(
            test_dataloader, description="bert-model-prediction-test"
        )
        if self.args.update_embeddings:
            train_embs = self.get_bert_embeddings(train_dataloader)
            eval_embs = self.get_bert_embeddings(eval_dataloader)
            test_embs = self.get_bert_embeddings(test_dataloader)
        else:
            train_embs = eval_embs = test_embs = None
        # --- transfer bert predictions to probabilities ---
        p_train = list()
        trimmed_train_embs = list()
        for i in range(len(self.train_dataset.features)):
            non_padding_indices = np.array(self.train_dataset.features[i].label_ids) \
                                  != nn.CrossEntropyLoss().ignore_index
            non_padding_preds = train_preds[i][non_padding_indices]
            p = soft_frequency(non_padding_preds)
            p_train.append(p)
            if self.args.update_embeddings:
                non_padding_emb = train_embs[i][non_padding_indices]
                trimmed_train_embs.append(non_padding_emb)

        p_eval = list()
        trimmed_eval_embs = list()
        for i in range(len(self.eval_dataset.features)):
            non_padding_indices = np.array(self.eval_dataset.features[i].label_ids) \
                                  != nn.CrossEntropyLoss().ignore_index
            non_padding_preds = eval_preds[i][non_padding_indices]
            p = soft_frequency(non_padding_preds)
            p_eval.append(p)
            if self.args.update_embeddings:
                non_padding_emb = eval_embs[i][non_padding_indices]
                trimmed_eval_embs.append(non_padding_emb)

        if test_dataloader is not None:
            p_test = list()
            trimmed_test_embs = list()
            for i in range(len(self.test_dataset.features)):
                non_padding_indices = np.array(self.test_dataset.features[i].label_ids) \
                                      != nn.CrossEntropyLoss().ignore_index
                non_padding_preds = test_preds[i][non_padding_indices]
                p = soft_frequency(non_padding_preds)
                p_test.append(p)
                if self.args.update_embeddings:
                    non_padding_emb = test_embs[i][non_padding_indices]
                    trimmed_test_embs.append(non_padding_emb)
        else:
            p_test = None
            trimmed_test_embs = None

        # re-group predictions
        training_lengths = [len(obs) for obs in self.label_denoiser.train_dataset.obs]
        eval_lengths = [len(obs) for obs in self.label_denoiser.eval_dataset.obs]
        test_lengths = [len(obs) for obs in self.label_denoiser.test_dataset.obs]
        p_train = self.regroup_predictions(p_train, training_lengths)
        p_eval = self.regroup_predictions(p_eval, eval_lengths)
        p_test = self.regroup_predictions(p_test, test_lengths) if p_test is not None else None

        if self.args.update_embeddings:
            train_embs = self.regroup_predictions(trimmed_train_embs, training_lengths)
            eval_embs = self.regroup_predictions(trimmed_eval_embs, eval_lengths)
            test_embs = self.regroup_predictions(trimmed_test_embs, test_lengths) \
                if trimmed_test_embs is not None else None

        return p_train, p_eval, p_test, train_embs, eval_embs, test_embs

    def update_chmm(self,
                    train_dataloader,
                    eval_dataloader,
                    test_dataloader):
        p_train, p_eval, p_test, train_embs, eval_embs, test_embs = self.get_bert_predictions(
            train_dataloader, eval_dataloader, test_dataloader
        )

        # --- update Neural HMM training and evaluation dataset ---
        self.label_denoiser.append_dataset_obs(self.label_denoiser.train_dataset, p_train)
        self.label_denoiser.append_dataset_obs(self.label_denoiser.eval_dataset, p_eval)
        if p_test is not None:
            self.label_denoiser.append_dataset_obs(self.label_denoiser.test_dataset, p_test)
        self.label_denoiser.has_appended_obs = True

        if self.args.update_embeddings:
            self.label_denoiser.update_embs(self.label_denoiser.train_dataset, train_embs)
            self.label_denoiser.update_embs(self.label_denoiser.eval_dataset, eval_embs)
            if test_embs is not None:
                self.label_denoiser.update_embs(self.label_denoiser.test_dataset, test_embs)

    @staticmethod
    def regroup_predictions(predictions: List[np.array],
                            seq_lengths: List[int]) -> List[np.array]:
        """
        Since the bert model split instances when they exceed the maximum length,
        we need to re-group those instances before feeding them to Neural HMM
        :param predictions: predictions
        :param seq_lengths: sequence lengths
        :return: re-grouped predictions
        """
        if len(seq_lengths) == len(predictions):
            return predictions

        p_update = list()
        p_update.append(predictions[0].copy())
        i = 0
        j = 0
        while i < len(predictions) - 1 or j < len(seq_lengths) - 1:
            if len(p_update[-1]) == seq_lengths[j]:
                i += 1
                j += 1
                p_update.append(predictions[i].copy())
            else:
                i += 1
                p_update[-1] = np.r_[p_update[-1], predictions[i].copy()]
                assert len(p_update[-1]) <= seq_lengths[j], "The length of splitted probabilities " \
                                                            "should be less than or equal to sequence length."
        assert len(p_update) == len(seq_lengths)
        for p, sl in zip(p_update, seq_lengths):
            assert len(p) == sl

        return p_update

    def update_training_dataset(self, annos):
        assert len(annos) <= len(self.train_dataset.features)
        j = 0
        start = 0
        for i in range(len(self.train_dataset.features)):
            update_annos = np.zeros_like(self.train_dataset.features[i].weak_lb_weights)
            non_padding_annos = np.array(self.train_dataset.features[i].label_ids) != nn.CrossEntropyLoss().ignore_index
            if non_padding_annos.sum() < len(annos[j]):
                update_annos[non_padding_annos] = annos[j][start: start + non_padding_annos.sum()]
                self.train_dataset.features[i].weak_lb_weights = update_annos
                start += non_padding_annos.sum()

                if start == len(annos[j]):
                    start = 0
                    j += 1
                elif start > len(annos[j]):
                    raise ValueError("The original annotation and the prediction do not match!")
            else:
                assert non_padding_annos.sum() == len(annos[j])
                update_annos[non_padding_annos] = annos[j]
                self.train_dataset.features[i].weak_lb_weights = update_annos

                j += 1
        assert j == len(annos)
        return self.train_dataset

    def reinitialize_bert_trainer(self, train_dataloader, max_steps, num_train_epochs, init_model_state_dict):
        self.state = TrainerState()
        self.control = TrainerControl()

        # Seed must be set before instantiating the model when using model_init.
        if self.args.model_reinit:
            self.model.load_state_dict(init_model_state_dict)

        # re-initialize optimizer and scheduler
        self.optimizer, self.lr_scheduler = None, None
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        self._logging_loss_scalar = 0
        self._total_flos = self.state.total_flos

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

    def optimizer_to(self, device):
        for param in self.optimizer.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)
