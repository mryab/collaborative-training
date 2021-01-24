import logging
import os
import uuid
import time
from multiprocessing import cpu_count
from dataclasses import dataclass, field
from typing import Sequence

import torch
import transformers
from pytorch_lamb import Lamb
from datasets import load_dataset, load_from_disk
from transformers import DataCollatorForLanguageModeling, HfArgumentParser, Trainer, TrainingArguments, set_seed, \
    AlbertTokenizerFast, AlbertConfig, AlbertForMaskedLM
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_utils import is_main_process
from huggingface_trainer import ExtendableTrainer

import hivemind
from hivemind.utils import run_in_background, ValueWithExpiration

logger = logging.getLogger(__name__)


@dataclass
class CollaborationArguments:
    experiment: str  # a unique identifier of this experimental run's metadata
    initial_peers: str  # one or more peers (comma-separated) that will welcome you into the collaboration
    metadata_expiration: float = 15  # peer's metadata will be removed if not updated in this many seconds
    global_batch_size: int = 4096  # perform optimizer step after all peers collectively accumulate this many samples
    listen_on: str = '[::]:*'  # network interface used for incoming communication. Default: all ipv6
    refresh_period: int = 1  # wait for this many seconds before checking if we should run optimizer

    model: str = 'albert-base-v2'
    seq_length: int = 512


class CollaborativeTrainer(ExtendableTrainer):
    def __init__(self, *, dht: hivemind.DHT, averager: hivemind.DecentralizedAverager,
                 collaboration_args: CollaborationArguments, **kwargs):
        super().__init__(**kwargs)
        self.dht = dht
        self.averager = averager
        self.is_alive = True
        self.should_update = False
        self.collaboration_args = collaboration_args
        self.trainer_uuid = str(uuid.uuid4())  # a unique identifier of this trainer (for collaborators)
        self.samples_accumulated = 0  # a number of samples accumulated since last optimizer update
        self.steps_accumulated = 0  # a number of steps skipped since last optimizer update
        run_in_background(self.check_collaboration_progress)
        # TODO warm start: if training is already in progress, download the latest model & optimizer

    def apply_gradients(self, epoch, step, tr_loss, trial, steps_in_epoch, local_batch_size):
        """
        Apply gradients with an optimizer, average parameters with peers.
        :note: this function is called every time the original trainer would perform optimizer step
        """
        self.samples_accumulated += local_batch_size
        run_in_background(self.report_training_progress)

        if self.should_update:
            logger.info(f"Running optimizer step {self.state.global_step}")
            self.optimizer_step(epoch, step, tr_loss, trial, steps_in_epoch)
            # ^-- TODO scale gradients based on the training parameters

            logger.info("Averaging with peers...")
            self.averager_step()
            logger.info("Averaging with peers: done!")
            self.samples_accumulated = self.steps_accumulated = 0
            self.report_training_progress()
            self.should_update = False

    def report_training_progress(self):
        """ Declare this trainer's current step and the number of batches accumulated towards the next step """
        self.dht.store(self.collaboration_args.experiment, subkey=self.trainer_uuid,
                       value=dict(samples_accumulated=self.samples_accumulated,
                                  global_step=self.state.global_step),
                       expiration_time=hivemind.get_dht_time() + self.collaboration_args.metadata_expiration,
                       return_future=True)

    def check_collaboration_progress(self):
        """
        Periodically check the training progress from all peers. Trigger update after global_batch_size total samples
        """
        while self.is_alive:
            try:
                time.sleep(self.collaboration_args.refresh_period)
                response = self.dht.get(self.collaboration_args.experiment, latest=True)
                if response is None:
                    continue
                max_step = max(peer_info.value['global_step'] for peer_uuid, peer_info in response.value.items())

                total_samples_accumulated = sum(peer_info.value['samples_accumulated']
                                                for peer_uuid, peer_info in response.value.items()
                                                if peer_info.value['global_step'] >= self.state.global_step)
                total_active_peers = len([peer_info for peer_uuid, peer_info in response.value.items()])

                if max_step > self.state.global_step + 1:
                    logger.info(f"Increasing self.state.global_step {self.state.global_step} -> {max_step}")
                    self.state.global_step = max_step

                logger.info(f"Accumulated {total_samples_accumulated} gradients from {total_active_peers} peers...")
                if total_samples_accumulated >= self.collaboration_args.global_batch_size:
                    self.should_update = True
            except Exception as e:
                logger.exception(e)
                raise

    def averager_step(self):
        """ Average parameters with random peers """
        with torch.no_grad(), self.averager.get_tensors() as averaged_tensors:
            for averaged_tensor, gpu_tensor in zip(averaged_tensors, self.model.parameters()):
                averaged_tensor[...] = gpu_tensor

        self.averager.step(timeout=30)

        with torch.no_grad(), self.averager.get_tensors() as averaged_tensors:
            for averaged_tensor, gpu_tensor in zip(averaged_tensors, self.model.parameters()):
                gpu_tensor[...] = averaged_tensor


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((TrainingArguments, CollaborationArguments))
    training_args, collaboration_args = parser.parse_args_into_dataclasses()

    collaboration_args.initial_peers = list(map(str.strip, collaboration_args.initial_peers.split(',')))
    logger.info(f"Found {len(collaboration_args.initial_peers)} initial peers: {collaboration_args.initial_peers}")
    if len(collaboration_args.initial_peers) == 0:
        raise ValueError("Please specify at least one network endpoint in initial peers.")

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AlbertConfig.from_pretrained(collaboration_args.model)

    tokenizer = AlbertTokenizerFast.from_pretrained(collaboration_args.model)

    model = AlbertForMaskedLM(config)
    model.resize_token_embeddings(len(tokenizer))

    if not os.path.exists('albert_tokenized_wikitext'):
        wikitext = load_dataset('wikitext', 'wikitext-103-v1', cache_dir='.data_cache')

        def tokenize_function(examples):
            # Remove empty lines
            examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
            return tokenizer(
                examples["text"],
                padding=False,
                truncation=True,
                max_length=collaboration_args.seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        tokenized_datasets = wikitext.map(
            tokenize_function,
            batched=True,
            num_proc=cpu_count(),
            remove_columns=["text"],
        )

        tokenized_datasets.save_to_disk('albert_tokenized_wikitext')
    else:
        tokenized_datasets = load_from_disk('albert_tokenized_wikitext')

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = Lamb(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=training_args.max_steps
    )

    dht = hivemind.DHT(initial_peers=list(collaboration_args.initial_peers), start=True)
    averager = hivemind.DecentralizedAverager(
        averaged_tensors=tuple(param.detach().float().cpu() for param in model.parameters()),
        dht=dht, prefix=f"{collaboration_args.experiment}-averager",
        target_group_size=8, averaging_expiration=3, start=True)

    trainer = CollaborativeTrainer(
        model=model, args=training_args,
        dht=dht, averager=averager, collaboration_args=collaboration_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, lr_scheduler)
    )

    # Training
    if training_args.do_train:
        trainer.train()


if __name__ == "__main__":
    main()
