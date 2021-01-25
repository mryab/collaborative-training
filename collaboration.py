import logging
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import hivemind
import torch
from hivemind import run_in_background
from torch.utils.data import DataLoader

from averager import AlbertAverager
from huggingface_trainer import ExtendableTrainer


logger = logging.getLogger(__name__)


@dataclass
class CollaborationArguments:
    initial_peers: str  # one or more peers (comma-separated) that will welcome you into the collaboration
    dht_key_for_averaging: str = 'my_experiment_name'  # a unique identifier of this experimental run's metadata
    averaging_expiration: float = 5.0  # averaging group will expire after this many seconds
    averaging_step_timeout: float = 30.0  # give up averaging step after this many seconds
    metadata_expiration: float = 15  # peer's metadata will be removed if not updated in this many seconds
    target_group_size: int = 32      # maximum group size for all-reduce
    global_batch_size: int = 4096  # perform optimizer step after all peers collectively accumulate this many samples
    listen_on: str = '[::]:*'  # network interface used for incoming communication. Default: all ipv6
    refresh_period: int = 1  # wait for this many seconds before checking if we should run optimizer
    bandwidth: float = 1000.0  # available network bandwidth, in mbps (used for load balancing in all-reduce)
    trainer_uuid: Optional[str] = None  # this trainer's identifier - used when publishing metadata to DHT


@dataclass(frozen=True)
class CollaborationState:
    optimizer_step: int
    samples_accumulated: int
    num_peers: int


class CollaborativeTrainer(ExtendableTrainer):
    def __init__(self, *, collaboration_args: CollaborationArguments, **kwargs):
        super().__init__(**kwargs)
        self.collaboration_args = collaboration_args
        self.matchmaking_prefix = collaboration_args.dht_key_for_averaging + '_matchmaking'
        self.training_progess_key = collaboration_args.dht_key_for_averaging + '_progress'

        self.dht, self.averager = self.initialize_dht_and_averager(collaboration_args)
        self.trainer_uuid = collaboration_args.trainer_uuid or uuid.uuid4().hex

        self.is_alive = True
        self.should_update = False
        self.samples_accumulated = 0  # a number of samples accumulated since last optimizer update
        self.local_steps_accumulated = 0  # a number of local (non-optimizer) steps since last optimizer update
        # each step contains {gradient_accumulation_steps} of forward and backward passes

        self.collaboration_state = CollaborationState(0, 0, 0)
        self.last_timestamp = hivemind.get_dht_time()
        run_in_background(self.check_collaboration_state_periodically)

    def initialize_dht_and_averager(self, collaboration_args: CollaborationArguments):
        collaboration_args.initial_peers = list(map(str.strip, collaboration_args.initial_peers.split(',')))
        logger.info(f"Found {len(collaboration_args.initial_peers)} initial peers: {collaboration_args.initial_peers}")
        if len(collaboration_args.initial_peers) == 0:
            raise ValueError("Please specify at least one network endpoint in initial peers.")

        dht = hivemind.DHT(initial_peers=list(collaboration_args.initial_peers), start=True)

        # averaged tensors are: (*model parameters, *model gradients)
        averaged_tensors = tuple(param.detach().cpu().float().clone() for param in self.model.parameters())
        averaged_tensors += tuple(torch.zeros_like(tensor) for tensor in averaged_tensors)

        averager = AlbertAverager(self, averaged_tensors=averaged_tensors, dht=dht, prefix=self.matchmaking_prefix,
                                  target_group_size=collaboration_args.target_group_size,
                                  compression_type=hivemind.utils.CompressionType.FLOAT16,
                                  averaging_expiration=collaboration_args.averaging_expiration, start=True)
        return dht, averager

    def on_train_begin(self, *args, **kwargs):
        self.last_timestamp = hivemind.get_dht_time()
        maybe_state_from_peers = self.averager.load_state_from_peers()
        if maybe_state_from_peers is not None:
            metadata, flat_tensors = maybe_state_from_peers
            optimizer_flat_keys = metadata['optimizer_flat_keys']
            optimizer_state_dict = metadata.pop('optimizer_state_scalars')
            assert isinstance(optimizer_state_dict, dict)

            model_parameters = flat_tensors[:len(flat_tensors) - len(optimizer_flat_keys)]
            optimizer_flat_tensors = flat_tensors[len(flat_tensors) - len(optimizer_flat_keys):]
            optimizer_state_dict.update(dict(zip(optimizer_flat_keys, optimizer_flat_tensors)))

            assert len(model_parameters) == len(list(self.model.parameters()))
            with torch.no_grad():
                for local_param, loaded_param in zip(self.model.parameters(), model_parameters):
                    local_param[...] = loaded_param
                self.optimizer.load_state_dict(optimizer_state_dict)

            logger.info("Loaded model/optimizer state from an active peer.")

    def apply_gradients(self, epoch, step, tr_loss, trial, steps_in_epoch, local_batch_size):
        """
        Apply gradients with an optimizer, average parameters with peers.
        :note: this function is called every time the original trainer would perform optimizer step
        """
        self.samples_accumulated += local_batch_size
        self.local_steps_accumulated += 1
        run_in_background(self.report_training_progress)

        if self.should_update:
            logger.info("Averaging parameters and gradients with peers...")
            self.averager_step()
            logger.info("Averaging with peers: done!")

            logger.info(f"Running optimizer step {self.state.global_step}")
            self.optimizer_step(epoch, step, tr_loss, trial, steps_in_epoch)
            self.samples_accumulated = self.local_steps_accumulated = 0
            self.should_update = False

    def averager_step(self):
        """ Average parameters and gradients with other peers """
        collaboration = self.get_collaboration_state()
        if collaboration.num_peers <= 1:
            logger.warning(f"Skipping averaging: collaboration consists of {collaboration.num_peers} peers.")
            return
        mean_samples_per_worker = collaboration.samples_accumulated / collaboration.num_peers
        grad_scale = self.samples_accumulated / mean_samples_per_worker / self.local_steps_accumulated

        with torch.no_grad(), self.averager.get_tensors() as averaged_tensors:
            tensors_for_averaging = [tensor.cpu().float() for tensor in self.model.parameters()]
            tensors_for_averaging += [tensor.grad.cpu().float() * grad_scale for tensor in self.model.parameters()]
            for averaged_tensor, gpu_tensor in zip(averaged_tensors, self.model.parameters()):
                averaged_tensor[...] = gpu_tensor

        self.averager.step(timeout=self.collaboration_args.averaging_step_timeout)

        with torch.no_grad(), self.averager.get_tensors() as averaged_tensors:
            for averaged_tensor, gpu_tensor in zip(averaged_tensors, self.model.parameters()):
                gpu_tensor[...] = averaged_tensor

    def report_training_progress(self):
        """ Declare this trainer's current step and the number of batches accumulated towards the next step """
        self.dht.store(self.training_progess_key, subkey=self.trainer_uuid,
                       value=dict(samples_accumulated=self.samples_accumulated, global_step=self.state.global_step),
                       expiration_time=hivemind.get_dht_time() + self.collaboration_args.metadata_expiration,
                       return_future=True)

    def get_collaboration_state(self) -> CollaborationState:
        """
        Read performance statistics reported by peers, estimate progress towards next batch
        """
        response, _expiration = self.dht.get(self.training_progess_key, latest=True) or (None, -float('inf'))
        if not isinstance(response, dict) or len(response) == 0:
            logger.warning(f"Found no active peers: {response}")
            return CollaborationState(0, 0, 0)

        total_active_peers = len(response)
        current_step = max(peer_info.value['global_step'] for peer_uuid, peer_info in response.items())

        total_samples_accumulated = sum(peer_info.value['samples_accumulated']
                                        for peer_uuid, peer_info in response.items()
                                        if peer_info.value['global_step'] >= self.state.global_step)

        if current_step > self.state.global_step:
            logger.info(f"Increasing self.state.global_step {self.state.global_step} -> {current_step}")
            self.state.global_step = current_step
            # TODO update scheduler!

        logger.info(f"Total {total_samples_accumulated} samples accumulated from {total_active_peers} peers...")
        return CollaborationState(current_step, total_samples_accumulated, total_active_peers)

    def check_collaboration_state_periodically(self):
        """
        Periodically check the training progress from all peers. Trigger update after global_batch_size total samples
        """
        while self.is_alive:
            try:
                time.sleep(self.collaboration_args.refresh_period)
                self.collaboration_state = self.get_collaboration_state()

                logger.info(f"Accumulated {self.collaboration_state.samples_accumulated} samples"
                            f" over {self.collaboration_state.num_peers} peers...")
                if self.collaboration_state.samples_accumulated >= self.collaboration_args.global_batch_size:
                    self.should_update = True
            except Exception as e:
                logger.exception(e)
                raise

    def get_train_dataloader(self):
        """ ensure that each worker will have a different (random) batch order (TODO there's gotta be a better way) """
        torch.manual_seed(hash(self.trainer_uuid))
        return super().get_train_dataloader()
