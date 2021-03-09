import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from hivemind import run_in_background, ValueWithExpiration
import hivemind

from utils import SimpleAverager, PerformanceEMA
from huggingface_trainer import ExtendableTrainer


logger = logging.getLogger(__name__)


@dataclass
class CollaborationArguments:
    initial_peers: str  # one or more peers (comma-separated) that will welcome you into the collaboration
    dht_key_for_averaging: str = 'my_experiment_name'  # a unique identifier of this experimental run's metadata
    averaging_expiration: float = 3.0  # averaging group will expire after this many seconds
    averaging_step_timeout: float = 30.0  # give up averaging step after this many seconds
    metadata_expiration: float = 15  # peer's metadata will be removed if not updated in this many seconds
    target_group_size: int = 64      # maximum group size for all-reduce
    target_batch_size: int = 4096  # perform optimizer step after all peers collectively accumulate this many samples
    dht_listen_on: str = '[::]:*'  # network interface used for incoming DHT communication. Default: all ipv6
    listen_on: str = '[::]:*'  # network interface used for incoming averager communication. Default: all ipv6
    endpoint: Optional[str] = None

    min_refresh_period: float = 0.5  # wait for at least this many seconds before fetching new collaboration state
    max_refresh_period: float = 30  # wait for at most this many seconds before fetching new collaboration state
    default_refresh_period: float = 3  # attempt to fetch collaboration state every this often until successful
    expected_collaboration_drift_peers: float = 3  # trainer assumes that this many new peers can join per step
    expected_collaboration_drift_rate = 0.2  # trainer assumes that this fraction of current size can join per step

    bandwidth: float = 1000.0  # available network bandwidth, in mbps (used for load balancing in all-reduce)
    performance_ema_alpha: float = 0.1  # uses this alpha for moving average estimate of samples per second
    trainer_uuid: Optional[str] = None  # this trainer's identifier - used when publishing metadata to DHT


@dataclass(frozen=False)
class CollaborationState:
    optimizer_step: int
    samples_accumulated: int
    target_batch_size: int
    num_peers: int
    eta_next_step: float
    next_fetch_time: float

    @property
    def should_perform_step(self):
        return self.samples_accumulated >= self.target_batch_size or hivemind.get_dht_time() >= self.eta_next_step

    def register_step(self):
        self.optimizer_step += 1
        self.samples_accumulated = 0
        self.eta_next_step = float('inf')


class CollaborativeTrainer(ExtendableTrainer):
    def __init__(self, *, collaboration_args: CollaborationArguments, **kwargs):
        super().__init__(**kwargs)
        self.is_alive = True
        self.lock = threading.Lock()

        self.collaboration_args = collaboration_args
        self.matchmaking_prefix = collaboration_args.dht_key_for_averaging + '_matchmaking'
        self.training_progess_key = collaboration_args.dht_key_for_averaging + '_progress'

        self.dht, self.averager = self.initialize_dht_and_averager(collaboration_args)

        self.trainer_uuid = collaboration_args.trainer_uuid or uuid.uuid4().hex
        self.local_samples_accumulated = 0  # a number of local samples accumulated since last optimizer update
        self.local_steps_accumulated = 0  # a number of calls to apply_gradients since last optimizer update
        # each step contains {gradient_accumulation_steps} of forward and backward passes
        self.performance_ema = PerformanceEMA(alpha=collaboration_args.performance_ema_alpha)
        self.collaboration_state = self.fetch_collaboration_state()

    def initialize_dht_and_averager(self, collaboration_args: CollaborationArguments):
        collaboration_args.initial_peers = list(map(str.strip, collaboration_args.initial_peers.split(',')))
        logger.info(f"Found {len(collaboration_args.initial_peers)} initial peers: {collaboration_args.initial_peers}")
        if len(collaboration_args.initial_peers) == 0:
            raise ValueError("Please specify at least one network endpoint in initial peers.")

        dht = hivemind.DHT(initial_peers=list(collaboration_args.initial_peers),
                           listen_on=self.collaboration_args.dht_listen_on,
                           endpoint=self.collaboration_args.endpoint or None,
                           start=True)
        averager = SimpleAverager(self, dht=dht, prefix=self.matchmaking_prefix,
                                  target_group_size=collaboration_args.target_group_size,
                                  throughput=collaboration_args.bandwidth,
                                  compression_type=hivemind.utils.CompressionType.FLOAT16,
                                  averaging_expiration=collaboration_args.averaging_expiration, start=True)
        return dht, averager

    def on_train_begin(self, *args, **kwargs):
        self.averager.load_state_from_peers()
        run_in_background(self.check_collaboration_state_periodically)

    def apply_gradients(self, epoch, step, tr_loss, trial, steps_in_epoch, local_batch_size) -> bool:
        """
        Accumulate gradients until peers collectively reach target batch size; then average and perform optimizer step
        :note: this function is called every time the original trainer would perform optimizer step
        """
        self.local_samples_accumulated += local_batch_size
        self.local_steps_accumulated += 1
        self.performance_ema.update(num_processed=local_batch_size)
        run_in_background(self.report_training_progress)

        if self.collaboration_state.optimizer_step > self.local_step:
            with self.lock:
                logger.info(f"Out of sync (local_step={self.local_step}, global={self.collaboration_state.optimizer_step})")
                self.averager.load_state_from_peers()
                self.local_samples_accumulated = self.local_steps_accumulated = 0
                self.optimizer.zero_grad()

        elif self.collaboration_state.should_perform_step:
            with self.lock:
                logger.info(f"Running optimizer step {self.local_step}")
                average_tr_loss = self.averager_step(tr_loss)
                tr_loss = self.optimizer_step(epoch, step, average_tr_loss, trial, steps_in_epoch)
                self.local_samples_accumulated = self.local_steps_accumulated = 0
                self.collaboration_state.register_step()
                logger.info(f"Optimizer step: done! Accumulating for step {self.local_step}...")
        return tr_loss

    def averager_step(self, tr_loss: torch.Tensor) -> torch.Tensor:
        """ Average parameters and gradients with other peers """
        logger.info("Averaging parameters and gradients with peers...")
        collaboration = self.fetch_collaboration_state()
        if collaboration.num_peers <= 1:
            logger.info(f"Skipping averaging: collaboration consists of {collaboration.num_peers} peers.")
            return tr_loss / self.local_steps_accumulated
        mean_samples_per_worker = collaboration.samples_accumulated / collaboration.num_peers
        weight = self.local_samples_accumulated / mean_samples_per_worker / self.local_steps_accumulated

        local_tensors = [tensor for tensor in self.model.parameters()]
        local_tensors += [tensor.grad for tensor in self.model.parameters()]

        with torch.no_grad(), self.averager.get_tensors() as averaged_tensors:
            assert len(averaged_tensors) == len(local_tensors)
            for averaged_tensor, local_tensor in zip(averaged_tensors, local_tensors):
                averaged_tensor[...] = local_tensor.detach().cpu().float() * weight

        info = dict(tr_loss=tr_loss.item(),
                    steps_accumulated=self.local_steps_accumulated,
                    samples_accumulated=self.local_samples_accumulated,
                    scale=self.local_samples_accumulated / mean_samples_per_worker)
        group_infos = self.averager.step(info, timeout=self.collaboration_args.averaging_step_timeout)
        if group_infos is None:
            logger.warning("Averaging step failed, using local updates only.")
            return tr_loss / self.local_steps_accumulated

        average_loss = self.estimate_average_loss(group_infos) or tr_loss / self.local_steps_accumulated

        # we averaged parameters multiplied by grad scale (aka weights). Time to compensate for that
        # by dividing weights by the sum of grad scales over the entire group.
        sum_of_weights = sum(info['scale'] for info in group_infos.values()
                             if isinstance(info.get('scale'), float))
        normalization_coefficient = (len(group_infos) / sum_of_weights) if sum_of_weights > 0 else 1.0

        with torch.no_grad(), self.averager.get_tensors() as averaged_tensors:

            assert len(averaged_tensors) == len(local_tensors)
            for averaged_tensor, local_tensor in zip(averaged_tensors, local_tensors):
                averaged_tensor *= normalization_coefficient
                local_tensor[...] = averaged_tensor.to(dtype=local_tensor.dtype, device=local_tensor.device)
        logger.info(f"Averaging with peers: done! [group size = {len(group_infos)}, loss = {average_loss:.3f}]")
        return torch.tensor(average_loss)

    def on_train_end(self, *args, **kwargs):
        self.is_alive = False
        logger.info("Sending goodbye to peers")
        self.dht.store(self.training_progess_key, subkey=self.trainer_uuid, value=None,
                       expiration_time=hivemind.get_dht_time() + self.collaboration_args.metadata_expiration)

    def report_training_progress(self):
        """ Declare this trainer's current step and the number of batches accumulated towards the next step """
        current_time = hivemind.get_dht_time()
        local_state_info = [self.local_step, self.local_samples_accumulated,
                            self.performance_ema.samples_per_second, current_time]
        assert self.is_valid_peer_state(local_state_info)
        self.dht.store(self.training_progess_key, subkey=self.trainer_uuid, value=local_state_info,
                       expiration_time=current_time + self.collaboration_args.metadata_expiration, return_future=True)

    def fetch_collaboration_state(self) -> CollaborationState:
        """ Read performance statistics reported by peers, estimate progress towards next batch """
        target_batch_size = self.collaboration_args.target_batch_size
        response, _expiration = self.dht.get(self.training_progess_key, latest=True) or (None, -float('inf'))
        current_time = hivemind.get_dht_time()

        if not isinstance(response, dict) or len(response) == 0:
            logger.warning(f"Found no active peers: {response}")
            local_eta_next_step = max(0, target_batch_size - self.local_steps_accumulated) / self.performance_ema.samples_per_second
            return CollaborationState(self.local_step, self.local_samples_accumulated, target_batch_size, 0,
                                      eta_next_step=current_time + local_eta_next_step,
                                      next_fetch_time=current_time + self.collaboration_args.default_refresh_period)

        valid_peer_states = [peer_state.value for peer_state in response.values()
                             if isinstance(peer_state, ValueWithExpiration)
                             and self.is_valid_peer_state(peer_state.value)]
        global_optimizer_step = max(self.local_step, max(step for step, *_ in valid_peer_states))

        num_peers = len(valid_peer_states)
        total_samples_accumulated = estimated_curent_samples = total_samples_per_second = 0

        for opt_step, samples_accumulated, samples_per_second, timestep in valid_peer_states:
            total_samples_per_second += samples_per_second
            if opt_step == global_optimizer_step:
                total_samples_accumulated += samples_accumulated
                estimated_curent_samples += samples_accumulated + max(0, current_time - timestep) * samples_per_second
            # note: we deliberately count only valid peers for samples_accumulated, but all peers for performance;
            # the rationale behind this is that outdated peers will synchronize and begin contributing shortly.

        estimated_time_to_next_step = max(0, target_batch_size - estimated_curent_samples) / total_samples_per_second

        expected_max_peers = max(num_peers + self.collaboration_args.expected_collaboration_drift_peers,
                                 num_peers * (1 + self.collaboration_args.expected_collaboration_drift_rate))
        time_to_next_fetch = float(np.clip(a=estimated_time_to_next_step * num_peers / expected_max_peers,
                                           a_min=self.collaboration_args.min_refresh_period,
                                           a_max=self.collaboration_args.max_refresh_period))
        logger.info(f"Collaboration accumulated {total_samples_accumulated} samples from {num_peers} peers; "
                    f"ETA {estimated_time_to_next_step:.2f} seconds (refresh in {time_to_next_fetch:.2f}s.)")
        return CollaborationState(global_optimizer_step, total_samples_accumulated, target_batch_size=target_batch_size,
                                  num_peers=num_peers, eta_next_step=current_time + estimated_time_to_next_step,
                                  next_fetch_time=current_time + time_to_next_fetch)

    def check_collaboration_state_periodically(self):
        """
        Periodically check the training progress from all peers. Trigger update after target_batch_size total samples
        """
        while self.is_alive:
            with self.lock:
                self.collaboration_state = self.fetch_collaboration_state()
            time.sleep(max(0, self.collaboration_state.next_fetch_time - hivemind.get_dht_time()))

    def get_train_dataloader(self):
        """ ensure that each worker will have a different (random) batch order """
        torch.manual_seed(hash(self.trainer_uuid))
        return super().get_train_dataloader()

    @property
    def local_step(self) -> int:
        """ Current trainer's local optimizer step """
        # note: v-- global_step is global from huggingface trainer perspective, but local from collaboration perspective
        return self.state.global_step

    @staticmethod
    def is_valid_peer_state(state):
        return isinstance(state, (list, tuple)) and len(state) == 4 \
               and all(map(isinstance, state, (int, int, float, float)))

    @staticmethod
    def estimate_average_loss(group_infos: dict) -> Optional[float]:
        numerator, denominator = 0.0, 0.0

        for peer, info in group_infos.items():
            loss, samples, steps = info.get('tr_loss'), info.get('samples_accumulated'), info.get('steps_accumulated')
            if isinstance(loss, float) and isinstance(samples, int) and isinstance(steps, int) and steps > 0:
                numerator += loss * samples / steps
                denominator += samples
            else:
                logger.info(f'Skipped peer {peer} due to invalid info {info}')

        if denominator > 0:
            return numerator / denominator
        else:
            logger.info("Failed to estimate average loss: no valid peer infos in group")
            return None
