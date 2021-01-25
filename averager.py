from itertools import chain

import hivemind
import torch


class AlbertAverager(hivemind.DecentralizedAverager):
    """ A daemon that runs decentralized averaging before each training step """
    def __init__(self, trainer, **kwargs):
        self.trainer = trainer  # note: this is a cyclic reference, we should to un-cycle it
        super().__init__(**kwargs)

    def get_current_state(self):
        """
        Get current model/optimizer state and when requested by a newbie peer. executed in the host process.
        :returns: a tuple of (serializable_small_metadata, sequence of torch tensors)
        """
        with torch.no_grad():
            model_state_tensors = list(map(torch.Tensor.cpu, self.trainer.model.parameters()))
            optimizer_state_tensors = {key: value.cpu() for key, value in self.trainer.optimizer.state_dict().items()
                                       if isinstance(value, torch.Tensor)}
            optimizer_state_scalars = {key: value for key, value in self.trainer.optimizer.state_dict().items()
                                       if not isinstance(value, torch.Tensor)}
        optimizer_flat_keys, optimizer_flat_tensors = tuple(zip(*optimizer_state_tensors.items())) or ([], [])
        all_flat_tensors = list(chain(model_state_tensors, optimizer_flat_tensors))
        metadata = dict(step=self.trainer.state.global_step, group_key=self.get_current_group_key(),
                        optimizer_state_scalars=optimizer_state_scalars, optimizer_flat_keys=optimizer_flat_keys)
        return metadata, all_flat_tensors
