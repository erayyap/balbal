"""Epoch-seeded target-length sampling, independent of model/global RNG."""
import torch
from torch.utils.data import Sampler
from transformers.trainer_pt_utils import get_length_grouped_indices


class EpochLengthSampler(Sampler):
    def __init__(self, lengths, batch_size, seed, shuffle_batches=False):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.shuffle_batches = shuffle_batches

    def __len__(self):
        return len(self.lengths)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=generator)
        if self.shuffle_batches:
            # Preserve every length-matched effective batch, but remove the
            # repeating long-to-short update sequence. Keep the longest batch
            # first for early OOM detection and the partial batch last.
            full_count = len(indices) // self.batch_size
            if full_count > 2:
                middle = (torch.randperm(full_count - 1, generator=generator) + 1).tolist()
                indices = (indices[:self.batch_size]
                           + [i for b in middle for i in indices[b*self.batch_size:(b+1)*self.batch_size]]
                           + indices[full_count*self.batch_size:])
        return iter(indices)
