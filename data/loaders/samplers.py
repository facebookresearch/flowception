import itertools

import numpy as np
import torch
import torch.distributed as distributed
from torch.utils.data.sampler import Sampler


class EpochSampler(Sampler):
    def __init__(
        self,
        *,
        size: int,
        sample_count: int,
        shuffle: bool = True,
        seed: int = 0,
        start: int | None = None,
        step: int | None = None,
    ):
        self._size = size
        self._sample_count = sample_count
        self._shuffle = shuffle
        self._seed = seed
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._epoch = 0

    def __iter__(self):
        seed = self._seed * self._epoch if self._seed != 0 else self._epoch
        rng = np.random.default_rng(seed)

        count = (self._size + self._sample_count - 1) // self._sample_count
        tiled_indices = np.tile(np.arange(self._sample_count), count)
        iterable = rng.choice(tiled_indices, self._size, replace=False)

        yield from itertools.islice(iterable, self._start, None, self._step)

    def __len__(self):
        return (self._size - self._start + self._step - 1) // self._step

    def set_epoch(self, epoch):
        self._epoch = epoch


def _generate_randperm_indices(size: int, generator: torch.Generator):
    """Generate the indices of a random permutation."""
    dtype = torch.int32 if size <= 2**31 else torch.int64
    # This is actually matching PyTorch's CPU implementation, see: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorFactories.cpp#L900-L921
    perm = torch.arange(size, dtype=dtype)

    for i in range(size):
        j = torch.randint(i, size, size=(1,), generator=generator).item()

        # Always swap even if no-op
        value = perm[j].item()
        perm[j] = perm[i].item()
        perm[i] = value

        yield value


class IterationSampler(Sampler):
    def __init__(
        self,
        *,
        sample_count: int,
        shuffle: bool = True,
        seed: int = 0,
        start: int | None = None,
        step: int | None = None,
        advance: int = 0,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._advance = advance

    def __iter__(self):
        yield from itertools.islice(self._iterator(), self._advance, None)

    def _iterator(self):
        # Instantiate a generator here (rather than in the ctor) to be keep the class
        # picklable (requirement of mp.spawn)
        generator = torch.Generator()
        generator.manual_seed(self._seed)

        while True:
            if self._shuffle:
                iterable = _generate_randperm_indices(self._sample_count, generator)
            else:
                iterable = range(self._sample_count)

            yield from itertools.islice(iterable, self._start, None, self._step)

    def __len__(self):
        return self._sample_count
