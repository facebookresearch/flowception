from collections.abc import Sequence

import numpy as np
from accelerate.data_loader import skip_first_batches
from torch.utils.data import DataLoader, Dataset
from webdataset import WebLoader
from torch.utils.data import IterableDataset


def get_train_dataloader(
    dataset_name: str, dataset: Dataset, batch_size: int, workers: int, prefetch_factor: float
):
    webdataset_names = [
        "custom_webdataset",
        "custom_webdataset_aes",
    ]
    if dataset_name in webdataset_names:
        assert dataset.nsamples > 0, (
            "nsamples must be specified to be able to compute the length of the dataset"
        )
        length = dataset.nsamples

        num_workers = 8 if dataset_name == "custom_webdataset" else workers

        return WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,  # workers,
        ).with_length(length)
    else:
        num_workers = workers

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=not isinstance(dataset, IterableDataset),
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )


def skip_webloader(webloader: WebLoader, to_skip: int):
    skipping_loader = WebLoader(
        webloader.pipeline[0].dataset,
        batch_size=None,
        shuffle=False,
        num_workers=webloader.pipeline[0].num_workers,
    ).with_length(len(webloader) - to_skip)
    skipping_loader.pipeline[0] = skip_first_batches(skipping_loader.pipeline[0], to_skip)
    return skipping_loader


class ConcatDataloader:
    def __init__(self, dataloaders: Sequence[DataLoader], seed: int = 0):
        self.dataloaders = dataloaders
        self.dataloader_iters = [iter(dataloader) for dataloader in dataloaders]
        self.weights = [len(dataloader) for dataloader in self.dataloaders]
        # self.weights = [int(10**6) for _ in range(len(self.dataloaders))]
        self.ordering = np.concatenate(
            [np.ones(self.weights[index], dtype=np.int32) * index for index in range(len(self.weights))]
        )
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(self.ordering)

    def __iter__(self):
        self.dataloader_iters = [iter(dataloader) for dataloader in self.dataloaders]
        self.iterator = 0
        return self

    def __next__(self):
        dataloader_index = self.ordering[self.iterator]
        dataloader_iter = self.dataloader_iters[dataloader_index]
        batch = next(dataloader_iter)
        self.iterator += 1
        return batch

    def __len__(self):
        return sum(len(dataloader) for dataloader in self.dataloaders)

    def set_epoch(self, epoch: int):
        for dataloader in self.dataloaders:
            if isinstance(dataloader, WebLoader):
                # dataloader.pipeline[0].dataset.pipeline[1].seed = epoch
                continue
            else:
                dataloader.set_epoch(epoch)

    def skip_first_batches(self, num_batches: int):
        skipping_dataloaders = []
        for index, dataloader in enumerate(self.dataloaders):
            to_skip = np.count_nonzero(self.ordering[: num_batches + 1] == index)
            if isinstance(dataloader, WebLoader):
                skipping_dataloaders.append(skip_webloader(dataloader, to_skip))
            else:
                skipping_dataloaders.append(skip_first_batches(dataloader, to_skip))

        skipping_concat_loader = ConcatDataloader(dataloaders=skipping_dataloaders)
        # Preserve the ordeing of this dataloader.
        skipping_concat_loader.ordering = self.ordering[num_batches:]
        return skipping_concat_loader
