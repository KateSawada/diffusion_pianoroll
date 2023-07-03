from logging import getLogger
from multiprocessing import Manager
from dataclasses import dataclass

import numpy as np
from hydra.utils import to_absolute_path
from diffusion_pianoroll.utils import read_txt
from torch.utils.data import Dataset
import torch

# A logger for this file
logger = getLogger(__name__)


@dataclass
class PianorollDatasetItem:
    filename: str
    pianoroll: np.ndarray


class PianorollDataset(Dataset):
    """PyTorch compatible audio and acoustic feat. dataset."""

    def __init__(
        self,
        pianoroll_list,
        n_tracks=5,
        measure_resolution=48,
        n_pitches=84,
        n_measures=4,
        return_filename=False,
        allow_cache=False,
    ) -> None:
        """Initialize dataset.

        Args:
            pianoroll_list (str): Filename of the list of pianoroll npy files.
            n_tracks (int): Number of tracks.
            measure_resolution (int): timestep resolution per measure.
            n_pitches (int): Number of pitches.
            n_measures (int): Number of measures.
            return_filename (bool): Whether to return the filename with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.
        """
        # load pianoroll files & check filename
        pianoroll_files = read_txt(to_absolute_path(pianoroll_list))

        self.pianoroll_files = pianoroll_files
        self.n_tracks = n_tracks
        self.measure_resolution = measure_resolution
        self.n_pitches = n_pitches
        self.n_measures = n_measures
        self.return_filename = return_filename
        self.allow_cache = allow_cache

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader
            # with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(pianoroll_files))]

    def __getitem__(self, idx: int) -> PianorollDatasetItem:
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            PianorollDatasetItem: Item with specified idx. It contains filename
            if self.return_filename is True.
        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]
        # load audio and features
        pianoroll = np.load(to_absolute_path(self.pianoroll_files[idx]))
        pianoroll = pianoroll.astype(np.float32)

        if self.return_filename:
            items = PianorollDatasetItem(self.pianoroll_files[idx], pianoroll)
        else:
            items = PianorollDatasetItem("", pianoroll)

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.pianoroll_files)

    @staticmethod
    def train_collator(batch: list[PianorollDatasetItem]) -> torch.FloatTensor:
        """collate function for training

        Args:
            batch (list[PianorollDatasetItem]): batch

        Returns:
            torch.FloatTensor: pianoroll tensor
        """
        pianoroll_batch = []
        for idx in range(len(batch)):
            pianoroll = batch[idx].pianoroll
            pianoroll_batch += [pianoroll.astype(np.float32)]

        # convert each batch to tensor,
        # assume that each item in batch has the same length
        pianoroll_batch = torch.FloatTensor(np.array(pianoroll_batch))

        pianoroll_batch = torch.permute(pianoroll_batch, (0, 4, 1, 2, 3))
        return pianoroll_batch
