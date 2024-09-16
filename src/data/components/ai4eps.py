"""
ai4eps.py: this file contains dataset following standard AI4EPS format.
Reference: https://ai4eps.github.io/homepage/ml4earth/seismic_event_format1/
"""
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src import utils
from src.data.components.utils import generate_label, normalize_waveform, stack_rand

log = utils.get_pylogger(__name__)


class Ai4epsDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        index_to_waveform_id: List[Tuple[str, str]] = [],
        transform: Optional[Callable] = None,
        label_shape: str = "gaussian",
        label_width_in_npts: int = 120,
        window_length_in_npts: int = 4800,
        phases: List[str] = ["P", "S", "PS"],
        first_arrival_index_in_final_window_if_no_shift: int = 400,
        random_stack_two_waveforms_ratio=0.0,
    ):
        """
        Args:
            data_dir (Path): the directory of the dataset
            index_to_waveform_id (List[Tuple[str, str]], optional): list of tuples, each tuple is (event_id, station_id). Defaults to []. Only the waveforms in the list will be used.
            transform (Optional[Callable], optional): Optional transform to be applied on a sample. Defaults to None.
            label_shape (str, optional): the shape of the label, can be "gaussian" or "triangle". Defaults to "gaussian".
            label_width_in_npts (int, optional): the width of the label in number of points. Defaults to 120.
            window_length_in_npts (int, optional): the length of the window in number of points. Defaults to 4800.
            phases (List[str], optional): list of phases. Defaults to ["P", "S", "PS"].
            first_arrival_index_in_final_window_if_no_shift (int, optional): the index of the first arrival in the final window if no shift. Defaults to 400.
            random_stack_two_waveforms_ratio (float, optional): the ratio of stacking two waveforms. Defaults to 0.0.
        """
        self.transform = transform
        self.label_shape = label_shape
        self.label_width_in_npts = label_width_in_npts
        self.window_length_in_npts = window_length_in_npts
        self.phases = phases
        self.first_arrival_index_in_final_window_if_no_shift = (
            first_arrival_index_in_final_window_if_no_shift
        )
        self.random_stack_two_waveforms_ratio = random_stack_two_waveforms_ratio

        # waveform.h5 is a hdf5 file containing waveform data
        # eg. f["11_52111"]["A01"][...] is a 3XNT numpy array
        # f["11_52111"]["A01"].attrs is the attributes of the waveform
        self.h5py_dir = data_dir / "waveform"
        self._handler = {}
        # index_to_waveform_id is a list of tuples, each tuple is (event_id, station_id)
        # eg. [("11_52111", "A01"), ("11_52111", "A02"), ...]
        self.index_to_waveform_id = index_to_waveform_id

    def get_handler(self, event_id) -> h5py.File:
        """
        Returns:
            h5py.File: the handler of the hdf5 file
        """
        if event_id not in self._handler:
            self._handler[event_id] = h5py.File(self.h5py_dir / (event_id + ".h5"), "r")
        return self._handler[event_id]

    def __len__(self) -> int:
        """
        Returns:
            int: the total number of waveforms in the dataset
        """
        return len(self.index_to_waveform_id)

    def get_item_without_stack(self, idx) -> dict:
        """
        Args:
            idx (int): the index of the waveform
        Returns:
            dict: a sample containing waveform data, phase index, phase type, event id, network, station id
        """
        event_id, station_id = self.index_to_waveform_id[idx]
        handler = self.get_handler(event_id)
        waveform = torch.tensor(handler[event_id][station_id][...], dtype=torch.float32)  # type: ignore
        if torch.isnan(waveform).any():
            waveform[torch.isnan(waveform)] = 0.0
            log.info(
                f"waveform contains nan, set to 0.0, event_id: {event_id}, station_id: {station_id}"
            )
        attrs = handler[event_id][station_id].attrs  # type: ignore

        sample = {
            "key": f"{event_id}_{attrs['network']}.{station_id}",
            "data": waveform,
            "phase_index": attrs["phase_index"].tolist(),  # type: ignore
            "phase_type": attrs["phase_type"].tolist(),  # type: ignore
        }
        min_index = min(sample["phase_index"])
        start_index, end_index = (
            min_index - self.first_arrival_index_in_final_window_if_no_shift,
            min_index
            - self.first_arrival_index_in_final_window_if_no_shift
            + self.window_length_in_npts,
        )

        # used by transforms to indicate the start and end index of the window
        sample["start_index"] = start_index
        sample["end_index"] = end_index

        if self.transform:
            sample = self.transform(sample)

        # cut sample['data'] to the window length
        sample["data"] = sample["data"][:, start_index:end_index]
        # shift the phase index to the window length
        sample["phase_index"] = [i - start_index for i in sample["phase_index"]]
        # generate label, arrivals should be in order as self.phases, if not exist, use -1
        expanded_phase_index = []
        for phase in self.phases:
            if phase in sample["phase_type"]:
                expanded_phase_index.append(
                    sample["phase_index"][sample["phase_type"].index(phase)]
                )
            else:
                expanded_phase_index.append(-999999999)
        sample["phase_index"] = expanded_phase_index
        sample["phase_type"] = self.phases
        # convert phase_idnex to tensor, otherwise in dataloader, it will be converted to list with wrong shape
        # eg: using list: 3X8 if batch_size=8, using tensor: 8X3
        sample["phase_index"] = torch.tensor(sample["phase_index"])

        sample["label"] = generate_label(
            self.label_shape,
            self.label_width_in_npts,
            self.window_length_in_npts,
            sample["phase_index"],
        )
        # normalize the data before possible stacking
        sample = normalize_waveform(sample)

        return sample

    def __getitem__(self, idx) -> dict:
        """
        Args:
            idx (int): the index of the waveform
        Returns:
            dict: a sample containing waveform data, phase index, phase type, event id, network, station id
        """
        # the difference between get_item_without_stack and __getitem__ is that get_item_without_stack does not stack two waveforms
        # the stack ratio can be obtained by self.random_stack_two_waveforms_ratio
        current_sample = self.get_item_without_stack(idx)
        if torch.rand(1) < self.random_stack_two_waveforms_ratio:
            random_idx = torch.randint(0, len(self), (1,)).item()
            random_sample = self.get_item_without_stack(random_idx)
            current_sample = stack_rand(
                current_sample, random_sample, self.label_width_in_npts
            )

        # normalize the data at the end
        current_sample = normalize_waveform(current_sample)
        # remove unused keys, including start_index, end_index
        current_sample.pop("start_index", None)
        current_sample.pop("end_index", None)

        return current_sample


def split_train_test_val_for_ai4eps(
    data_dir: Path,
    ratio: List[float] = [0.9, 0.05, 0.05],
    seed: int = 3407,
    split_based_on: str = "S",
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Split the dataset into train, test, and val set
    Args:
        data_dir (Path): the directory of the dataset
        ratio (List[float], optional): the ratio of train, test, and val. Defaults to [0.9, 0.05, 0.05].
        seed (int, optional): the seed for random shuffle. Defaults to 3407.
        split_based_on (str, optional): the phase type as reference to split the dataset. Defaults to "S".
    Returns:
        Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]: train, test, and val set
    """

    def extract_unique_pairs(
        phase_picks: pd.DataFrame, split_based_on: str
    ) -> Tuple[set, set]:
        unique_pairs_set = set()
        unique_pairs_set_other = set()

        def get_unique_pairs(row):
            if row["phase_type"] == split_based_on:
                unique_pairs_set.add((row["event_id"], row["station_id"]))
            else:
                unique_pairs_set_other.add((row["event_id"], row["station_id"]))

        phase_picks.apply(get_unique_pairs, axis=1)  # type: ignore
        unique_pairs_set_other -= unique_pairs_set

        return unique_pairs_set, unique_pairs_set_other

    def split_pairs(
        unique_pairs: Tuple[set, set], ratio: List[float], seed: int
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
        unique_pairs_set, unique_pairs_set_other = unique_pairs
        rng = np.random.default_rng(seed)

        def split_and_combine(
            pairs: List,
        ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
            rng.shuffle(pairs)
            train_size = int(len(pairs) * ratio[0])
            train_test_size = int(len(pairs) * (ratio[0] + ratio[1]))
            return (
                pairs[:train_size],
                pairs[train_size:train_test_size],
                pairs[train_test_size:],
            )

        train_pairs, test_pairs, val_pairs = split_and_combine(list(unique_pairs_set))
        train_pairs_other, test_pairs_other, val_pairs_other = split_and_combine(
            list(unique_pairs_set_other)
        )

        return (
            train_pairs + train_pairs_other,
            test_pairs + test_pairs_other,
            val_pairs + val_pairs_other,
        )

    phase_picks = pd.read_csv(data_dir / "phase_picks.csv")
    # event_id column to str type
    phase_picks["event_id"] = phase_picks["event_id"].astype("str")
    unique_pairs = extract_unique_pairs(phase_picks, split_based_on)
    train_pairs, test_pairs, val_pairs = split_pairs(unique_pairs, ratio, seed)

    return train_pairs, test_pairs, val_pairs
