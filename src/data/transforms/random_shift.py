""" 
random_shift.py: this file contains the random shift transform.
"""

import torch


class RandomShift:
    def __init__(self):
        pass

    def __call__(self, sample: dict) -> dict:
        """
        Args:
            sample (dict): a sample containing waveform data, phase index, phase type, event id, network, station id
        Returns:
            dict: a sample containing waveform data, phase index, phase type, event id, network, station id
        """
        first_arrival_index = min(sample["phase_index"])
        last_arrival_index = max(sample["phase_index"])
        # our target final window will be from start_index to end_index
        # randomly shift sample['data'], so that there is at least one arrival in the final window
        shift = torch.randint(
            sample["start_index"] - last_arrival_index,
            sample["end_index"] - first_arrival_index,
            (1,),
        ).item()
        assert type(shift) is int
        sample["data"] = torch.roll(sample["data"], shift, dims=1)
        # shift the phase index
        sample["phase_index"] = [i + shift for i in sample["phase_index"]]
        return sample
