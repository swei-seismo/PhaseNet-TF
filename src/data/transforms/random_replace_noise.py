"""
random_replace_noise.py: this file contains the random replace noise transform.
"""
import torch


class RandomReplaceNoise:
    def __init__(self, replace_ratio=0.5):
        self.replace_ratio = replace_ratio

    def __call__(self, sample: dict) -> dict:
        """
        Args:
            sample (dict): a sample containing waveform data, phase index, phase type, event id, network, station id
        Returns:
            dict: a sample containing waveform data, phase index, phase type, event id, network, station id
        """
        final_window_length = sample["end_index"] - sample["start_index"]
        # cut the first final_window_length points of the waveform as the noise
        # so replace start_index->end_index with noise
        noise = sample["data"][:, :final_window_length]
        if torch.rand(1) < self.replace_ratio:
            sample["data"][:, sample["start_index"] : sample["end_index"]] = noise
            sample["phase_index"] = [-999999999 for _ in sample["phase_index"]]
        return sample
