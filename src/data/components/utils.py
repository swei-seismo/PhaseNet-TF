"""
utils.py: this file contains some utility functions for data processing
"""
from typing import List

import torch


def generate_label(
    label_shape: str, label_width: int, wave_length: int, arrivals: List[int] = []
) -> torch.Tensor:
    """
    Args:
        label_shape (str): the shape of the label, can be "gaussian" or "triangle"
        label_width (int): the width of the label
        wave_length (int): the shape of the data (as the same shape as the waveform)
        arrivals (List[int], optional): list of arrival indices. Defaults to an empty list.
    Returns:
        torch.Tensor: generated label tensor
    """
    res = torch.zeros(len(arrivals) + 1, wave_length)
    if label_shape == "gaussian":
        label_window = torch.exp(
            -((torch.arange(-label_width // 2, label_width // 2 + 1)) ** 2)
            / (2 * (label_width / 6) ** 2)
        )
    elif label_shape == "triangle":
        label_window = 1 - torch.abs(
            2 / label_width * (torch.arange(-label_width // 2, label_width // 2 + 1))
        )
    else:
        raise Exception(f"label shape {label_shape} is not supported!")

    for i, idx in enumerate(arrivals):
        # the index for arrival times
        # if idx==-1, then start<0, so the label will be all zeros (the case no phase arrival)
        start = idx - label_width // 2
        end = idx + label_width // 2 + 1
        if start >= 0 and end <= res.shape[1]:
            res[i + 1, start:end] = label_window

    # the first row represents the noise label
    res[0, :] = 1 - torch.sum(res, 0)
    return res


def stack_rand(sample: dict, random_sample: dict, label_width: int) -> dict:
    """
    Args:
        sample (dict): the sample dict
        random_sample (dict): the random sample dict
        label_width (int): the width of the label
    Returns:
        dict: the stacked sample dict
    """
    data, random_data = sample["data"], random_sample["data"]
    phase_index, random_phase_index = (
        sample["phase_index"],
        random_sample["phase_index"],
    )
    label, random_label = sample["label"], random_sample["label"]

    # verify any pair of phase_index and random_phase_index's ditsance is larger than label_width
    for i in phase_index:
        for j in random_phase_index:
            if i < 0 or j < 0:
                # if i or j is -1, then the phase arrival is not detected
                continue
            if abs(i - j) < label_width:
                return sample

    # stack two waveforms and labels
    stacked_data = data + random_data
    stacked_label = label + random_label

    # for label, we need to make sure the sum of possibility to be 1
    stacked_label = torch.clamp_max(stacked_label, 1.0)
    stacked_label[0, :] = 1 - torch.sum(stacked_label[1:], 0)

    sample.update({"data": stacked_data, "label": stacked_label})
    return sample


def normalize_waveform(sample: dict) -> dict:
    """
    Args:
        sample (dict): the sample dict
    Returns:
        dict: the normalized sample dict
    """
    data = sample["data"]
    mean_vals = torch.mean(data, dim=1, keepdim=True)
    data = data - mean_vals
    max_std_val = torch.max(torch.std(data, dim=1))
    if max_std_val == 0:
        max_std_val = torch.ones(1)
    data = data / max_std_val

    sample.update({"data": data})
    return sample
