""" 
single_inference.py: Do the model inference for a single input, usually from the API.
"""
from typing import Dict, List

import torch
from lightning.pytorch import LightningModule

from src.models.utils.peaks import extract_peaks

from .base_inference import (
    BatchToWaveformTransform,
    PadZeroTransform,
    BatchNormalizeTransform,
    WaveformToBatchTransform,
)


def single_inference(
    model: LightningModule,
    data: torch.Tensor,
    extract_phases: List[str],
    model_phases: List[str],
    extract_phase_sensitivity: List[float],
    dt_s: float,
    window_length_in_npts: int,
    hop_length_in_npts: int,
    sensitive_distances_in_seconds: float,
) -> Dict[str, Dict[str, List]]:
    """Do the model inference for a single input, usually from the API.

    Args:
        model (LightningModule): model
        data (torch.tensor): shape (3, npts)
        extract_phases (List[str]): phases to extract
        model_phases (List[str]): phases in the model
        extract_phase_sensitivity (List[float]): phase sensitivity for each phase in extract_phases, should be the same length as extract_phases
        dt_s (float): delta t in seconds
        window_length_in_npts (int): window length in number of points
        hop_length_in_npts (int): hop (step) length in number of points

    Returns:
        dict: dict of arrivals, amps, and inference tensor
    """
    # initiliaze the transforms
    pad_zero_transform = PadZeroTransform(window_length_in_npts, hop_length_in_npts)
    # sliding_window_normalize_transform = SlidingWindowNormalizeTransform(
    #     window_length_in_npts, hop_length_in_npts
    # )
    waveform_to_batch_transform = WaveformToBatchTransform(
        window_length_in_npts, hop_length_in_npts
    )
    batch_normalize_transform = BatchNormalizeTransform()
    batch_to_waveform_transform = BatchToWaveformTransform(
        window_length_in_npts, hop_length_in_npts
    )

    # prepare the input tensor
    data = pad_zero_transform(data)
    # data = sliding_window_normalize_transform(data)
    data = waveform_to_batch_transform(data)
    data = batch_normalize_transform(data)

    # do the inference, assume model is already in eval mode
    with torch.no_grad():
        output, _ = model(data)
        output = torch.nn.functional.softmax(output, dim=1)
        output = batch_to_waveform_transform(output)

    # add batch axis to output
    output = output.unsqueeze(0)
    # for phases in model_phases but not in extract_phases, set the phase_sensitivity to 1.1, so no peaks will be extracted
    phase_sensitivity = {
        phase: 1.1 for phase in model_phases if phase not in extract_phases
    }
    phase_sensitivity.update(
        {
            phase: sensitivity
            for phase, sensitivity in zip(extract_phases, extract_phase_sensitivity)
        }
    )

    sensitive_distance = {
        phase: sensitive_distances_in_seconds for phase in model_phases
    }
    sampling_rate = int(1 / dt_s)

    # extract peaks
    peaks = extract_peaks(
        predict_batch=output,
        phases=model_phases,
        sensitive_heights=phase_sensitivity,
        sensitive_distances=sensitive_distance,
        sampling_rate=sampling_rate,
    )

    # only keep the phases we want in extract_phases, also remove the unnecessary batch axis in peaks
    arrivals, amps = {}, {}
    output_keep_only_extract_phases = {
        "noise": output[0][0].detach().cpu().numpy().tolist()
    }
    for iphase, phase in enumerate(model_phases):
        if phase in extract_phases:
            arrivals[phase] = peaks["arrivals"][0][iphase].tolist()
            amps[phase] = peaks["amps"][0][iphase].tolist()
            output_keep_only_extract_phases[phase] = (
                output[0][iphase + 1].detach().cpu().numpy().tolist()
            )

    return {
        "arrivals": arrivals,
        "amps": amps,
        "possibility": output_keep_only_extract_phases,
    }
