from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.signal import find_peaks


def extract_peaks_single_phase(
    signal: np.ndarray,
    sensitive_height: float = 0.5,
    sensitive_distance: float = 0.5,
    sampling_rate: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    distance = int(sensitive_distance * sampling_rate)
    peaks_idx, _ = find_peaks(signal, height=sensitive_height, distance=distance)
    peaks_amp = signal[peaks_idx]
    return peaks_idx, peaks_amp


def extract_peaks(
    predict_batch: torch.Tensor,
    phases: List[str],
    sensitive_heights: Dict[str, float],
    sensitive_distances: Dict[str, float],
    sampling_rate: int = 40,
) -> Dict[str, List[List[np.ndarray]]]:
    """Extract peaks from the predicted batch

    Args:
        predict_batch (torch.Tensor): shape (batch, channel, time)
        phases (List[str]): list of phases
        sensitive_heights (Dict[str, float]): sensitive height for each phase
        sensitive_distances (Dict[str, float]): sensitive distance for each phase
        sampling_rate (int, optional): sampling rate. Defaults to 40.

    Returns:
        Dict[str, List[List[List]]]: dict of peaks idx and peaks amp
    """
    # * predict_batch.shape => batch, channel, time
    batch_size = len(predict_batch)
    # * in the arrivals of data item, it has the same (len_batch,len_phases)
    # * we might detect multiple peaks for each phase, so our shape should be (len_batch,len_phases,len_peaks), here len_peaks might be dynamic
    predict_arrivals = []
    predict_arrival_amps = []
    for ibatch in range(batch_size):
        predict_arrivals.append([])
        predict_arrival_amps.append([])
        for iphase, phase in enumerate(phases):
            # find peaks
            peaks_idx, peaks_amp = extract_peaks_single_phase(
                signal=predict_batch[ibatch][iphase + 1]
                .detach()
                .to(dtype=torch.float32)
                .cpu()
                .numpy(),
                sensitive_height=sensitive_heights[phase],
                sensitive_distance=sensitive_distances[phase],
                sampling_rate=sampling_rate,
            )
            predict_arrivals[-1].append(peaks_idx)
            predict_arrival_amps[-1].append(peaks_amp)
    return {"arrivals": predict_arrivals, "amps": predict_arrival_amps}
