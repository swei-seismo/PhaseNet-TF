""" 
batch_inference_step.py: Lightning callback to run predict_step when doing batch inference.
"""
import torch
from lightning import LightningModule
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer import Trainer
from src.inference.single_inference import single_inference
from typing import List


class BatchInferenceStepCallback(Callback):
    def __init__(
        self,
        extract_phases: List[str] = ["P", "S"],
        model_phases: List[str] = ["P", "S", "PS"],
        extract_phase_sensitivity: List[float] = [0.5, 0.5],
        dt_s: float = 0.025,
        window_length_in_npts: int = 4800,
        hop_length_in_npts: int = 2400,
        sensitive_distances_in_seconds: float = 5.0,
    ) -> None:
        self.extract_phases = extract_phases
        self.model_phases = model_phases
        self.extract_phase_sensitivity = extract_phase_sensitivity
        self.dt_s = dt_s
        self.window_length_in_npts = window_length_in_npts
        self.hop_length_in_npts = hop_length_in_npts
        self.sensitive_distances_in_seconds = sensitive_distances_in_seconds

    def on_predict_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if "data" not in batch:
            return
        # update batch dict with output from single inference
        with torch.no_grad():
            data = batch["data"][0]
            inference_result = single_inference(
                model=pl_module,
                data=data,
                extract_phases=self.extract_phases,
                model_phases=self.model_phases,
                extract_phase_sensitivity=self.extract_phase_sensitivity,
                dt_s=self.dt_s,
                window_length_in_npts=self.window_length_in_npts,
                hop_length_in_npts=self.hop_length_in_npts,
                sensitive_distances_in_seconds=self.sensitive_distances_in_seconds,
            )
        # update batch dict with name predictions
        batch["predictions"] = [inference_result]
