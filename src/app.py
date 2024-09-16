"""
app.py: serve the model as a REST API
"""
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import pandas as pd
import pyrootutils
import torch
import uvicorn
from fastapi import FastAPI
from lightning import LightningModule
from omegaconf import DictConfig
from pydantic import BaseModel

from src import utils
from src.inference.single_inference import single_inference

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #


log = utils.get_pylogger(__name__)
app = FastAPI()

# global variables
model: Optional[LightningModule] = None
dt_s: float = 0.0
model_phases: List[str] = []
window_length_in_npts: int = 0
hop_length_in_npts: int = 0
sensitive_distances_in_seconds: float = 0.0


@hydra.main(version_base="1.3", config_path="../configs", config_name="app.yaml")
def main(cfg: DictConfig) -> None:
    global model, dt_s, model_phases, window_length_in_npts, hop_length_in_npts, sensitive_distances_in_seconds
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    if not Path(cfg.get("ckpt_path", "")).is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {cfg.ckpt_path}")

    # load model from checkpoint
    log.info(f"Instantiating model <{cfg.model._target_}>")
    Model = hydra.utils.get_class(cfg.model._target_)

    model = Model.load_from_checkpoint(  # type: ignore
        cfg.ckpt_path, map_location=torch.device(cfg.get("device", "cpu"))
    )

    # load other parameters
    dt_s = cfg.model.get("dt_s", 0.025)
    model_phases = cfg.model.get("phases", ["P", "S", "PS"])
    window_length_in_npts = cfg.model.get("window_length_in_npts", 4800)
    hop_length_in_npts = cfg.app.get("hop_length_in_npts", 2400)
    sensitive_distances_in_seconds = cfg.model.get(
        "extract_peaks_sensitive_distances_in_seconds", 5.0
    )

    uvicorn.run(app, host=cfg.app.host, port=cfg.app.port)


class PredictionRequest(BaseModel):
    """
    Request body for prediction endpoint.

    Args:
        id (List[str]): List of IDs.
        timestamp (List[str]): List of timestamps.
        vec (List[List[List[float]]]): List of 3 X NPTS points.
    """

    id: List[str]
    timestamp: List[str]
    vec: List[List[List[float]]]
    extract_phases: List[str]
    extract_phase_sensitivity: List[float]


class PredictResponse(BaseModel):
    """
    Response body for prediction endpoint.

    Args:
        id (List[str]): List of IDs.
        possibility (List[Dict[str, List[float]]]): List of possibility, the keys of the dict are phases, and the values are the possibility of the phases.
        arrivals (List[Dict[str, List[float]]]): List of arrivals, the keys of the dict are phases, and the values are the arrival index of the phases.
        amps (List[Dict[str, List[float]]]): List of amps, the keys of the dict are phases, and the values are the amps of the phases possibility.
        arrival_times (List[Dict[str, List[float]]]): List of arrival times, the keys of the dict are phases, and the values are the arrival times of the phases.
    """

    id: List[str]
    possibility: List[Dict[str, List[float]]]
    arrivals: List[Dict[str, List[float]]]
    amps: List[Dict[str, List[float]]]
    arrival_times: List[Dict[str, List[str]]]


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictionRequest) -> PredictResponse:
    """
    Prediction endpoint.

    Args:
        request (PredictionRequest): Request body.

    Returns:
        PredictResponse: List of predictions, with keys "arrivals", "amps", "arrival_times", and "possibility".
    """
    assert model is not None, "Model is not loaded yet."
    number_of_traces = len(request.vec)
    res = []
    for itrace in range(number_of_traces):
        # run inference
        model.eval()
        pred = single_inference(
            model=model,
            data=torch.tensor(request.vec[itrace], dtype=torch.float32),
            extract_phases=request.extract_phases,
            extract_phase_sensitivity=request.extract_phase_sensitivity,
            model_phases=model_phases,
            window_length_in_npts=window_length_in_npts,
            hop_length_in_npts=hop_length_in_npts,
            dt_s=dt_s,
            sensitive_distances_in_seconds=sensitive_distances_in_seconds,
        )

        # the keys of the dict are arrivals, amps, and possibility
        # create a new arrival_times list from the arrivals dict and timestamp
        # example request.timestamp[itrace]: 2020-01-01 00:00:00.000
        start_time = pd.Timestamp(request.timestamp[itrace])
        pred["arrival_times"] = {}
        for phase in pred["arrivals"]:
            pred["arrival_times"][phase] = [
                start_time + pd.Timedelta(seconds=arrival * dt_s)
                for arrival in pred["arrivals"][phase]
            ]
            pred["arrival_times"][phase] = [
                arrival.strftime("%Y-%m-%d %H:%M:%S.%f")
                for arrival in pred["arrival_times"][phase]
            ]

        # append to res
        res.append(pred)

    # convert to PredictResponse
    res = PredictResponse(
        id=request.id,
        possibility=[pred["possibility"] for pred in res],
        arrivals=[pred["arrivals"] for pred in res],
        amps=[pred["amps"] for pred in res],
        arrival_times=[pred["arrival_times"] for pred in res],
    )

    return res


if __name__ == "__main__":
    main()
