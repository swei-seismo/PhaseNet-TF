"""
app.py: serve the model as a REST API
"""
from pathlib import Path

import hydra
import pyrootutils
import torch
from omegaconf import DictConfig

from src import utils

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


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="export_onnx.yaml"
)
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    if not Path(cfg.get("ckpt_path", "")).is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {cfg.ckpt_path}")

    # load model from checkpoint
    log.info(f"Instantiating model <{cfg.model._target_}>")
    Model = hydra.utils.get_class(cfg.model._target_)

    model = Model.load_from_checkpoint(cfg.ckpt_path, map_location=torch.device("cpu"))  # type: ignore
    input_shape = torch.randn(1, 3, cfg.model.get("window_length_in_npts", 4800))
    onnx_path = cfg.get("onnx_path", None)
    if onnx_path is not None:
        log.info(
            f"Exporting model to ONNX format: {onnx_path} with input shape {input_shape.shape}"
        )
        model.to_onnx(
            onnx_path,
            input_shape,
            export_params=True,
            input_names=["waveform"],
            output_names=["prediction", "spectrogram"],
        )
    else:
        log.info("Skipping ONNX export as no ONNX export path was provided")


if __name__ == "__main__":
    main()
