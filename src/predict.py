"""
predict.py: serve the model as a REST API
"""
import warnings
from pathlib import Path
from typing import List

import hydra
import pyrootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src import utils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# ignore warnings
warnings.filterwarnings("ignore", ".*SELECT statement has a cartesian product*")

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


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    if not Path(cfg.get("ckpt_path", "")).is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {cfg.ckpt_path}")

    # load model from checkpoint
    log.info(f"Instantiating model <{cfg.model._target_}>")
    Model = hydra.utils.get_class(cfg.model._target_)

    model: LightningModule = Model.load_from_checkpoint(  # type: ignore
        cfg.ckpt_path, map_location=torch.device("cpu")
    )

    # load data module
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # callbacks containing inference step and writer
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    # do inference
    log.info("Starting inference...")
    model.eval()
    trainer.predict(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
