""" 
tsindex_datamodule.py: this file contains the datamodule for the dataset following obspy's TSIndex database, designed for batch inference.
"""
from pathlib import Path
from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.tsindex import TSIndexDataset


class TSIndexDataModule(LightningDataModule):
    def __init__(
        self,
        inference_requirement_path: str,
        inference_output_dir: str,
        tsindex_database_path: str,
        datapath_name_replace: Optional[list] = None,
        continuous_window_time_in_sec: float = 3600,
        # tranform params
        window_length_in_npts: int = 4800,
        hop_length_in_npts: int = 2400,
        # data loader params
        num_workers: int = 4,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.dataset:
            self.dataset = TSIndexDataset(
                Path(self.hparams["inference_requirement_path"]),
                Path(self.hparams["inference_output_dir"]),
                Path(self.hparams["tsindex_database_path"]),
                self.hparams["datapath_name_replace"],
                self.hparams["continuous_window_time_in_sec"],
                # we use None here as we will call single inference function, which already has a transform
                None,
            )

    def predict_dataloader(self):
        assert self.dataset is not None, "Please call setup() first."
        loader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
        )
        return loader
