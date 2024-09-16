"""
ai4eps_datamodule.py: this file contains the datamodule for the dataset following standard AI4EPS format.
"""
from pathlib import Path
from typing import List, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from src.data.components.ai4eps import Ai4epsDataset, split_train_test_val_for_ai4eps
from src.data.transforms import RandomReplaceNoise, RandomShift


class Ai4epsDataModule(LightningDataModule):
    """
    DataModule for the dataset following standard AI4EPS format.
    """

    def __init__(
        self,
        # data dir and split params
        # due to hydra omegaconf limitation, we cannot use Path type here
        data_dir: str = "dataset/",
        train_val_test_split_ratio: List[float] = [0.90, 0.05, 0.05],
        train_val_test_split_seed: int = 3407,
        # dataset params
        noise_replace_ratio: float = 0.0,
        label_shape: str = "gaussian",
        label_width_in_npts: int = 120,
        window_length_in_npts: int = 4800,
        phases: List[str] = ["P", "S", "PS"],
        first_arrival_index_in_final_window_if_no_shift: int = 400,
        random_stack_two_waveforms_ratio=0.0,
        # data loader params
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [
                RandomShift(),
                RandomReplaceNoise(replace_ratio=noise_replace_ratio),
            ]
        )

        # train/val/test split
        self.train_ids, self.val_ids, self.test_ids = split_train_test_val_for_ai4eps(
            Path(data_dir), train_val_test_split_ratio, train_val_test_split_seed
        )

        # init the dataset to be None
        self.data_train, self.data_val, self.data_test = None, None, None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = Ai4epsDataset(
                Path(self.hparams["data_dir"]),
                self.train_ids,
                self.transforms,
                self.hparams["label_shape"],
                self.hparams["label_width_in_npts"],
                self.hparams["window_length_in_npts"],
                self.hparams["phases"],
                self.hparams["first_arrival_index_in_final_window_if_no_shift"],
                self.hparams["random_stack_two_waveforms_ratio"],
            )
            self.data_val = Ai4epsDataset(
                Path(self.hparams["data_dir"]),
                self.val_ids,
                None,
                self.hparams["label_shape"],
                self.hparams["label_width_in_npts"],
                self.hparams["window_length_in_npts"],
                self.hparams["phases"],
                self.hparams["first_arrival_index_in_final_window_if_no_shift"],
                0.0,
            )
            self.data_test = Ai4epsDataset(
                Path(self.hparams["data_dir"]),
                self.test_ids,
                None,
                self.hparams["label_shape"],
                self.hparams["label_width_in_npts"],
                self.hparams["window_length_in_npts"],
                self.hparams["phases"],
                self.hparams["first_arrival_index_in_final_window_if_no_shift"],
                0.0,
            )

    def train_dataloader(self):
        assert (
            self.data_train is not None
        ), "self.data_train is None. Please call self.setup() first."
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=True,
            # persistent_workers=True,
        )

    def val_dataloader(self):
        assert (
            self.data_val is not None
        ), "self.data_val is None. Please call self.setup() first."
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
            # persistent_workers=True,
        )

    def test_dataloader(self):
        assert (
            self.data_test is not None
        ), "self.data_test is None. Please call self.setup() first."
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
            # persistent_workers=True,
        )


if __name__ == "__main__":
    _ = Ai4epsDataModule()
