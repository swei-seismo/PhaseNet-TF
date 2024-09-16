""" 
visualize.py: Lightning callback for visualizing the model's predictions
"""
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning_utilities.core.rank_zero import rank_zero_only
from matplotlib import cm
from obspy.core.trace import Trace
from wandb import Image
from matplotlib.figure import Figure


class VisualizeCallback(Callback):
    """Callback for visualizing the model's predictions"""

    def __init__(
        self,
        phases: List[str] = ["P", "S", "PS"],
        dt_s: float = 0.025,
        window_length_in_npts: int = 4800,
        freqmin: float = 0.0,
        freqmax: float = 10.0,
        sgram_threshold: int = 25,
        plot_waveform_based_on: str = "P",
        if_log_train: bool = True,
        if_log_val: bool = True,
        if_log_test: bool = True,
        log_every_n_epoch: int = 10,
        show_number_of_examples: int = 10,
    ):
        """Initialize the callback

        Args:
            phases (List[str], optional): the phases to be predicted. Defaults to ["P", "S", "PS"].
            dt_s (float, optional): the sampling distance (s) of the input data. Defaults to 0.025.
            window_length_in_npts (int, optional): the length of the input data. Defaults to 4800.
            freqmin (float, optional): the minimum frequency of the wave to plot. Only used when plot_waveform_based_on is P. Defaults to 1.0.
            freqmax (float, optional): the maximum frequency of the wave to plot. Only used when plot_waveform_based_on is P. Defaults to 10.0.
            sgram_threshold (int, optional): the threshold of the spectrogram. Defaults to 500.
            plot_waveform_based_on (str, optional): the phase to plot the waveform. Defaults to "P" (0.2 to 5 HZ), can also be "all" (no filtering) or "PS" (dynamic based on PS)
            if_log_train (bool, optional): whether to log the training figures. Defaults to True.
            if_log_val (bool, optional): whether to log the validation figures. Defaults to True.
            if_log_test (bool, optional): whether to log the test figures. Defaults to True.
            log_every_n_epoch (int, optional): log the figures every n epochs. Defaults to 10.
            show_number_of_examples (int, optional): the number of examples to show. Defaults to 10.
        """
        super().__init__()
        self.if_log_train = if_log_train
        self.if_log_val = if_log_val
        self.if_log_test = if_log_test
        self.log_every_n_epoch = log_every_n_epoch
        self.show_number_of_examples = show_number_of_examples

        self.show_figs = VisualizeInfo(
            phases=phases,
            sampling_rate=int(1 / dt_s),
            x_range=[0, int(window_length_in_npts * dt_s)],
            freq_range=[freqmin, freqmax],
            global_max=False,
            sgram_threshold=sgram_threshold,
            plot_waveform_based_on=plot_waveform_based_on,
        )
        self.figs_train_store = []
        self.figs_val_store = []
        self.figs_test_store = []

    @rank_zero_only
    def _log_figs(
        self,
        batch: Dict,
        batch_idx: int,
        sgram: torch.Tensor,
        predict: torch.Tensor,
        peaks: Dict[str, List[List[List]]],
        stage: str,
        trainer: Trainer,
    ):
        """Log figs to the logger

        Args:
            batch (Dict): the current batch
            batch_idx (int): the current batch index
            sgram (torch.Tensor): the sgram of the current batch
            predict (torch.Tensor): the prediction result from the model
            peaks (Dict[str, List[List[List]]]): the peaks extracted from the peak extracter
            stage (str): current stage, either train/val/test
            trainer (Trainer): the trainer
        """
        if_log = {
            "train": self.if_log_train,
            "val": self.if_log_val,
            "test": self.if_log_test,
        }
        figs_store = {
            "train": self.figs_train_store,
            "val": self.figs_val_store,
            "test": self.figs_test_store,
        }
        if not if_log[stage]:
            return

        assert trainer.max_epochs is not None, "max_epochs is None"
        if (
            (trainer.current_epoch == trainer.max_epochs - 1)
            or (trainer.current_epoch + 1) % self.log_every_n_epoch == 0
        ) or stage == "test":
            batch_size = len(sgram)
            finished_examples = batch_size * batch_idx
            if finished_examples < self.show_number_of_examples:
                if finished_examples + batch_size < self.show_number_of_examples:
                    example_this_batch = batch_size
                    last_step = False
                else:
                    example_this_batch = (
                        self.show_number_of_examples - finished_examples
                    )
                    last_step = True

                figs = self.show_figs(batch, sgram, predict, peaks, example_this_batch)
                if figs is not None:
                    figs_store[stage].extend(figs)
                if last_step:
                    trainer.logger.experiment.log(  # type: ignore
                        {f"figs_{stage}": [Image(item) for item in figs_store[stage]]}
                    )
                    for each in figs_store[stage]:
                        plt.close(each)
                    figs_store[stage].clear()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
    ) -> None:
        if outputs is None or ("sgram_power" not in outputs):
            return
        self._log_figs(
            batch,
            batch_idx,
            outputs["sgram_power"],
            outputs["predict"],
            outputs["peaks"],
            "train",
            trainer,
        )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
    ) -> None:
        if outputs is None or ("sgram_power" not in outputs):
            return
        self._log_figs(
            batch,
            batch_idx,
            outputs["sgram_power"],
            outputs["predict"],
            outputs["peaks"],
            "val",
            trainer,
        )

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
    ) -> None:
        if outputs is None or ("sgram_power" not in outputs):
            return
        self._log_figs(
            batch,
            batch_idx,
            outputs["sgram_power"],
            outputs["predict"],
            outputs["peaks"],
            "test",
            trainer,
        )


class VisualizeInfo:
    def __init__(
        self,
        phases: List[str],
        sampling_rate: int,
        x_range: List[int],
        freq_range: List[float],
        global_max: bool = False,
        sgram_threshold: Optional[int] = None,
        plot_waveform_based_on: str = "P",
    ) -> None:
        """The class to visualize the results

        Args:
            phases (List[str]): the phases to plot
            sampling_rate (int): the sampling rate of the data
            x_range (List[int]): the range of the x axis
            freq_range (List[float]): the range of the frequency axis
            global_max (bool, optional): whether to use the global max to normalize the sgram. Defaults to False.
            sgram_threshold (Optional[int], optional): the threshold to plot the sgram. Defaults to None.
            plot_waveform_based_on (str, optional): the phase to plot the waveform based on. Defaults to "P".
        """
        self.phases = phases
        self.sampling_rate = sampling_rate
        self.x_range = x_range
        self.freq_range = freq_range
        self.global_max = global_max
        self.sgram_threshold = sgram_threshold
        self.plot_waveform_based_on = plot_waveform_based_on
        self.ps_idx = None
        for iphase, phase in enumerate(self.phases):
            if phase == "PS":
                self.ps_idx = iphase
                break

    def __call__(
        self,
        input_batch: Dict,
        sgram_batch: torch.Tensor,
        predict_batch: torch.Tensor,
        peaks_batch: Dict[str, List[List[List]]],
        cur_example_num: int = 0,
    ) -> Optional[List[Figure]]:
        """The function to visualize the results

        Args:
            input_batch (Dict): the input batch
            sgram_batch (torch.Tensor): the sgram of the input batch
            predict_batch (torch.Tensor): the prediction of the input batch
            peaks_batch (Dict[str, List[List[List]]]): the peaks extracted from the peak extracter
            cur_example_num (int, optional): the number of examples to plot. Defaults to 0.

        Returns:
            Optional[List[Figure]]: the figures to plot
        """
        if cur_example_num == 0:
            return None
        figs = []
        # * load data
        data_batch: torch.Tensor = input_batch["data"].detach().cpu()
        arrivals_batch: torch.Tensor = input_batch["phase_index"].detach().cpu()
        key_batch: List[str] = input_batch["key"]
        label_batch: torch.Tensor = input_batch["label"].detach().cpu()
        sgram_batch = sgram_batch.detach().cpu()
        predict_batch = predict_batch.detach().cpu()

        # * plot each batch
        batch_size = data_batch.shape[0]
        for ibatch in range(min(batch_size, cur_example_num)):
            # * prepare
            # generate figures for each ibatch
            data, arrivals, key, sgram, label, predict = (
                data_batch[ibatch],
                arrivals_batch[ibatch],
                key_batch[ibatch],
                sgram_batch[ibatch],
                label_batch[ibatch],
                predict_batch[ibatch],
            )
            peaks_idx, peaks_val = (
                peaks_batch["arrivals"][ibatch],
                peaks_batch["amps"][ibatch],
            )
            # here we assume the data has been procesed
            fig, axes = plt.subplots(
                8,
                1,
                sharex=True,
                figsize=(20, 34),
                gridspec_kw={"wspace": 0, "hspace": 0},
            )
            axes[0].set_title(key, fontsize=24)
            x = np.arange(data.shape[1]) / self.sampling_rate
            # the max of sgram plot is after 5s of P to 10s of P
            # but we should care if specify the max threshold
            # * max threshold
            vmax = []
            if self.sgram_threshold == None:
                p_arrival = min(arrivals)  # type: ignore
                for i in range(3):
                    if self.global_max:
                        i = 0
                    if (
                        p_arrival + self.sampling_rate * 5 >= 0
                        and p_arrival + self.sampling_rate * 15 <= sgram.shape[-1]
                    ):
                        vmax.append(
                            torch.max(
                                sgram[i][
                                    :,
                                    p_arrival
                                    + self.sampling_rate * 5 : p_arrival
                                    + self.sampling_rate * 15,
                                ]
                            )
                        )
                    else:
                        vmax.append(30)
            else:
                vmax = [self.sgram_threshold] * 3
                # print(torch.max(sgram[2]), "++++")
                # p_arrival = min(arrivals)
                # if p_arrival+self.sampling_rate * 5 >= 0 and p_arrival+self.sampling_rate*15 <= sgram.shape[-1]:
                #     print(torch.max(sgram[2][:, p_arrival+self.sampling_rate *
                #                              5:p_arrival+self.sampling_rate*15]), "@@@@")
            max_scale = torch.max(torch.abs(data))
            # * plot sgram
            # 1 component
            axes[1].imshow(
                sgram[0],
                aspect="auto",
                cmap="jet",
                origin="lower",
                vmin=0,
                vmax=vmax[0],
                extent=self.x_range + self.freq_range,
            )
            # 2 component
            axes[3].imshow(
                sgram[1],
                aspect="auto",
                cmap="jet",
                origin="lower",
                vmin=0,
                vmax=vmax[1],
                extent=self.x_range + self.freq_range,
            )
            # Z component
            axes[5].imshow(
                sgram[2],
                aspect="auto",
                cmap="jet",
                origin="lower",
                vmin=0,
                vmax=vmax[2],
                extent=self.x_range + self.freq_range,
            )
            # * ps freq line (raw)
            if "ps_freqs" in input_batch:
                for iax in [1, 3, 5]:
                    axes[iax].hlines(
                        y=input_batch["ps_freqs"][ibatch].detach().cpu(),
                        xmin=0,
                        xmax=sgram.shape[-1] / self.sampling_rate,
                        colors="w",
                        ls="-",
                        lw=1,
                    )
            # * ps freq range line (predict)
            # * should plot max ps loc
            # * ps x range -2s +5s
            fs, fe = None, None
            if self.ps_idx and len(peaks_idx[self.ps_idx]) > 0:
                freq_win_length = 12
                freq_range = [10, 64]
                ps_idx = peaks_idx[self.ps_idx][np.argmax(peaks_val[self.ps_idx])]
                ps_idx_start = ps_idx - int(1 * self.sampling_rate)
                ps_idx_end = ps_idx + int(3 * self.sampling_rate)
                noise_idx_start = ps_idx - int(5 * self.sampling_rate)
                noise_idx_end = ps_idx - int(1 * self.sampling_rate)
                if ps_idx_start < 0:
                    ps_idx_start = 0
                if ps_idx_end > sgram.shape[-1]:
                    ps_idx_end = sgram.shape[-1]
                fs, fe = spectrogram_extract_ps_freq(
                    sgram,
                    ps_idx_start,
                    ps_idx_end,
                    freq_range,
                    freq_win_length,
                    noise_idx_start,
                    noise_idx_end,
                )
                fs = (
                    fs / sgram.shape[-2] * (self.freq_range[1] - self.freq_range[0])
                    + self.freq_range[0]
                )
                fe = (
                    fe / sgram.shape[-2] * (self.freq_range[1] - self.freq_range[0])
                    + self.freq_range[0]
                )
                # plot
                for iax in [1, 3, 5]:
                    axes[iax].hlines(
                        y=fs,
                        xmin=0,
                        xmax=sgram.shape[-1] / self.sampling_rate,
                        colors="w",
                        ls="--",
                        lw=1,
                    )
                    axes[iax].hlines(
                        y=fe,
                        xmin=0,
                        xmax=sgram.shape[-1] / self.sampling_rate,
                        colors="w",
                        ls="--",
                        lw=1,
                    )

            # * plot wave
            # put it here as we may need PS filter range
            fig_label = {
                "all": " (no further filtering)",
                "P": " (0.2->5 HZ)",
                "PS": " (dynamic based on PS)",
            }

            filtered, status = self.filter_waveform(data[0, :], fs, fe)
            axes[0].plot(x, filtered, c="black", lw=1, label="1" + fig_label[status])
            axes[0].legend()
            filtered, status = self.filter_waveform(data[1, :], fs, fe)
            axes[2].plot(x, filtered, c="black", lw=1, label="2" + fig_label[status])
            axes[2].legend()
            filtered, status = self.filter_waveform(data[2, :], fs, fe)
            axes[4].plot(x, filtered, c="black", lw=1, label="Z" + fig_label[status])
            axes[4].legend()

            # * plot predictions and targets
            color = cm.rainbow(np.linspace(0, 1, len(self.phases)))  # type: ignore
            for i, each_phase in enumerate(self.phases):
                axes[6].plot(
                    x, label[i + 1, :].numpy(), "--", c=color[i], label=each_phase
                )
                axes[7].plot(
                    x, predict[i + 1, :].numpy(), "--", c=color[i], label=each_phase
                )
                for idx in [0, 2, 4]:
                    if 0 < arrivals[i] < sgram.shape[-1]:
                        axes[idx].vlines(
                            x=arrivals[i] / self.sampling_rate,
                            ymin=-max_scale,
                            ymax=max_scale,
                            colors=color[i],
                            ls="--",
                            lw=1,
                        )
                    axes[idx].margins(0)
                    axes[idx].set_ylabel("Amplitude", fontsize=18)
                for idx in [1, 3, 5]:
                    if 0 < arrivals[i] < sgram.shape[-1]:
                        axes[idx].vlines(
                            x=arrivals[i] / self.sampling_rate,
                            ymin=self.freq_range[0],
                            ymax=self.freq_range[1],
                            colors=color[i],
                            ls="--",
                            lw=1,
                        )
                    axes[idx].set_ylabel("Frequency (HZ)", fontsize=18)
                # * plot peaks
                peaksx, peaksy = peaks_idx[i], peaks_val[i]
                for px, py in zip(peaksx, peaksy):
                    axes[7].scatter(
                        px / self.sampling_rate, py, s=50, color="k", marker="+"
                    )
            axes[6].plot(x, label[0, :].numpy(), "--", c="black", label="Noise")
            axes[7].plot(x, predict[0, :].numpy(), "--", c="black", label="Noise")
            axes[7].set_xlabel("time (s)", fontsize=24)
            axes[7].legend()

            figs.append(fig)
        return figs

    def filter_waveform(
        self, data: torch.Tensor, fs: Optional[float] = None, fe: Optional[float] = None
    ) -> Tuple[torch.Tensor, str]:
        """
        filter waveform based on the freq range info

        Args:
            data (torch.Tensor): waveform
            fs (Optional[float], optional): freq start. Defaults to None.
            fe (Optional[float], optional): freq end. Defaults to None.

        Returns:
            Tuple[torch.Tensor, str]: filtered waveform, status (plot_waveform_based_on)
        """
        if self.plot_waveform_based_on == "all":
            return data, "all"

        wave = Trace(data=data.numpy())
        wave.stats.sampling_rate = self.sampling_rate
        # filter based on the freq range info
        # P 0.2 -> 5 HZ, PS: dynamic
        if self.plot_waveform_based_on == "P":
            wave.filter("bandpass", freqmin=0.2, freqmax=5, corners=4, zerophase=False)
        elif self.plot_waveform_based_on == "PS":
            if fs == None or fe == None:
                # no fs/fe. no PS picks, return origional waveform
                return data, "all"
            else:
                wave.filter(
                    "bandpass", freqmin=fs, freqmax=fe, corners=4, zerophase=False
                )
        else:
            raise Exception(f"no support for {self.plot_waveform_based_on=}")
        return torch.tensor(wave.data), self.plot_waveform_based_on


def spectrogram_extract_ps_freq(
    sgram_all_phases: torch.Tensor,
    y_start: int,
    y_end: int,
    x_range: List[int],
    x_length: int,
    noise_idx_start: int,
    noise_idx_end: int,
):
    """
    extract the freq range of PS based on the sgram

    Args:
        sgram_all_phases (torch.Tensor): sgram of all phases
        y_start (int): start index of y
        y_end (int): end index of y
        x_range (List[int]): x range
        x_length (int): x length
        noise_idx_start (int): noise idx start
        noise_idx_end (int): noise idx end

    Returns:
        Tuple[int, int]: freq range of PS
    """
    # * given sgram, and y (time) indexes start and end, find x (freq) indexes start and end
    # sgram = sgram_all_phases.sum(axis=0)
    # we want to only consider the R component when finding PS
    # 0:R, 1:T, 2:Z  Since we are looking at PS, we should use the horizontal component, so it should be some l2 mean
    sgram = torch.sqrt(sgram_all_phases[0] ** 2 + sgram_all_phases[1] ** 2)
    themax = 0
    s, e = 0, 0
    for x_start in range(x_range[0], x_range[1]):
        cur = sgram[x_start : x_start + x_length, y_start:y_end].sum()
        if cur > themax:
            # s, e = x_start, x_start+x_length
            themax = cur
    # now the range is s->e, value is the max
    # we only consider the range max/10 -> max, but with best SNR
    ratio_global = float("-inf")
    for x_start in range(x_range[0], x_range[1]):
        cur = sgram[x_start : x_start + x_length, y_start:y_end].sum()
        noise = sgram[x_start : x_start + x_length, noise_idx_start:noise_idx_end].sum()
        if cur >= themax / 2:
            cur_level = cur / (y_end - y_start)
            noise_level = noise / (noise_idx_end - noise_idx_start)
            ratio = cur_level / noise_level
            if ratio > ratio_global:
                ratio_global = ratio
                s, e = x_start, x_start + x_length
    return s, e
