""" 
batch_inference_writer.py: Lightning callback to write batch inference results to file.
"""
from pathlib import Path
from typing import Dict, List

import numpy as np
from lightning import LightningModule
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.trainer import Trainer
from obspy import Stream, Trace, UTCDateTime


class InferenceWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        extract_phases: List[str] = ["P", "S"],
        dt_s: float = 0.025,
        save_prediction_stream: bool = False,
        save_waveform_stream: bool = False,
        save_phase_arrivals: bool = True,
    ) -> None:
        super().__init__(write_interval="batch")
        self.output_dir = Path(output_dir)
        self.extract_phases = extract_phases
        self.sampling_rate = int(1 / dt_s)

        self.save_prediction_stream = save_prediction_stream
        self.save_waveform_stream = save_waveform_stream
        self.save_phase_arrivals = save_phase_arrivals

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction,
        batch_indices,
        batch: Dict,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # we follow the same logic as in batch_inference_step.py
        if "predictions" not in batch:
            return
        inference_predictions = batch["predictions"][0]

        start = UTCDateTime(batch["start"][0])
        end = UTCDateTime(batch["end"][0])
        true_start = UTCDateTime(batch["true_start"][0])
        net = batch["net"][0]
        sta = batch["sta"][0]

        # * save to phase_arrivals.csv
        if self.save_phase_arrivals:
            phase_save_path = (
                self.output_dir / f"phase_arrivals_{trainer.global_rank}.csv"
            )
            if not phase_save_path.is_file():
                with phase_save_path.open("w") as f:
                    f.write("net,sta,win_start,win_end,phase,point,time,amp\n")

            with phase_save_path.open("a") as f:
                for phase in self.extract_phases:
                    for arrival, amp in zip(
                        inference_predictions["arrivals"][phase],
                        inference_predictions["amps"][phase],
                    ):
                        phase_offset = float(f"{arrival/self.sampling_rate:.2f}")
                        arrival_time = true_start + phase_offset
                        if true_start < start:
                            # arrival_pos is measured from the start of the window
                            time_diff = start - true_start
                            assert type(time_diff) == float
                            arrival_pos = int(arrival - time_diff * self.sampling_rate)
                        else:
                            arrival_pos = arrival

                        if start <= arrival_time <= end:
                            f.write(
                                f"{net},{sta},{str(start)},{str(end)},{phase},{arrival_pos},{str(arrival_time)},{amp:.2f}\n"
                            )

        # * save to net.sta.start.end.waveform.sac
        if self.save_waveform_stream:
            fname = (
                self.output_dir / f"{net}.{sta}.{str(start)}.{str(end)}.waveform.mseed"
            )
            stream = Stream()
            for icomponent in range(len(batch["ids"])):
                d = batch["data"][0][icomponent].detach().cpu().numpy()
                trace = Trace(data=d)
                trace.stats.starttime = true_start
                trace.stats.sampling_rate = self.sampling_rate
                network, station, locaton, channel = batch["ids"][icomponent][0].split(
                    "."
                )
                trace.stats.network = network
                trace.stats.station = station
                trace.stats.locaton = locaton
                trace.stats.channel = channel
                trace.trim(start, end)
                stream += trace
            stream.write(str(fname), format="MSEED")

        # * write the prediction result stream
        if self.save_prediction_stream:
            fname = (
                self.output_dir
                / f"{net}.{sta}.{str(start)}.{str(end)}.prediction.mseed"
            )
            stream = Stream()
            for phase in self.extract_phases:
                d = np.array(inference_predictions["possibility"][phase])
                trace = Trace(data=d)
                trace.stats.starttime = true_start
                trace.stats.sampling_rate = self.sampling_rate
                trace.stats.network = net
                trace.stats.station = sta
                trace.stats.locaton = ""
                trace.stats.channel = phase
                trace.trim(start, end)
                stream += trace
            stream.write(str(fname), format="MSEED")
