""" 
tsindex.py: The pytorch dataset for the TSIndex database in obspy. It's mainly used to do model batch inference.
"""
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from obspy import Stream, UTCDateTime
from obspy.clients.filesystem.tsindex import Client
from torch.utils.data import Dataset


class TSIndexDataset(Dataset):
    def __init__(
        self,
        inference_requirement_path: Path,
        inference_output_dir: Path,
        tsindex_database_path: Path,
        datapath_name_replace: Optional[list] = None,
        continuous_window_time_in_sec: float = 3600,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.inference_output_dir = inference_output_dir
        self.tsindex_database_path = tsindex_database_path
        self.datapath_name_replace = datapath_name_replace
        self.transform = transform
        self.stream_to_tensor_transform = StreamToTensorTransform()

        # handle the inference requirement file
        requirement = pd.read_csv(inference_requirement_path, comment="#")
        requirement.sort_values(
            by=["network", "station", "start_time", "end_time"],
            ascending=[True, True, True, True],
        )

        self.all_inference_windows = []
        for _, row in requirement.iterrows():
            time_diff = UTCDateTime(row.end_time) - UTCDateTime(row.start_time)
            assert type(time_diff) == float
            steps = int(np.ceil(time_diff / continuous_window_time_in_sec))
            for istep in range(steps):
                start = (
                    UTCDateTime(row.start_time) + istep * continuous_window_time_in_sec
                )
                end = (
                    UTCDateTime(row.start_time)
                    + (istep + 1) * continuous_window_time_in_sec
                )
                self.all_inference_windows.append(
                    (row.network, row.station, start, end)
                )

    def __len__(self):
        return len(self.all_inference_windows)

    def __getitem__(self, idx: int) -> dict:
        # load client here but not init to avoid multiprocessing issues
        client = Client(
            database=str(self.tsindex_database_path),
            datapath_replace=tuple(self.datapath_name_replace)
            if self.datapath_name_replace
            else None,
        )

        net, sta, start, end = self.all_inference_windows[idx]
        try:
            st = client.get_waveforms(net, sta, "*", "*", start, end)
        except Exception as e:
            error_log_file = self.inference_output_dir / "error.log"
            if not error_log_file.exists():
                error_log_file.touch()
                # write the header
                with open(error_log_file, "w") as f:
                    f.write("net,sta,start,end,stream_len,error_info\n")
            with open(error_log_file, "a") as f:
                f.write(f"{net},{sta},{start},{end},,{str(e)}\n")
            return {}

        if len(st) != 3:
            st.merge()
            if len(st) != 3:
                error_log_file = self.inference_output_dir / "error.log"
                if not error_log_file.exists():
                    error_log_file.touch()
                    # write the header
                    with open(error_log_file, "w") as f:
                        f.write("net,sta,start,end,stream_len,error_info\n")
                with open(error_log_file, "a") as f:
                    f.write(f"{net},{sta},{start},{end},{len(st)},\n")
                return {}

        # convert to tensor
        res = self.stream_to_tensor_transform(st)
        if res is None:
            raise Exception(
                f"Cannot find data for {net}.{sta} from {start} to {end} for RTZ or ENZ or 12Z components"
            )
        # now res has keys: ids, data, true_start
        res.update(
            {
                "net": net,
                "sta": sta,
                "start": str(start),
                "end": str(end),
            }
        )

        # use transform if provided to process res["data"]
        if self.transform:
            res["data"] = self.transform(res["data"])

        return res


class StreamToTensorTransform:
    def __init__(self) -> None:
        pass

    def __call__(self, stream: Stream) -> Optional[dict]:
        res = {
            "ids": [],
            "true_start": "",
            "data": None,
        }

        components = ["R", "T", "Z"]
        components_replace = ["E", "N", "Z"]
        components_replace2 = ["1", "2", "Z"]
        traces = []
        for i in range(3):
            trace = stream.select(component=components[i])
            if len(trace) == 0:
                trace = stream.select(component=components_replace[i])
            if len(trace) == 0:
                trace = stream.select(component=components_replace2[i])
            if len(trace) == 0:
                return None
            trace = trace[0]
            traces.append(trace)
            res["ids"].append(trace.id)
        # min_length = min(len(item) for item in traces)
        # there might be cases that the starting time is not aligned, so here we trim everything first
        true_start = max(item.stats.starttime for item in traces)
        true_end = min(item.stats.endtime for item in traces)
        for i in range(len(traces)):
            traces[i].trim(true_start, true_end)
        min_length = min(len(item) for item in traces)
        # curent min_time and max_time will be different with start and end (as having trim)
        # so we need to record to cut them later
        res["true_start"] = str(true_start)

        # convert to torch tensor with shape (3, min_length)
        data = torch.zeros((3, min_length))
        for i in range(3):
            data[i, :] = torch.from_numpy(traces[i].data[:min_length])
        res["data"] = data

        return res
