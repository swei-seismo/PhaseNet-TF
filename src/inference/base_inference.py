""" 
base_inference.py: inference functions and classes that are shared between single and batch inference.
"""
import torch


class PadZeroTransform:
    def __init__(self, window_length_in_npts: int, hop_length_in_npts: int):
        """
        Pad input tensor with zeros.

        Args:
            window_length_in_npts (int): Window length in number of points, should be divided by hop_length_in_npts.
            hop_length_in_npts (int): Hop (Step) length in number of points.
        """
        self.window_length_in_npts = window_length_in_npts
        self.hop_length_in_npts = hop_length_in_npts

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pad input tensor with zeros.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Padded input tensor.
        """
        # x has shape of (3, NPTS), where NPTS is the number of points
        # pad NPTS to be at least window_length_in_npts, and then pad to be divisible by hop_length_in_npts
        _, npts = x.shape
        if npts < self.window_length_in_npts:
            x = torch.cat(
                (
                    x,
                    torch.zeros(
                        (3, self.window_length_in_npts - npts), device=x.device
                    ),
                ),
                dim=1,
            )
        if npts % self.hop_length_in_npts != 0:
            x = torch.cat(
                (
                    x,
                    torch.zeros(
                        (3, self.hop_length_in_npts - npts % self.hop_length_in_npts),
                        device=x.device,
                    ),
                ),
                dim=1,
            )
        return x


class SlidingWindowNormalizeTransform:
    def __init__(self, window_length_in_npts: int, hop_length_in_npts: int):
        """
        Normalize input tensor with sliding window.

        Args:
            window_length_in_npts (int): Window length in number of points, should be divided by hop_length_in_npts.
            hop_length_in_npts (int): Hop (Step) length in number of points.
        """
        self.window_length_in_npts = window_length_in_npts
        self.hop_length_in_npts = hop_length_in_npts

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input tensor with sliding window.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized input tensor.
        """
        # x has shape of (3, NPTS), where NPTS is the number of points
        _, npts = x.shape

        # firstly calculate the mean and std of each window
        means, stds = [], []
        for i in range(
            0, npts - self.window_length_in_npts + 1, self.hop_length_in_npts
        ):
            window = x[:, i : i + self.window_length_in_npts]
            means.append(window.mean(dim=1))
            stds.append(window.std(dim=1))

        # then interp the mean and std of the whole signal
        means = torch.stack(means, dim=1)
        stds = torch.stack(stds, dim=1)
        means = torch.nn.functional.interpolate(
            means.unsqueeze(0), size=npts, mode="linear", align_corners=False
        ).squeeze(0)
        stds = torch.nn.functional.interpolate(
            stds.unsqueeze(0), size=npts, mode="linear", align_corners=False
        ).squeeze(0)

        # finally normalize the input tensor, be careful of the zero std
        x = (x - means) / (stds + 1e-8)
        return x


class BatchNormalizeTransform:
    def __init__(self) -> None:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        for i in range(batch_size):
            data = x[i]
            mean_vals = torch.mean(data, dim=1, keepdim=True)
            data = data - mean_vals
            max_std_val = torch.max(torch.std(data, dim=1))
            if max_std_val == 0:
                max_std_val = torch.ones(1, device=x.device)
            data = data / max_std_val
            x[i] = data
        return x


class WaveformToBatchTransform:
    def __init__(self, window_length_in_npts: int, hop_length_in_npts: int):
        """
        Transform input tensor to be suitable for batch inference.

        Args:
            window_length_in_npts (int): Window length in number of points, should be divided by hop_length_in_npts.
            hop_length_in_npts (int): Hop (Step)
        """
        # convert the input tensor from (3, NPTS) to (number_of_windows, 3, window_length_in_npts)
        self.window_length_in_npts = window_length_in_npts
        self.hop_length_in_npts = hop_length_in_npts

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform input tensor to be suitable for batch inference.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed input tensor.
        """
        # x has shape of (3, NPTS), where NPTS is the number of points
        _, npts = x.shape

        # suppose we have already padded the input tensor with zeros
        # convert the input tensor from (3, NPTS) to (number_of_windows, 3, window_length_in_npts)
        res = []
        for i in range(
            0, npts - self.window_length_in_npts + 1, self.hop_length_in_npts
        ):
            res.append(x[:, i : i + self.window_length_in_npts])
        x = torch.stack(res, dim=0)
        return x


class BatchToWaveformTransform:
    def __init__(self, window_length_in_npts: int, hop_length_in_npts: int):
        """
        Transform batched output tensor back to waveform, inverse operation of WaveformToBatchTransform.

        Args:
            window_length_in_npts (int): Window length in number of points, should be divided by hop_length_in_npts.
            hop_length_in_npts (int): Hop (Step)
        """
        self.window_length_in_npts = window_length_in_npts
        self.hop_length_in_npts = hop_length_in_npts

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform batched output tensor back to waveform, inverse operation of WaveformToBatchTransform.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed input tensor.
        """
        # x has shape of (number_of_windows, channel, window_length_in_npts)
        # note here channel should number of phases + 1
        number_of_windows, channel, _ = x.shape

        # there will be overlap between windows, so we need to maximize them
        npts = (
            number_of_windows * self.hop_length_in_npts
            + self.window_length_in_npts
            - self.hop_length_in_npts
        )
        res = torch.zeros((channel, npts), device=x.device)
        for i in range(number_of_windows):
            # for the overlap part, we take the max of the two/three/... windows
            res[
                :,
                i * self.hop_length_in_npts : i * self.hop_length_in_npts
                + self.window_length_in_npts,
            ] = torch.max(
                res[
                    :,
                    i * self.hop_length_in_npts : i * self.hop_length_in_npts
                    + self.window_length_in_npts,
                ],
                x[i, :, :],
            )
        return res
