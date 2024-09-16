import torch
import torchvision.transforms.functional as F
from nnAudio import features
from torch import nn


class GenSgram(nn.Module):
    """
    Generate spectrogram from waveform
    """

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 1,
        freqmin: float = 0,
        freqmax: float = 10,
        dt_s: float = 0.025,
        height: int = 64,
        width: int = 4800,
        max_clamp: int = 3000,
    ):
        """
        Args:
            n_fft: use n_fft points to do fft in one sliding window, default is 256
            hop_length: the number of samples between the starts of consecutive frames, default is 1
            freqmin: the minimum frequency to be shown in the spectrogram, default is 0
            freqmax: the maximum frequency to be shown in the spectrogram, default is 10
            dt_s: the time interval between two samples, default is 0.025
            height: the height of the spectrogram (number of points in frequency axis), default is 64
            width: the width of the spectrogram (number of points in time axis), default is 4800
            max_clamp: the maximum value to be shown in the spectrogram, default is 3000
        """
        super().__init__()
        # since Spectrogram has no params, we don't need to set it as no_grad
        self.spec_layer = features.STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            output_format="Complex",
            window="hann",
            verbose=False,
        )
        self.n_fft = n_fft
        self.freqmin = freqmin
        self.freqmax = freqmax
        self.sampling_rate = int(1 / dt_s)
        self.height = height
        self.width = width
        self.max_clamp = max_clamp

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: a tensor of shape (batch, channel, time)
        Returns:
            sgram: a tensor of shape (batch, channel, freq, time)
        """
        # sgram: torch.Tensor = super().__call__(waveform)
        sgrams = [self.spec_layer(waveform[:, i, :]) for i in range(waveform.shape[1])]
        sgram = torch.stack(sgrams, dim=1)
        # sgram: torch.Tensor = self.func(data)
        # we should cut the frequency between freqmin to freqmax
        # the time bin length is nt//hop_length+1, we assume the mod is nt%hop_length==0
        df = (self.sampling_rate / 2) / (self.n_fft // 2)
        freqmin_pos = round(self.freqmin / df)
        freqmax_pos = round(self.freqmax / df)
        sgram = sgram[..., freqmin_pos : freqmax_pos + 1, :-1, :]

        # first 3 channel as real, last 3 as imag
        real = sgram[..., 0]
        imag = sgram[..., 1]
        p = real**2 + imag**2 + 0.001
        ratio = torch.clamp_max(p, self.max_clamp) / p
        # ! note, this can only be done with batch dimension
        sgram = torch.cat([real * ratio, imag * ratio], dim=1)
        # we don't need anti-aliasing here, as we are always downsampling and it's spectrogram
        sgram = F.resize(sgram, [self.height, self.width], antialias=None)

        return sgram
