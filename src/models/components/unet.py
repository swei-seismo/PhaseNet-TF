"""
unet.py

An Unet modificaion for the spectrogram based seismic phase picking
"""
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn

# * ============== define blocks ============== * #


class RepeatingConv(nn.Module):
    """CNN(i->f) => BN => ReLU => (CNN(f->f) => BN => ReLU)*r"""

    def __init__(self, i: int, f: int, r: int, ksize: Tuple[int, int]) -> None:
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=i,
                out_channels=f,
                kernel_size=ksize,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=f),
            nn.ReLU(inplace=True),
        )
        # repeating convs
        repeating_convs = []
        for _ in range(r):
            repeating_convs.extend(
                [
                    nn.Conv2d(
                        in_channels=f,
                        out_channels=f,
                        kernel_size=ksize,
                        padding="same",
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_features=f),
                    nn.ReLU(inplace=True),
                ]
            )
        self.second_conv = nn.Sequential(*repeating_convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second_conv(self.first_conv(x))


class Down(nn.Module):
    """RepeatingConv => (MaxPool2d , Sequential(h->1))"""

    def __init__(
        self,
        i: int,
        f: int,
        r: int,
        h: int,
        ksize: Tuple[int, int],
        calculate_skip_for_encoder: bool,
    ) -> None:
        super().__init__()
        # repeating conv
        self.repeating_conv = RepeatingConv(i, f, r, ksize)
        # fc and maxpooling
        if calculate_skip_for_encoder:
            self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            self.fc = nn.Sequential(nn.Linear(h, 1), nn.ReLU(inplace=True))
        else:
            self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
            self.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.repeating_conv(x)
        return self.pool(x), self.fc(x)


class BottleNeck(nn.Module):
    """RepeatingConv => Sequential(h->1)"""

    def __init__(
        self,
        i: int,
        f: int,
        r: int,
        h: int,
        ksize: Tuple[int, int],
        calculate_skip_for_encoder,
    ) -> None:
        super().__init__()
        # repeating conv
        self.repeating_conv = RepeatingConv(i, f, r, ksize)
        # fc
        if calculate_skip_for_encoder:
            self.fc = nn.Sequential(nn.Linear(h, 1), nn.ReLU(inplace=True))
        else:
            self.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.repeating_conv(x)
        return self.fc(x)


class Up(nn.Module):
    def __init__(self, i: int, f: int, r: int, ksize: Tuple[int, int]) -> None:
        """(ConvTranspose2d => ReLU + skip) => RepeatingConv"""
        super().__init__()
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(i, f, kernel_size=(2, 1), stride=(2, 1)),
            nn.ReLU(inplace=True),
        )
        self.decoder = RepeatingConv(2 * f, f, r, ksize)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        x = torch.cat((x, skip), dim=1)
        return self.decoder(x)


# * ============== define U-Net model ============== * #


class UNet(nn.Module):
    def __init__(
        self,
        features: int = 32,
        in_cha: int = 6,
        out_cha: int = 4,
        first_layer_repeating_cnn: int = 3,
        n_freq: int = 64,
        ksize_down: Tuple[int, int] = (5, 5),
        ksize_up: Tuple[int, int] = (5, 5),
        encoder_decoder_depth: int = 5,
        calculate_skip_for_encoder: bool = True,
    ):
        super().__init__()
        # * encoders
        self.encoder_dict = OrderedDict()
        self.encoder_dict["enc1"] = Down(
            in_cha,
            features,
            first_layer_repeating_cnn,
            n_freq,
            ksize_down,
            calculate_skip_for_encoder,
        )

        # eg:
        # self.enc2 = Down(features*1, features*2, 1, n_freq//2, ksize_down)
        # self.enc3 = Down(features*2, features*4, 1, n_freq//4, ksize_down)
        for idx in range(2, encoder_decoder_depth + 1):
            self.encoder_dict[f"enc{idx}"] = Down(
                features * (2 ** (idx - 2)),
                features * (2 ** (idx - 1)),
                1,
                n_freq // (2 ** (idx - 1)),
                ksize_down,
                calculate_skip_for_encoder,
            )
        self.encoder = nn.ModuleDict(self.encoder_dict)

        # * bottleneck
        self.bottleneck = BottleNeck(
            features * (2 ** (encoder_decoder_depth - 1)),
            features * (2**encoder_decoder_depth),
            1,
            n_freq // (2**encoder_decoder_depth),
            ksize_down,
            calculate_skip_for_encoder,
        )

        # * decoders
        # eg:
        # self.dec3 = Up(features*8, features*4, 1, ksize_up)
        # self.dec2 = Up(features*4, features*2, 1, ksize_up)
        # self.dec1 = Up(features*2, features*1, 1, ksize_up)
        self.decoder_dict = OrderedDict()
        for idx in range(encoder_decoder_depth, 0, -1):
            self.decoder_dict[f"dec{idx}"] = Up(
                features * (2**idx), features * (2 ** (idx - 1)), 1, ksize_up
            )
        self.decoder = nn.ModuleDict(self.decoder_dict)

        # * out
        self.conv_out = nn.Conv2d(
            in_channels=features, out_channels=out_cha, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # the input is a batch of spectrograms, expected nf=32k, where k is positive
        # bt, ch, nf, nt = x.shape
        # the sgram need to be rotated to nt,nf we can map frequency to size 1
        x = x.transpose(-1, -2)

        # * encode
        # eg:
        # x, skip1 = self.enc1(x)
        # x, skip2 = self.enc2(x)
        # x, skip3 = self.enc3(x)
        skips = []
        for key in self.encoder:
            x, skip = self.encoder[key](x)
            skips.append(skip)

        # * bottleneck
        x = self.bottleneck(x)

        # * decode
        # eg:
        # x = self.dec3(x, skip3)
        # x = self.dec2(x, skip2)
        # x = self.dec1(x, skip1)
        skips = skips[::-1]
        for skip_id, key in enumerate(self.decoder):
            x = self.decoder[key](x, skips[skip_id])

        # * output
        x = self.conv_out(x)
        # change out from nt,1 to nt
        # so here the out will be bt, ch, nt
        # we design nt to be npts, so no need to upsampling
        x = x.squeeze(-1)

        # * prepare result
        result = {}
        result["predict"] = x
        return result
