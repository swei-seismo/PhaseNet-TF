import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabV3Plus(smp.DeepLabV3Plus):
    def __init__(
        self,
        out_channels: int = 4,
        image_height: int = 64,
    ):
        # fix the parameters, as we are using the predefined model
        super().__init__(
            encoder_name="resnet34",
            encoder_depth=5,
            encoder_weights=None,
            encoder_output_stride=16,
            decoder_channels=256,
            decoder_atrous_rates=(12, 24, 36),
            in_channels=6,  # three component real and neg spectrograms
            classes=out_channels,
            upsampling=4,
        )

        self.fc1 = nn.Linear(image_height, image_height // 2)
        self.fc2 = nn.Linear(image_height // 2, 1)

    def forward(self, x: torch.Tensor):
        result = {}
        x = F.relu(super().forward(x))  # type: ignore

        # transpose to have the last axis as freq, as Linear required
        x = x.transpose(-1, -2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(-1)

        # * prepare result
        result["predict"] = x
        return result
