import torch.nn as nn
import torch
import torch.nn.functional as F


class MRIProjector(nn.Module):
    def __init__(self, output_dims=3):
        super().__init__()
        # A 1x1 convolution to learn the mapping from 1 -> 3 channels
        self.projector = nn.Conv2d(1, output_dims, kernel_size=1)

        # Initialize to distribute signal across channels evenly
        nn.init.constant_(self.projector.weight, 1.0 / 3.0)
        nn.init.zeros_(self.projector.bias)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        # Project and then squash into the VAE's [-1, 1] range
        return torch.tanh(self.projector(x))


class LatentMRIProjector(nn.Module):
    """
    A 1x1 convolution to learn the mapping from MRI latent space to RGB SD1.5 latent space

    Shoule be computationally more efficient than projeting mri to rgb as we have a lower
    spatial resolution (i.e. 64x64 compared to 512x512 spatial size)
    Uses a kernel of size 3 because I like them but should do ablations down the line

    Notes:
        - MRI-VAE results in 2x512x512 -> 4x128x128
        - SD1.5-VAR results in 3x512x512 -> 4x64x64
    """

    def __init__(
        self,
        spatial_in: int,
        spatial_out: int,
        in_channels: int = 4,
        out_channels: int = 4,
    ):
        super().__init__()
        s = spatial_in // spatial_out
        self.projector = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=s
        )

    def forward(self, x):
        return self.projector(x)

class InverseProjectorLearned(nn.Module):
    def __init__(
        self, spatial_in, spatial_out, in_channels=4, out_channels=4, hidden=64
    ):
        super().__init__()
        s = spatial_out // spatial_in
        self.upsample_scale = s
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x: (B, C_sd, H_sd, W_sd)
        x_up = F.interpolate(
            x, scale_factor=self.upsample_scale, mode="bilinear", align_corners=False
        )
        return self.conv(x_up)  # (B, C_mri, H_mri, W_mri)


def robust_mri_scale(tensor: torch.Tensor, pmin=0.5, pmax=99.5):
    """
    Normalizes a tensor based on percentiles to handle MRI intensity outliers.
    """
    b, c, h, w = tensor.shape
    flat = tensor.float().view(b, -1)

    # Calculate quantiles per batch item
    mi = torch.quantile(flat, pmin / 100.0, dim=1, keepdim=True).view(b, 1, 1, 1)
    ma = torch.quantile(flat, pmax / 100.0, dim=1, keepdim=True).view(b, 1, 1, 1)

    normalized = (tensor - mi) / (ma - mi + 1e-8)
    return normalized.clamp(0.0, 1.0)
