import torch.nn as nn
import torch


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


def robust_mri_scale(tensor, pmin=0.5, pmax=99.5):
    """
    Normalizes a tensor based on percentiles to handle MRI intensity outliers.
    """
    b, c, h, w = tensor.shape
    flat = tensor.view(b, -1)

    # Calculate quantiles per batch item
    mi = torch.quantile(flat, pmin / 100.0, dim=1, keepdim=True).view(b, 1, 1, 1)
    ma = torch.quantile(flat, pmax / 100.0, dim=1, keepdim=True).view(b, 1, 1, 1)

    normalized = (tensor - mi) / (ma - mi + 1e-8)
    return normalized.clamp(0.0, 1.0)
