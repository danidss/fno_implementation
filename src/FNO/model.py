import torch
import torch.nn as nn

from functools import partial


class FourierLayer(nn.Module):
    """
    """
    def __init__(self, dim: int, in_channels: int, out_channels: int, modes: int):
        super().__init__()

        self.modes_slices = (slice(0, modes),) * dim

        self.skip_weight = nn.Linear(in_channels, out_channels)
        self.fourier_weight = nn.Linear(in_channels, out_channels, dtype=torch.complex64)
        self.gelu = nn.GELU()

        if dim == 1:
            self.fft = partial(torch.fft.rfft, dim=1)
            self.ifft = partial(torch.fft.irfft, dim=1)
        elif dim == 2:
            self.fft = partial(torch.fft.rfft2, dim=(1, 2))
            self.ifft = partial(torch.fft.irfft2, dim=(1, 2))
        else:
            self.fft = partial(torch.fft.rfftn, dim=tuple(range(1, dim+1)))
            self.ifft = partial(torch.fft.irfftn, dim=tuple(range(1, dim+1)))

    def forward(self, vt: torch.Tensor):
        # vt: (batch, *spatial, in_channels)
        skip_connection = self.skip_weight(vt)  # (batch, *spatial, out_channels)

        fouriered = self.fft(vt)  # (batch, *spatial[-1], spatial[-1]//2+1, in_channels)
        lowest_modes = fouriered[:, *self.modes_slices, :]  # (batch, *modes, in_channels)
        mixed_channels = self.fourier_weight(lowest_modes)  # (b, *modes, out_channels)

        # Fill removed modes with zeros
        full_spectrum = torch.zeros(fouriered.shape[:-1] + (mixed_channels.shape[-1],), device=vt.device, dtype=torch.complex64)
        full_spectrum[:, *self.modes_slices, :] = mixed_channels  # (b, *spatial, out_channels)

        inversed = self.ifft(full_spectrum)  # (batch, *spatial, out_channels)

        return self.gelu(inversed + skip_connection)



class FNO(nn.Module):
    """
    """
