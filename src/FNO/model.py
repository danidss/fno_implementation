import torch
import torch.nn as nn

from functools import partial


class FourierLayer(nn.Module):
    """
    """

    def __init__(
        self, dim: int, in_channels: int, out_channels: int, modes: int
    ) -> None:
        super().__init__()

        self.modes_slices = (slice(0, modes),) * dim

        self.skip_weight = nn.Linear(in_channels, out_channels)
        self.fourier_weight = nn.Linear(
            in_channels, out_channels, dtype=torch.complex64
        )
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

    def forward(self, vt: torch.Tensor) -> torch.Tensor:
        # vt: (batch, *spatial, in_channels)
        skip_connection = self.skip_weight(vt)  # (batch, *spatial, out_channels)

        fouriered = self.fft(vt)  # (batch, *spatial[-1], spatial[-1]//2+1, in_channels)
        lowest_modes = fouriered[:, *self.modes_slices, :]  # (b, *modes, in_channels)
        mixed_channels = self.fourier_weight(lowest_modes)  # (b, *modes, out_channels)

        # Fill removed modes with zeros
        full_spectrum = torch.zeros(
            fouriered.shape[:-1] + (mixed_channels.shape[-1],),
            device=vt.device,
            dtype=torch.complex64,
        )
        full_spectrum[:, *self.modes_slices, :] = (
            mixed_channels  # (b, *spatial, out_channels)
        )

        inversed = self.ifft(full_spectrum)  # (batch, *spatial, out_channels)

        return self.gelu(inversed + skip_connection)



class FNO(nn.Module):
    """
    """

    def __init__(
        self,
        dim: int,
        modes: int,
        layer_shapes: list[tuple[int, int]],
        in_channels: int = 1,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        self.lift = nn.Linear(in_channels, layer_shapes[0][0])
        self.fourier_layers = nn.ModuleList(
            [
                FourierLayer(dim=dim, in_channels=in_c, out_channels=out_c, modes=modes)
                for in_c, out_c in layer_shapes
            ]
        )
        self.project = nn.Linear(layer_shapes[-1][1], out_channels)

    def forward(self, vt: torch.Tensor) -> torch.Tensor:
        # vt: (batch, *spatial, in_channels)
        lifted = self.lift(vt)  # (batch, *spatial, layer_shapes[0][0])
        fouriered = lifted
        for layer in self.fourier_layers:
            fouriered = layer(fouriered)  # (batch, *spatial, layer_shapes[-1][1])
        projected = self.project(fouriered)  # (batch, *spatial, out_channels)
        return projected


class LpLoss(object):
    """
    Relative L2 norm loss, widely used for PDE surrogate models like FNO.
    Computes: ||x - y||_2 / ||y||_2
    """

    def __init__(
        self, d: int = 2, p: int = 2, size_average: bool = True, reduction: bool = True
    ) -> None:
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: prediction, y: ground truth
        num_examples = x.size(0)

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rel(x, y)
