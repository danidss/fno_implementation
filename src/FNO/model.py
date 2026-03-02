import torch
import torch.nn as nn
import torch.fft as fft


# TODO check better contiguous allocation for the matrix
# multiplication instead of one einsum for each corner
# TODO check if we can compute only the needed modes instead of all the fft
# TODO check if we can optimize the padding / zeros creation


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes = modes

        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale
            * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape of x: (batch_size, in_channels, n)
        batchsize = x.shape[0]

        x_ft: torch.Tensor = fft.rfft(x)

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x_ft.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )
        # 'b' = batch, 'i' = input channel, 'o' = output channel, 'x' = frequency mode
        out_ft[:, :, : self.modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, : self.modes], self.weights
        )

        return fft.irfft(out_ft, n=x.shape[-1])


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes1 = modes1
        self.modes2 = modes2

        # We initialize the weights from an uniform u_r+u_c*i
        scale = 1 / (in_channels * out_channels)

        # In 2D using Real FFT, we have 2 corners of low frequencies:
        # 1. Pos x, Pos y
        # 2. Neg x, Pos y (because of the real FFT, y is only positive)
        self.weights1 = nn.Parameter(
            scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # 'b' = batch, 'i' = input channel, 'o' = output channel, 'x', 'y' = frequency modes
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape of x: (batch_size, in_channels, n, m)
        batchsize = x.shape[0]

        x_ft: torch.Tensor = fft.rfft2(x)

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.shape[-2],
            x_ft.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )
        # Corner 1: Top-Left (Positive frequencies for dim 1, Positive for dim 2)
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        # Corner 2: Bottom-Left (Negative frequencies for dim 1, Positive for dim 2)
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        return fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))


class SpectralConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        # We initialize the weights from an uniform u_r+u_c*i
        scale = 1 / (in_channels * out_channels)

        # In 3D using Real FFT, we have 4 corners of low frequencies:
        # 1. Pos x, Pos y (Pos z is given by RFFT)
        # 2. Neg x, Pos y
        # 3. Pos x, Neg y
        # 4. Neg x, Neg y
        self.weights1 = nn.Parameter(
            scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights3 = nn.Parameter(
            scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights4 = nn.Parameter(
            scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )

    def compl_mul3d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # 'b' = batch, 'i' = input channel, 'o' = output channel, 'x', 'y', 'z' = frequency modes
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape of x: (batch_size, in_channels, n, m, p)
        batchsize = x.shape[0]

        x_ft = fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.shape[-3],
            x.shape[-2],
            x_ft.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )

        # Corner 1: Positive x, Positive y, Positive z
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        # Corner 2: Negative x, Positive y, Positive z
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        # Corner 3: Positive x, Negative y, Positive z
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        # Corner 4: Negative x, Negative y, Positive z
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )

        return fft.irfftn(out_ft, s=(x.shape[-3], x.shape[-2], x.shape[-1]))


SPECTRAL_CONV = {
    1: SpectralConv1d,
    2: SpectralConv2d,
    3: SpectralConv3d,
}

SKIP_CONV = {
    1: nn.Conv1d,
    2: nn.Conv2d,
    3: nn.Conv3d,
}

BATCH_NORM = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d,
}


class FourierLayer(nn.Module):
    """ """

    def __init__(
        self, dim: int, in_channels: int, out_channels: int, modes: int
    ) -> None:
        super().__init__()

        assert dim in SPECTRAL_CONV.keys(), (
            f"Dimension must be one of {list(SPECTRAL_CONV.keys())}"
        )

        self.skip_weight = SKIP_CONV[dim](in_channels, out_channels, kernel_size=1)
        self.conv = SPECTRAL_CONV[dim](in_channels, out_channels, *((modes,) * dim))
        self.bn = BATCH_NORM[dim](out_channels)
        self.relu = nn.ReLU()

    def forward(self, vt: torch.Tensor) -> torch.Tensor:
        # vt: (batch, in_channels, *spatial)
        return self.relu(self.bn(self.conv(vt) + self.skip_weight(vt)))


class FNO(nn.Module):
    """
    """

    def __init__(
        self,
        dim: int,
        modes: int,
        layer_shapes: tuple[int],
        in_channels: int = 1,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        self.lift = SKIP_CONV[dim](in_channels, layer_shapes[0], kernel_size=1)
        self.fourier_layers = nn.ModuleList(
            [
                FourierLayer(dim=dim, in_channels=in_c, out_channels=out_c, modes=modes)
                for in_c, out_c in zip(layer_shapes[:-1], layer_shapes[1:])
            ]
        )
        self.project = SKIP_CONV[dim](layer_shapes[-1], out_channels, kernel_size=1)

    def forward(self, vt: torch.Tensor) -> torch.Tensor:
        # vt: (batch, in_channels, *spatial)
        lifted = self.lift(vt)  # (batch, layer_shapes[0], *spatial)
        fouriered = lifted
        for layer in self.fourier_layers:
            fouriered = layer(fouriered)  # (batch, layer_shapes[i], *spatial)
        projected = self.project(fouriered)  # (batch, out_channels, *spatial)
        return projected


class LpLoss:
    """
    Relative L2 norm loss, widely used for PDE surrogate models like FNO.
    Computes: ||x - y||_2 / ||y||_2
    """

    def __init__(
        self, d: int = 2, p: int = 2, size_average: bool = True, reduction: bool = True
    ) -> None:
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: prediction, y: ground truth
        dims = tuple(range(1, x.ndim))

        diff_norms = torch.linalg.vector_norm(x - y, ord=self.p, dim=dims)
        y_norms = torch.linalg.vector_norm(y, ord=self.p, dim=dims)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rel(x, y)
