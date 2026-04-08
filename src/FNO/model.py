import torch
import torch.nn as nn
import torch.fft as fft
from typing import Type

from src.utils import build_activation, build_normalization


# TODO check better contiguous allocation for the matrix
# multiplication instead of one einsum for each corner
# TODO check if we can optimize the padding / zeros creation


class SpectralConv(nn.Module):
    """Base class and factory for dimension-specific spectral convolutions."""

    @classmethod
    def create(
        cls,
        in_channels: int,
        out_channels: int,
        modes: tuple[int, ...],
    ) -> "SpectralConv":
        """Creates the dimension-specific spectral convolution implementation.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            modes: Number of retained Fourier modes per spatial dimension.

        Returns:
            An instance of `SpectralConv1d`, `SpectralConv2d`, or `SpectralConv3d`.
        """
        conv_cls: Type[SpectralConv]
        dim = len(modes)
        if dim == 1:
            conv_cls = SpectralConv1d
        elif dim == 2:
            conv_cls = SpectralConv2d
        elif dim == 3:
            conv_cls = SpectralConv3d
        else:
            raise ValueError(
                f"Unsupported dimension inferred from modes={modes}. Expected 1D, 2D or 3D."
            )

        return conv_cls(in_channels, out_channels, modes)


class PointwiseMLP(nn.Module):
    """MLP-like block applied pointwise over arbitrary spatial dimensions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dims: tuple[int, ...] = (),
        nonlinearity: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        """Initializes the pointwise MLP.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            hidden_dims: Hidden layer channel sizes used by intermediate pointwise linear layers.
            nonlinearity: Activation name used between hidden layers.
                Supported values: "relu", "gelu", "silu", "tanh", "elu", "leaky_relu".
            dropout: Dropout rate applied after hidden activations.
        """
        super().__init__()

        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1). Received {dropout}.")

        channels = (in_channels,) + hidden_dims + (out_channels,)

        layers: list[nn.Module] = []
        for idx, (in_c, out_c) in enumerate(zip(channels[:-1], channels[1:])):
            layers.append(nn.Linear(in_c, out_c))

            if not idx == len(channels) - 2:  # No activation or dropout on last layer.
                layers.append(build_activation(nonlinearity))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the pointwise MLP over flattened spatial tokens.

        Args:
            x: Input tensor of shape `(batch, channels, *spatial)`.

        Returns:
            Output tensor of shape `(batch, out_channels, *spatial)`.
        """
        # x: (batch, channels, *spatial). Flatten spatial axes into tokens.
        _, _, *spatial_shape = x.shape
        tokens = x.flatten(start_dim=2).transpose(1, 2)  # (batch, n_tokens, channels)
        tokens = self.network(tokens)
        return tokens.transpose(1, 2).unflatten(2, spatial_shape)


class SpectralConv1d(SpectralConv):
    """1D spectral convolution layer for Fourier Neural Operators."""

    def __init__(self, in_channels: int, out_channels: int, modes: tuple[int, ...]):
        """
        Initializes the 1D Spectral Convolution layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            modes: Number of low-frequency modes to retain.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes = modes[0]

        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale
            * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 1D Spectral Convolution.

        Args:
            x: Input tensor of shape (batch_size, in_channels, n).

        Returns:
            Filtered output tensor of shape (batch_size, out_channels, n).
        """
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


class SpectralConv2d(SpectralConv):
    """2D spectral convolution layer for Fourier Neural Operators."""

    def __init__(self, in_channels: int, out_channels: int, modes: tuple[int, ...]):
        """
        Initializes the 2D Spectral Convolution layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            modes: Number of modes to retain in each spatial dimension.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes1, self.modes2 = modes

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
        """Performs complex multiplication in Fourier space for 2D tensors.

        Args:
            input: Complex input tensor in Fourier space.
            weights: Complex learnable spectral weights.

        Returns:
            Complex output tensor after channel mixing.
        """
        # 'b' = batch, 'i' = input channel, 'o' = output channel, 'x', 'y' = frequency modes
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 2D Spectral Convolution.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Filtered output tensor of the same spatial dimensions.
        """
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


class SpectralConv3d(SpectralConv):
    """3D spectral convolution layer for Fourier Neural Operators."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: tuple[int, ...],
    ):
        """
        Initializes the 3D Spectral Convolution layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            modes: Number of modes to retain in each spatial dimension.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes1, self.modes2, self.modes3 = modes

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
        """Performs complex multiplication in Fourier space for 3D tensors.

        Args:
            input: Complex input tensor in Fourier space.
            weights: Complex learnable spectral weights.

        Returns:
            Complex output tensor after channel mixing.
        """
        # 'b' = batch, 'i' = input channel, 'o' = output channel, 'x', 'y', 'z' = frequency modes
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 3D Spectral Convolution.

        Args:
            x: Input tensor of shape (batch, channels, d, h, w).

        Returns:
            Filtered output tensor.
        """
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


class FourierLayer(nn.Module):
    """Single Fourier layer with spectral branch, pointwise skip, normalization, and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: tuple[int, ...],
        norm_class: str | None = None,
        skip_hidden_dims: tuple[int, ...] = (),
        nonlinearity: str = "relu",
        pointwise_dropout: float = 0.0,
    ) -> None:
        """Initializes the Fourier layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            modes: Number of retained Fourier modes per spatial dimension.
            norm_class: Normalization name applied after spectral + skip sum.
                Supported values: "none", "batch", "instance", "layer".
            skip_hidden_dims: Hidden channel sizes for the pointwise skip MLP.
            nonlinearity: Activation name applied after normalization.
                Supported values: "relu", "gelu", "silu", "tanh", "elu", "leaky_relu".
            pointwise_dropout: Dropout rate used inside pointwise MLP blocks.
        """
        super().__init__()

        dim = len(modes)

        assert dim in (1, 2, 3), "Only 1D, 2D and 3D FNO are supported"

        self.skip_weight = PointwiseMLP(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dims=skip_hidden_dims,
            nonlinearity=nonlinearity,
            dropout=pointwise_dropout,
        )
        self.spectral_conv = SpectralConv.create(in_channels, out_channels, modes)
        self.bn = build_normalization(norm_class, out_channels, dim)
        self.nonlinearity = build_activation(nonlinearity)

    def forward(self, vt: torch.Tensor) -> torch.Tensor:
        """Applies spectral convolution, skip connection, optional normalization, and activation.

        Args:
            vt: Input tensor of shape `(batch, in_channels, *spatial)`.

        Returns:
            Output tensor of shape `(batch, out_channels, *spatial)`.
        """
        # vt: (batch, in_channels, *spatial)
        x = self.spectral_conv(vt) + self.skip_weight(vt)
        if self.bn is not None:
            x = self.bn(x)
        return self.nonlinearity(x)


class FNO(nn.Module):
    """
    Fourier Neural Operator architecture for solving PDEs.

    Lifts input to a higher-dimensional latent space, passes it through sequential
    Fourier layers, and projects back to the target output space.
    """

    def __init__(
        self,
        modes: tuple[int, ...],
        in_channels: int,
        out_channels: int,
        layer_shapes: tuple[int],
        norm_class: str | None = None,
        lift_hidden_dims: tuple[int, ...] = (),
        projection_hidden_dims: tuple[int, ...] = (),
        skip_hidden_dims: tuple[int, ...] = (),
        nonlinearity: str = "relu",
        pointwise_dropout: float = 0.0,
    ) -> None:
        """Initializes the FNO model.

        Args:
            modes: Number of Fourier modes to keep. Dimension is inferred from length of this tuple.
            layer_shapes: List of channel dimensions for Fourier layers.
            norm_class: Normalization name (optional).
                Supported values: "none", "batch", "instance", "layer".
            in_channels: Number of input features.
            out_channels: Number of output features.
            lift_hidden_dims: Hidden channel sizes for pointwise lift MLP.
            projection_hidden_dims: Hidden channel sizes for pointwise projection MLP.
            skip_hidden_dims: Hidden channel sizes for each Fourier-layer pointwise skip MLP.
            nonlinearity: Activation name used in all pointwise MLPs and Fourier layers.
                Supported values: "relu", "gelu", "silu", "tanh", "elu", "leaky_relu".
            pointwise_dropout: Dropout rate used in all pointwise MLP blocks.
        """
        super().__init__()

        dim = len(modes)
        assert dim in (1, 2, 3), "Only 1D, 2D and 3D FNO are supported"

        self.lift = PointwiseMLP(
            in_channels=in_channels,
            out_channels=layer_shapes[0],
            hidden_dims=lift_hidden_dims,
            nonlinearity=nonlinearity,
            dropout=pointwise_dropout,
        )
        self.fourier_layers = nn.ModuleList(
            [
                FourierLayer(
                    in_channels=in_c,
                    out_channels=out_c,
                    modes=modes,
                    norm_class=norm_class,
                    skip_hidden_dims=skip_hidden_dims,
                    nonlinearity=nonlinearity,
                    pointwise_dropout=pointwise_dropout,
                )
                for in_c, out_c in zip(layer_shapes[:-1], layer_shapes[1:])
            ]
        )
        self.project = PointwiseMLP(
            in_channels=layer_shapes[-1],
            out_channels=out_channels,
            hidden_dims=projection_hidden_dims,
            nonlinearity=nonlinearity,
            dropout=pointwise_dropout,
        )

    def forward(self, vt: torch.Tensor) -> torch.Tensor:
        """Runs the full FNO forward pass.

        Args:
            vt: Input tensor of shape `(batch, in_channels, *spatial)`.

        Returns:
            Output tensor of shape `(batch, out_channels, *spatial)`.
        """
        # vt: (batch, in_channels, *spatial)
        lifted = self.lift(vt)  # (batch, layer_shapes[0], *spatial)
        fouriered = lifted
        for layer in self.fourier_layers:
            fouriered = layer(fouriered)  # (batch, layer_shapes[i], *spatial)
        projected = self.project(fouriered)  # (batch, out_channels, *spatial)
        return projected


class LpLoss:
    """Relative $L_p$ loss, commonly used for PDE surrogate models like FNO."""

    def __init__(
        self, d: int = 2, p: int = 2, size_average: bool = True, reduction: bool = True
    ) -> None:
        """Initializes the relative $L_p$ loss.

        Args:
            d: Physical dimension parameter retained for compatibility.
            p: Norm degree used in the relative error.
            size_average: When reducing, compute mean if True, else sum.
            reduction: Whether to reduce batch losses to a scalar.
        """
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the relative p-norm loss.

        Args:
            x: Predicted tensor.
            y: Ground truth tensor.

        Returns:
            Computed relative loss as a scalar or per-sample tensor.
        """
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
        """Computes the relative $L_p$ loss.

        Args:
            x: Predicted tensor.
            y: Ground-truth tensor.

        Returns:
            Relative error as a scalar or per-sample tensor, depending on `reduction`.
        """
        return self.rel(x, y)
