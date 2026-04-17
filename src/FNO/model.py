import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

from src.utils import build_activation, build_normalization


# TODO consider using (batch, *spatial, channels) format for better pointwise MLP


class SpectralConv(nn.Module):
    """N-dimensional spectral convolution using fftshift-centered mode selection."""

    def __init__(self, in_channels: int, out_channels: int, modes: tuple[int, ...]):
        """Initializes the spectral convolution.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            modes: Number of retained Fourier modes per spatial dimension.
        """
        super().__init__()

        if len(modes) == 0:
            raise ValueError("modes must contain at least one spatial dimension.")
        if any(mode <= 0 for mode in modes):
            raise ValueError(
                f"Each entry in modes must be positive. Received modes={modes}."
            )

        self.out_channels = out_channels
        self.modes = tuple(int(mode) for mode in modes)

        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(  # TODO check initialization
            scale
            * torch.rand(
                in_channels,
                out_channels,
                *self.modes,
                dtype=torch.cfloat,
            )
        )

    # TODO bias?

    @staticmethod
    def _spectral_slices(
        spatial_shape: tuple[int, ...],
        modes: tuple[int, ...],
    ) -> tuple[tuple[slice, ...], tuple[int, ...]]:
        """Builds frequency-domain slices for rfftn output."""
        if len(spatial_shape) != len(modes):
            raise ValueError(
                f"Input has {len(spatial_shape)} spatial dims but modes has {len(modes)}."
            )
        # Because it's rfft, last dim is truncated to floor(N/2)+1, 0 to Nyquist freq
        # The other dims go from [-N/2, N/2) with low freqs in the center
        shifted_shape = (*spatial_shape[:-1], spatial_shape[-1] // 2 + 1)
        slices: list[slice] = []

        # For each axis, calculate the slice object that selects the low-frequency modes
        for axis, (size, mode) in enumerate(zip(shifted_shape, modes)):
            if mode > size:
                raise ValueError(
                    f"Requested modes {modes} exceed spatial resolution in rfftn space {shifted_shape}."
                )

            is_last_axis = axis == len(modes) - 1
            if is_last_axis:
                slices.append(slice(0, mode))
            else:
                start = (size - mode) // 2
                slices.append(slice(start, start + mode))

        return tuple(slices), shifted_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies spectral convolution while retaining centered low-frequency modes."""
        spatial_shape = tuple(x.shape[2:])

        # We compute the slices of the lowest self.modes frequencies
        fft_dims = tuple(range(2, x.ndim))
        freq_slices, rfft_shape = self._spectral_slices(spatial_shape, self.modes)
        freq_index = (slice(None), slice(None), *freq_slices)

        x_ft = fft.rfftn(x, dim=fft_dims)

        # We shift the output freqs from [0,...,N/2,-N/2,...,0) to [-N/2,...,0,...,N/2)
        # Last dim is not shifted as it is [0, N/2] instead, due to real fft
        if len(fft_dims) > 1:
            x_ft = fft.fftshift(x_ft, dim=fft_dims[:-1])

        out_ft = torch.zeros(
            x.shape[0],
            self.out_channels,
            *rfft_shape,
            device=x.device,
            dtype=torch.cfloat,
        )
        # Select only the low frequency modes
        x_low = x_ft[freq_index]
        # We initialize weights as a parameter and directly multiply instead of using
        # a linear layer because frequencies are not mixed in the spectral convolution
        out_ft[freq_index] = torch.einsum("bi...,io...->bo...", x_low, self.weights)

        if len(fft_dims) > 1:
            out_ft = fft.ifftshift(out_ft, dim=fft_dims[:-1])

        return fft.irfftn(out_ft, s=spatial_shape, dim=fft_dims)


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
            hidden_dims: Hidden layer channel sizes used by intermediate pointwise 1x1 convolutions.
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
            layers.append(nn.Conv1d(in_c, out_c, kernel_size=1))

            if not idx == len(channels) - 2:  # No activation or dropout on last layer.
                layers.append(build_activation(nonlinearity))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the pointwise MLP over flattened spatial points.

        Args:
            x: Input tensor of shape `(batch, channels, *spatial)`.

        Returns:
            Output tensor of shape `(batch, out_channels, *spatial)`.
        """
        # x: (batch, channels, *spatial). Flatten spatial axes into a 1D grid.
        _, _, *spatial_shape = x.shape
        points = x.flatten(start_dim=2)  # (batch, channels, n_points)
        points = self.network(points)
        return points.unflatten(2, spatial_shape)


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

        self.spectral_conv = SpectralConv(in_channels, out_channels, modes)
        self.skip_weight = PointwiseMLP(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dims=skip_hidden_dims,
            nonlinearity=nonlinearity,
            dropout=pointwise_dropout,
        )
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
        padding: float | list[float] | tuple[float, ...] = 0.0,
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
            padding: Symmetric spatial padding ratio(s) applied after lift and removed
                before projection. Provide either a single float in [0, 1] or a list/tuple
                of floats in [0, 1] with length equal to len(modes).
        """
        super().__init__()

        if len(modes) == 0:
            raise ValueError("modes must contain at least one spatial dimension.")
        self.padding = self._validate_padding(padding, modes)

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
        lifted = self.lift(vt)

        padded, pad_sizes = self._pad_spatial(lifted, self.padding)

        fouriered = padded
        for layer in self.fourier_layers:
            fouriered = layer(fouriered)

        if pad_sizes is not None:
            fouriered = self._unpad_spatial(fouriered, pad_sizes)

        projected = self.project(fouriered)  # out: (batch, out_channels, *spatial)
        return projected

    @staticmethod
    def _validate_padding(
        padding: float | list[float] | tuple[float, ...],
        modes: tuple[int, ...],
    ) -> tuple[float, ...]:
        """Validates and normalizes padding into one value per spatial dimension."""
        n_dim = len(modes)

        if isinstance(padding, (int, float)):
            value = float(padding)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"padding must be in [0, 1]. Received {padding}.")
            return (value,) * n_dim

        if isinstance(padding, (list, tuple)):
            if len(padding) != n_dim:
                raise ValueError(
                    f"Padding list length ({len(padding)}) must match len(modes) ({n_dim})."
                )
            values = tuple(float(value) for value in padding)
            if any(value < 0.0 or value > 1.0 for value in values):
                raise ValueError(
                    f"Each padding value must be in [0, 1]. Received {padding}."
                )
            return values

        raise TypeError(
            "padding must be a float or a list/tuple of floats. "
            f"Received type {type(padding).__name__}."
        )

    @staticmethod
    def _pad_spatial(
        x: torch.Tensor,
        padding: tuple[float, ...],
    ) -> tuple[torch.Tensor, tuple[int, ...] | None]:
        """
        Applies symmetric zero padding in spatial dimensions. Done to mitigate
        spectral ringing from sharp boundaries and non-periodicity.
        """
        spatial_shape = x.shape[2:]
        pad_sizes = tuple(
            int(round(ratio * size)) for ratio, size in zip(padding, spatial_shape)
        )

        if all(size == 0 for size in pad_sizes):
            return x, None

        pad_spec: list[int] = []
        for size in reversed(pad_sizes):  # Reverse to match F.pad's order
            pad_spec.extend([size, size])
        return F.pad(x, pad_spec, mode="constant", value=0.0), pad_sizes

    @staticmethod
    def _unpad_spatial(x: torch.Tensor, pad_sizes: tuple[int, ...]) -> torch.Tensor:
        """Removes symmetric padding from spatial dimensions."""
        slices: list[slice] = [slice(None), slice(None)]
        for size in pad_sizes:
            if size == 0:
                slices.append(slice(None))
            else:
                slices.append(slice(size, -size))
        return x[tuple(slices)]


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
