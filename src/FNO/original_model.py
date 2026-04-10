import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO as NeuralOperatorFNO


class OriginalFNO(nn.Module):
    """
    Wrapper for the NeuralOperator library's FNO implementation.
    
    Used to benchmark the current implementation against a reference model, 
    with hyperparameters configured for equivalence.
    """

    def __init__(
        self,
        modes: tuple[int, ...],
        layer_shapes: tuple[int],
        in_channels: int = 1,
        out_channels: int = 1,
        norm_class: str | None = "none",
        nonlinearity: str = "relu",
        lift_hidden_dims: tuple[int, ...] = (),
        projection_hidden_dims: tuple[int, ...] = (),
        padding: float | list[float] | tuple[float, ...] = 0.0,
    ) -> None:
        """
        Initializes the reference FNO model.

        Args:
            modes: Frequency modes to truncate per dimension
            layer_shapes: Fourier layer widths
            in_channels: Number of input features
            out_channels: Number of output features
            norm_class: Normalization name. Supported: "none", "batch", "instance", "layer"
            nonlinearity: Activation name. Supported: "relu", "gelu", "silu", "tanh", "elu", "leaky_relu"
            lift_hidden_dims: Hidden dimensions for lift MLP in our model; mapped to neuralop lifting ratio
            projection_hidden_dims: Hidden dimensions for projection MLP in our model; mapped to neuralop projection ratio
            padding: Symmetric spatial padding ratio(s). Use a single float in [0, 1] for
                all dimensions or a list/tuple with one value per dimension in modes.
        """
        super().__init__()

        if len(modes) not in (1, 2, 3):
            raise ValueError(
                f"Unsupported dimension inferred from modes={modes}. Expected 1D, 2D or 3D modes tuple."
            )

        hidden_channels = layer_shapes[0]
        # Our FNO builds one Fourier block per adjacent pair in layer_shapes
        # For (width,) * L this is L-1 blocks, so match that behavior here
        n_layers = max(1, len(layer_shapes) - 1)

        non_linearity = self._resolve_nonlinearity(nonlinearity)
        norm = self._resolve_norm(norm_class)

        lifting_channel_ratio = (
            lift_hidden_dims[0] / hidden_channels if lift_hidden_dims else 0
        )
        projection_channel_ratio = (
            projection_hidden_dims[0] / hidden_channels if projection_hidden_dims else 1
        )

        self.model = NeuralOperatorFNO(
            # Keep the same Fourier dimensionality and channels as our implementation
            n_modes=modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            # Approximate our lift/projection channel widths via neuralop ratios
            lifting_channel_ratio=lifting_channel_ratio,
            projection_channel_ratio=projection_channel_ratio,
            # Best-effort mapping from our normalization choices to neuralop options
            norm=norm,
            # Match our nonlinearity
            non_linearity=non_linearity,
            # Optional domain padding to match our implementation behavior
            domain_padding=padding,
            # Do not append coordinate channels (we add them manually for now)
            positional_embedding=None,
            # Resolution scaling is done in data preprocessing
            resolution_scaling_factor=None,
            # No tensorized/factorized spectral weights (dense baseline)
            factorization=None,
            rank=1.0,
            fixed_rank_modes=False,
            implementation="factorized",
            separable=False,
            # No preactivation variant
            preactivation=False,
            # Static modes in training
            max_n_modes=None,
            # Linear skip in spectral block (soft gating and no skip not implemented)
            fno_skip="linear",
            # Full precision path (no other precisions implemented)
            fno_block_precision="full",
            # No stabilizer (mainly useful for mixed precision, which we don't implement)
            stabilizer=None,
            # ours has no extra channel MLP branch
            use_channel_mlp=False,
            channel_mlp_dropout=0.0,
            channel_mlp_expansion=0.5,
            channel_mlp_skip="linear",
        )

    @staticmethod
    def _resolve_nonlinearity(nonlinearity: str):
        nonlinearity_map = {
            "relu": F.relu,
            "gelu": F.gelu,
            "silu": F.silu,
            "tanh": torch.tanh,
            "elu": F.elu,
            "leaky_relu": F.leaky_relu,
        }
        key = nonlinearity.lower()
        if key not in nonlinearity_map:
            options = ", ".join(sorted(nonlinearity_map))
            raise ValueError(
                f"Unsupported nonlinearity '{nonlinearity}'. Available: {options}."
            )
        return nonlinearity_map[key]

    @staticmethod
    def _resolve_norm(norm_class: str | None):
        if norm_class is None or norm_class.lower() == "none":
            return None
        norm_map = {
            "instance": "instance_norm",
            "layer": "group_norm",
            # neuralop does not expose batch norm in this API; use no norm as closest fallback
            "batch": None,
        }
        key = norm_class.lower()
        if key not in norm_map:
            raise ValueError(
                f"Unsupported norm_class '{norm_class}'. Available: none, batch, instance, layer."
            )
        return norm_map[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Our datasets and training loop use channel-first tensors:
        #   1D -> (batch, channels, x)
        #   2D -> (batch, channels, x, y)
        #   3D -> (batch, channels, x, y, z)
        # neuraloperator.models.FNO expects the same layout
        return self.model(x)
