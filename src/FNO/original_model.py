import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO as NeuralOperatorFNO


class OriginalFNO(nn.Module):
    def __init__(
        self,
        dim: int,
        modes: int,
        layer_shapes: tuple[int],
        in_channels: int = 1,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        n_modes = (modes,) * dim
        hidden_channels = layer_shapes[0]
        # Our FNO builds one Fourier block per adjacent pair in layer_shapes.
        # For (width,) * L this is L-1 blocks, so match that behavior here.
        n_layers = max(1, len(layer_shapes) - 1)

        self.model = NeuralOperatorFNO(
            # Keep the same Fourier dimensionality and channels as our implementation.
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            # lifting_channel_ratio=0 forces neuralop lifting to a single linear
            # ChannelMLP layer (effectively a 1x1 channel map, like our 1x1 lift).
            lifting_channel_ratio=0,
            # DIFFERENCE projection in neuralop is internally a 2-layer ChannelMLP
            # in this class. ratio=1 is the closest to a simple 1x1 project.
            projection_channel_ratio=1,
            # Do not append coordinate channels (we add them manually for now).
            positional_embedding=None,
            # Keep block channel mixer off (ours has no extra channel MLP branch).
            use_channel_mlp=False,
            # No normalization branch (DIFFERENCE ours uses BatchNorm
            # in FourierLayer, they have ada_in, group_norm and instance_norm).
            norm=None,
            # No domain padding / multi-resolution behavior.
            domain_padding=None,
            resolution_scaling_factor=None,
            # No tensorized/factorized spectral weights (dense baseline).
            factorization=None,
            rank=1.0,
            fixed_rank_modes=False,
            implementation="factorized",
            separable=False,
            # No preactivation variant.
            preactivation=False,
            # No stabilizer, no dynamic mode schedule.
            stabilizer=None,
            max_n_modes=None,
            # Match our nonlinearity.
            non_linearity=F.relu,
            # Linear skip in spectral block.
            fno_skip="linear",
            # Full precision path (avoid mixed/half precision differences).
            fno_block_precision="full",
            # Inactive because use_channel_mlp=False (kept explicit for clarity)
            channel_mlp_dropout=0.0,
            channel_mlp_expansion=0.5,
            channel_mlp_skip="linear",
        )
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim not in (1, 2, 3):
            raise ValueError(f"Unsupported dimension: {self.dim}")

        # Our datasets and training loop use channel-first tensors:
        #   1D -> (batch, channels, x)
        #   2D -> (batch, channels, x, y)
        #   3D -> (batch, channels, x, y, z)
        # neuraloperator.models.FNO expects the same layout.
        return self.model(x)
