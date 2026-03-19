import torch
import torch.nn as nn
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
        n_layers = len(layer_shapes)

        self.model = NeuralOperatorFNO(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            positional_embedding="grid",
        )
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1:
            x = x.permute(0, 2, 1)
            out = self.model(x)
            return out.permute(0, 2, 1)
        elif self.dim == 2:
            x = x.permute(0, 3, 1, 2)
            out = self.model(x)
            return out.permute(0, 2, 3, 1)
        elif self.dim == 3:
            x = x.permute(0, 4, 1, 2, 3)
            out = self.model(x)
            return out.permute(0, 2, 3, 4, 1)
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}")
