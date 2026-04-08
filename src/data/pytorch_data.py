from src.data.generate_data import generate_darcy_data, generate_burgers_data

import torch
from torch.utils.data import Dataset


class BurgersDataset(Dataset):
    """
    PyTorch Dataset for the 1D Burgers' equation.

    Loads pre-generated simulation data and appends spatial grid coordinates
    to the input features. Optional subsampling can reduce grid resolution.
    """

    def __init__(self, n_samples: int, subsample: int = 1) -> None:
        """
        Initializes the Burgers dataset.

        Args:
            n_samples: Number of samples to include.
            subsample: Resolution reduction factor.
        """
        super().__init__()
        self.subsample = subsample

        data = generate_burgers_data(n_samples=n_samples)

        self.a = torch.tensor(data["a"], dtype=torch.float32)
        self.u = torch.tensor(data["u"], dtype=torch.float32)

        # Spatial Sub-sampling
        if self.subsample > 1:
            self.a = self.a[:, :: self.subsample]
            self.u = self.u[:, :: self.subsample]

        self.n_samples, self.n_x = self.a.shape

        # Append spatial grid coordinates (x in [0, 1]) as an additional channel
        self.grid = torch.linspace(0, 1, self.n_x)

        # Add channel dimension
        self.grid = self.grid.unsqueeze(0)
        self.a = self.a.unsqueeze(1)
        self.u = self.u.unsqueeze(1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        a_i = self.a[idx]
        u_i = self.u[idx]

        # Concatenate 'a' and 'grid' along the channel dimension (dim=0) (b, c, x, y)
        x_i = torch.cat([a_i, self.grid], dim=0)

        return x_i, u_i


class DarcyDataset(Dataset):
    """
    PyTorch Dataset for the 2D Darcy Flow equation.

    Processes permeability coefficients 'a' to predict steady-state fluid flow 'u'
    on a 2D grid, including spatial coordinates as additional input channels.
    """

    def __init__(self, n_samples: int, subsample: int = 1) -> None:
        """
        Initializes the Darcy dataset.

        Args:
            n_samples: Number of samples to include.
            subsample: Resolution reduction factor.
        """
        super().__init__()
        self.subsample = subsample

        data = generate_darcy_data(n_samples=n_samples)

        self.a = torch.tensor(data["a"], dtype=torch.float32)
        self.u = torch.tensor(data["u"], dtype=torch.float32)

        # Spatial Sub-sampling
        if self.subsample > 1:
            self.a = self.a[:, :: self.subsample, :: self.subsample]
            self.u = self.u[:, :: self.subsample, :: self.subsample]

        self.n_samples, self.n_x, self.n_y = self.a.shape

        # Append spatial grid coordinates (x, y in [0, 1]) as additional channels
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(0, 1, self.n_x),
            torch.linspace(0, 1, self.n_y),
            indexing="ij",
        )
        # Add channel dimension
        self.grid_x = grid_x.unsqueeze(0)
        self.grid_y = grid_y.unsqueeze(0)
        self.a = self.a.unsqueeze(1)
        self.u = self.u.unsqueeze(1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        a_i = self.a[idx]
        u_i = self.u[idx]

        # Concatenate 'a' and 'grid' along the channel dimension (dim=0) (b, c, x, y)
        x_i = torch.cat([a_i, self.grid_x, self.grid_y], dim=0)

        return x_i, u_i


def get_dataset(args, n_samples: int = 1000) -> tuple[Dataset, Dataset, int, int]:
    """
    Factory function to load and split the specified PDE dataset.

    Handles configuration-driven loading for Burgers or Darcy datasets,
    manages train/test splits with a fixed seed, and infers core metadata.

    Args:
        args: Config object with dataset and train_split.
        n_samples: Total samples to draw.

    Returns:
        tuple: (train_dataset, test_dataset, in_channels, dim)
    """
    if args.dataset == "burgers":
        dataset = BurgersDataset(n_samples=n_samples, subsample=args.subsample)
        train_size = int(args.train_split * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        # 'a' + 'x' grid coordinates (1D)
        in_channels = 2
        dim = 1

    elif args.dataset == "darcy":
        dataset = DarcyDataset(n_samples=n_samples, subsample=args.subsample)
        train_size = int(args.train_split * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        # 'a' + 'x' and 'y' grid coordinates (2D)
        in_channels = 3
        dim = 2

    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    return train_dataset, test_dataset, in_channels, dim
