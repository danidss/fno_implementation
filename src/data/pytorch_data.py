import os
import glob
import argparse
import h5py
import torch
from torch.utils.data import Dataset


class BurgersDataset(Dataset):
    def __init__(self, data_path: str, subsample: int = 1) -> None:
        super().__init__()
        self.subsample = subsample

        with h5py.File(data_path, "r") as f:
            # a is the initial condition, u is the solution
            self.a = torch.tensor(f["a"][:], dtype=torch.float32)
            self.u = torch.tensor(f["u"][:], dtype=torch.float32)

        # Spatial Sub-sampling
        if self.subsample > 1:
            self.a = self.a[:, :: self.subsample]
            self.u = self.u[:, :: self.subsample]

        self.n_samples, self.n_x = self.a.shape

        # Append spatial grid coordinates (x in [0, 1]) as an additional channel
        self.grid = torch.linspace(0, 1, self.n_x).reshape(1, self.n_x, 1)

        # Add channel dimension
        self.a = self.a.unsqueeze(-1)
        self.u = self.u.unsqueeze(-1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        a_i = self.a[idx]
        u_i = self.u[idx]
        grid_i = self.grid[0]

        # Concatenate 'a' and 'grid' along the channel dimension (dim=-1)
        x_i = torch.cat([a_i, grid_i], dim=-1)

        return x_i, u_i


class DarcyDataset(Dataset):
    def __init__(self, data_path, subsample=1):
        super().__init__()
        self.subsample = subsample

        with h5py.File(data_path, "r") as f:
            # a: (N, X, Y) coefficient, u: (N, X, Y) solution
            self.a = torch.tensor(f["a"][:], dtype=torch.float32)
            self.u = torch.tensor(f["u"][:], dtype=torch.float32)

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
        self.grid_x = grid_x.reshape(1, self.n_x, self.n_y, 1)
        self.grid_y = grid_y.reshape(1, self.n_x, self.n_y, 1)

        # Add channel dimension
        self.a = self.a.unsqueeze(-1)
        self.u = self.u.unsqueeze(-1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        a_i = self.a[idx]
        u_i = self.u[idx]
        grid_x_i = self.grid_x[0]
        grid_y_i = self.grid_y[0]

        x_i = torch.cat([a_i, grid_x_i, grid_y_i], dim=-1)

        return x_i, u_i


def get_dataset(
    args: argparse.Namespace, data_dir: str
) -> tuple[Dataset, Dataset, int, int]:
    if args.dataset == "burgers":
        files = glob.glob(os.path.join(data_dir, "burgers_*.h5"))
        if not files:
            raise FileNotFoundError(
                "Burgers data not found. Please run generate_data.py first."
            )

        dataset = BurgersDataset(files[0], subsample=args.subsample)
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
        files = glob.glob(os.path.join(data_dir, "darcy_*.h5"))
        if not files:
            raise FileNotFoundError(
                "Darcy Flow data not found. Please run generate_data.py first."
            )

        dataset = DarcyDataset(files[0], subsample=args.subsample)
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
