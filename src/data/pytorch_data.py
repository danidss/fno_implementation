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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--train_split", type=float, default=0.8)
    args = parser.parse_args()

    train_dataset, test_dataset, in_channels, dim = get_dataset(args, args.data_dir)

    print(f"Example input: {train_dataset[0][0]}")
    print(f"Example input shape: {train_dataset[0][0].shape}")
    print(f"Example target: {train_dataset[0][1]}")
    print(f"Example target shape: {train_dataset[0][1].shape}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print(f"Input channels: {in_channels}")
    print(f"Dimension: {dim}")
