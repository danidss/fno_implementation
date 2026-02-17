from neuralop.data.datasets import DarcyDataset, Burgers1dTimeDataset, NavierStokesDataset


ROOT_DIR = "./data"

def generate_darcy_data(n_train , n_tests, batch_size, test_batch_sizes, train_resolution=16, test_resolutions=[16, 32]):
    dataset = DarcyDataset(
        root_dir=ROOT_DIR,
        n_train=n_train,
        n_tests=n_tests,
        batch_size=batch_size,
        test_batch_sizes=test_batch_sizes,
        train_resolution=train_resolution,
        test_resolutions=test_resolutions
    )
    return dataset
