import os
import json
import hashlib
import h5py
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from scipy.fft import fft, ifft, fftfreq, fft2, ifft2
from functools import wraps


DATA_DIR = "generated_data"
os.makedirs(DATA_DIR, exist_ok=True)


def cache_incremental_hdf5(base_filepath):
    """
    Decorator that caches generated data and incrementally appends to it if
    more samples are requested than currently exist in the cache.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Hash only the physics parameters
            assert "n_samples" in kwargs, "Must specify n_samples as a keyword argument"
            requested_samples = kwargs.get("n_samples")
            physics_params = {k: v for k, v in kwargs.items() if k != "n_samples"}

            canonical_string = json.dumps(
                physics_params, sort_keys=True, separators=(",", ":")
            )
            param_hash = hashlib.sha256(canonical_string.encode("utf-8")).hexdigest()[
                :16
            ]

            target_filename = f"{base_filepath}_{param_hash}.h5"
            os.makedirs(os.path.dirname(target_filename) or ".", exist_ok=True)

            # Handle existing cache
            if os.path.exists(target_filename):
                with h5py.File(target_filename, "a") as f:
                    # Check current number of samples using the first dataset
                    sample_key = list(f.keys())[0]
                    current_samples = f[sample_key].shape[0]

                    if current_samples >= requested_samples:
                        print(
                            f"Loading {requested_samples} cached samples from: {target_filename}"
                        )
                        return {key: f[key][:requested_samples] for key in f.keys()}
                    else:
                        needed_samples = requested_samples - current_samples
                        print(
                            f"Found {current_samples} samples. Generating {needed_samples} more..."
                        )

                        # Generate only the missing data
                        kwargs["n_samples"] = needed_samples
                        new_data_dict = func(*args, **kwargs)

                        for key, new_array in new_data_dict.items():
                            dataset = f[key]
                            dataset.resize((current_samples + needed_samples), axis=0)
                            dataset[current_samples:] = new_array

                        return {key: f[key][:requested_samples] for key in f.keys()}

            # Generating from scratch
            print(f"Generating {requested_samples} new samples for: {target_filename}")
            data_dict = func(*args, **kwargs)

            with h5py.File(target_filename, "w") as f:
                for key, array_data in data_dict.items():
                    # Enable chunking and maxshape to allow future resizing along the sample axis (axis 0)
                    dataset_shape = array_data.shape
                    max_shape = (None,) + dataset_shape[1:]
                    f.create_dataset(
                        key,
                        data=array_data,
                        maxshape=max_shape,
                        chunks=True,
                        compression="gzip",
                    )

                # Embed physics parameters as metadata
                for param_key, param_val in physics_params.items():
                    f.attrs[param_key] = param_val

            return data_dict

        return wrapper

    return decorator


def generate_grf_1d(N: int, scale: float, alpha: float, power: float) -> np.ndarray:
    k = fftfreq(N, 1 / N)
    eig = scale * ((2 * np.pi * k) ** 2 + alpha) ** (-power)
    eig[0] = 0.0
    noise = np.random.randn(N) + 1j * np.random.randn(N)
    return np.real(ifft(noise * np.sqrt(eig) * N))


def generate_grf_2d(N: int, scale: float, alpha: float, power: float) -> np.ndarray:
    kx = fftfreq(N, 1 / N)
    ky = fftfreq(N, 1 / N)
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    eig = scale * ((2 * np.pi * kx) ** 2 + (2 * np.pi * ky) ** 2 + alpha) ** (-power)
    eig[0, 0] = 0.0
    noise = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    return np.real(ifft2(noise * np.sqrt(eig) * (N**2)))


@cache_incremental_hdf5(f"{DATA_DIR}/burgers")
def generate_burgers_data(
    n_samples: int = 1000, **kwargs: float
) -> dict[str, np.ndarray]:
    """
    1D Burgers' equation using split-step method.
    """
    params = {
        "resolution": 8192,
        "nu": 0.1,
        "t_end": 1.0,
        "dt": 1e-4,
        "grf_scale": 625,
        "grf_alpha": 25,
        "grf_power": 2,
    }
    params.update(kwargs)

    N = params["resolution"]
    nu = params["nu"]
    dt = params["dt"]
    steps = int(params["t_end"] / dt)

    k = fftfreq(N, 1 / N)
    k_sq = (2 * np.pi * k) ** 2
    heat_mult = np.exp(-nu * k_sq * dt)
    ik = 1j * 2 * np.pi * k

    a_data = np.zeros((n_samples, N))
    u_data = np.zeros((n_samples, N))

    for i in tqdm(range(n_samples), desc="Generating Burgers' data"):
        u0 = generate_grf_1d(
            N, params["grf_scale"], params["grf_alpha"], params["grf_power"]
        )
        a_data[i] = u0

        u_hat = fft(u0)
        for _ in range(steps):
            u = np.real(ifft(u_hat))
            non_linear_term = -0.5 * fft(u**2) * ik
            u_hat = (u_hat + dt * non_linear_term) * heat_mult

        u_data[i] = np.real(ifft(u_hat))

    return {"a": a_data, "u": u_data}


@cache_incremental_hdf5(f"{DATA_DIR}/darcy")
def generate_darcy_data(
    n_samples: int = 1000, **kwargs: float
) -> dict[str, np.ndarray]:
    """
    2D Darcy Flow equation using second-order finite difference scheme.
    """
    params = {
        "resolution": 421,
        "grf_scale": 1.0,
        "grf_alpha": 9,
        "grf_power": 2,
        "forcing": 1.0,
    }
    params.update(kwargs)

    N = params["resolution"]
    h = 1.0 / (N - 1)

    a_data = np.zeros((n_samples, N, N))
    u_data = np.zeros((n_samples, N, N))

    for i in tqdm(range(n_samples), desc="Generating Darcy data"):
        grf = generate_grf_2d(
            N, params["grf_scale"], params["grf_alpha"], params["grf_power"]
        )
        a = np.where(grf > 0, 12.0, 3.0)
        a_data[i] = a

        a_x = 0.5 * (a[:-1, :] + a[1:, :])
        a_y = 0.5 * (a[:, :-1] + a[:, 1:])

        diag = np.zeros((N, N))
        diag[1:-1, 1:-1] = (
            a_x[1:, 1:-1] + a_x[:-1, 1:-1] + a_y[1:-1, 1:] + a_y[1:-1, :-1]
        )

        idx = np.arange(N**2).reshape(N, N)
        rows, cols, vals = [], [], []

        for x in range(1, N - 1):
            for y in range(1, N - 1):
                curr = idx[x, y]
                rows.extend([curr] * 5)
                cols.extend(
                    [curr, idx[x - 1, y], idx[x + 1, y], idx[x, y - 1], idx[x, y + 1]]
                )
                vals.extend(
                    [
                        diag[x, y] / h**2,
                        -a_x[x - 1, y] / h**2,
                        -a_x[x, y] / h**2,
                        -a_y[x, y - 1] / h**2,
                        -a_y[x, y] / h**2,
                    ]
                )

        for x in range(N):
            for y in range(N):
                if x == 0 or x == N - 1 or y == 0 or y == N - 1:
                    curr = idx[x, y]
                    rows.append(curr)
                    cols.append(curr)
                    vals.append(1.0)

        A = sp.csr_matrix((vals, (rows, cols)), shape=(N**2, N**2))
        b = np.full(N**2, params["forcing"])
        b[idx[0, :]] = 0
        b[idx[-1, :]] = 0
        b[idx[:, 0]] = 0
        b[idx[:, -1]] = 0

        u = splinalg.spsolve(A, b).reshape(N, N)
        u_data[i] = u

    return {"a": a_data, "u": u_data}


@cache_incremental_hdf5(f"{DATA_DIR}/navier_stokes")
def generate_navier_stokes_data(
    n_samples: int = 1000, **kwargs: float
) -> dict[str, np.ndarray]:
    """
    2D Navier-Stokes equation (vorticity form) using pseudospectral method.
    """
    params = {
        "resolution": 256,
        "nu": 1e-3,
        "t_end": 50.0,
        "dt": 1e-4,
        "record_step": 1.0,
        "grf_scale": 7 ** (1.5),
        "grf_alpha": 49,
        "grf_power": 2.5,
    }
    params.update(kwargs)

    N = params["resolution"]
    nu = params["nu"]
    dt = params["dt"]
    record_interval = int(params["record_step"] / dt)
    num_records = int(params["t_end"] / params["record_step"])

    kx = fftfreq(N, 1 / N)
    ky = fftfreq(N, 1 / N)
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    k_sq = (2 * np.pi * kx) ** 2 + (2 * np.pi * ky) ** 2
    k_sq[0, 0] = 1.0

    ikx = 1j * 2 * np.pi * kx
    iky = 1j * 2 * np.pi * ky

    X, Y = np.meshgrid(
        np.linspace(0, 1, N, endpoint=False),
        np.linspace(0, 1, N, endpoint=False),
        indexing="ij",
    )
    f = 0.1 * (np.sin(2 * np.pi * (X + Y)) + np.cos(2 * np.pi * (X + Y)))
    f_hat = fft2(f)

    # Crank-Nicolson for linear diffusion
    cn_num = 1.0 - 0.5 * dt * nu * k_sq
    cn_den = 1.0 / (1.0 + 0.5 * dt * nu * k_sq)

    a_data = np.zeros((n_samples, N, N))
    u_data = np.zeros((n_samples, N, N, num_records))

    for i in tqdm(range(n_samples), desc="Generating Navier-Stokes data"):
        w0 = generate_grf_2d(
            N, params["grf_scale"], params["grf_alpha"], params["grf_power"]
        )
        a_data[i] = w0
        w_hat = fft2(w0)

        for step in range(1, int(params["t_end"] / dt) + 1):
            psi_hat = w_hat / k_sq
            psi_hat[0, 0] = 0.0

            u_hat = iky * psi_hat
            v_hat = -ikx * psi_hat

            dw_dx_hat = ikx * w_hat
            dw_dy_hat = iky * w_hat

            u = np.real(ifft2(u_hat))
            v = np.real(ifft2(v_hat))
            dw_dx = np.real(ifft2(dw_dx_hat))
            dw_dy = np.real(ifft2(dw_dy_hat))

            convection = u * dw_dx + v * dw_dy
            conv_hat = fft2(convection)

            w_hat = cn_den * (cn_num * w_hat + dt * (f_hat - conv_hat))

            if step % record_interval == 0:
                u_data[i, ..., (step // record_interval) - 1] = np.real(ifft2(w_hat))

    return {"a": a_data, "u": u_data}


if __name__ == "__main__":
    print("Generating Burgers' data...")
    generate_burgers_data(n_samples=1000)

    print("Generating Darcy Flow data...")
    generate_darcy_data(n_samples=1000)

    print("Generating Navier-Stokes data...")
    generate_navier_stokes_data(n_samples=100)
