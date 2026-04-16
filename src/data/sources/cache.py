import hashlib
import json
import os
from functools import wraps

import h5py


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
