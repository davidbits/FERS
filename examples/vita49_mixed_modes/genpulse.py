import os

import h5py
import numpy as np


def generate_pulse_file(filename="pulse.h5", sample_rate=5.0e6, pulse_width=1.0e-6):
    num_samples = int(sample_rate * pulse_width)
    i_data = np.ones(num_samples, dtype=np.float64)
    q_data = np.zeros(num_samples, dtype=np.float64)

    if os.path.exists(filename):
        os.remove(filename)

    with h5py.File(filename, "w") as h5_file:
        h5_file.create_group("I").create_dataset("value", data=i_data)
        h5_file.create_group("Q").create_dataset("value", data=q_data)

    print(f"Generated pulse file '{filename}' with {num_samples} samples.")


if __name__ == "__main__":
    generate_pulse_file()
