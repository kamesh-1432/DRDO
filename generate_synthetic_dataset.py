"""
generate_synthetic_dataset.py

Creates a SYNTHETIC active sonar dataset that mimics the structure described:

    Dataset/
        PRI_0001/
            channel_00.txt
            channel_01.txt
            ...
            channel_31.txt
        PRI_0002/
            ...

Each channel_XX.txt is a plain text file with ONE NUMBER PER LINE
(just like what you saw when your guide opened a channel in Notepad).

This is NOT real sonar data - it's a physically-motivated simulation:
- Background = ocean ambient noise + reverberation (random noise)
- Some PRIs also contain a simulated "target echo": a short burst of
  higher energy at a random time-of-flight (range), arriving at slightly
  different times/amplitudes across the 32 channels (like a real echo
  would, since the array elements are physically spaced apart).

Run this once to create the dataset, then run hybrid_labeling.py on it.
"""

import numpy as np
import pandas as pd
import os

# ----------------------------
# Configuration
# ----------------------------
N_PRI = 60              # number of PRI folders to generate
N_CHANNELS = 32          # channels per PRI (fixed, matches your task)
N_SAMPLES = 2000         # samples per channel per PRI (i.e. length of the listening window)
SHIP_PROBABILITY = 0.45  # fraction of PRIs that will actually contain a ship echo
OUTPUT_DIR = "Dataset"
SEED = 42

rng = np.random.default_rng(SEED)


def generate_background_noise(n_channels, n_samples, noise_std=1.0):
    """Ambient ocean noise + reverberation: just random noise per channel."""
    return rng.normal(0, noise_std, size=(n_channels, n_samples))


def inject_echo(signal, n_channels, n_samples, echo_strength=6.0):
    """
    Adds a simulated target echo: a short burst of energy, arriving at a
    random range (sample index), with small inter-channel delay/amplitude
    differences to mimic a real wavefront hitting a physical array.
    """
    echo_len = rng.integers(20, 60)          # duration of the echo burst
    echo_start = rng.integers(200, n_samples - echo_len - 200)  # random range
    base_echo = np.sin(np.linspace(0, 8 * np.pi, echo_len)) * echo_strength

    for ch in range(n_channels):
        # small random delay (samples) and amplitude variation across channels,
        # like a real wavefront arriving across a physical array
        delay = rng.integers(-2, 3)
        amp_variation = rng.uniform(0.7, 1.0)
        start = echo_start + delay
        end = start + echo_len
        if 0 <= start and end < n_samples:
            signal[ch, start:end] += base_echo * amp_variation

    return signal


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ground_truth = []  # we keep the TRUE labels aside, only to score your work later

    for i in range(1, N_PRI + 1):
        pri_id = f"PRI_{i:04d}"
        pri_path = os.path.join(OUTPUT_DIR, pri_id)
        os.makedirs(pri_path, exist_ok=True)

        signal = generate_background_noise(N_CHANNELS, N_SAMPLES)

        has_ship = rng.random() < SHIP_PROBABILITY
        if has_ship:
            signal = inject_echo(signal, N_CHANNELS, N_SAMPLES)

        for ch in range(N_CHANNELS):
            fname = os.path.join(pri_path, f"channel_{ch:02d}.txt")
            # one number per line, like Notepad showed
            np.savetxt(fname, signal[ch], fmt="%.6f")

        ground_truth.append({"PRI_id": pri_id, "true_label": int(has_ship)})

    gt_df = pd.DataFrame(ground_truth)
    gt_df.to_csv("ground_truth_labels.csv", index=False)

    print(f"Generated {N_PRI} PRI folders in '{OUTPUT_DIR}/' "
          f"({N_CHANNELS} channels each, {N_SAMPLES} samples/channel)")
    print(f"Ship-present count: {gt_df['true_label'].sum()} / {N_PRI}")
    print("Ground truth saved to ground_truth_labels.csv "
          "(kept separately - only used later to check your hybrid pipeline's accuracy)")


if __name__ == "__main__":
    main()
