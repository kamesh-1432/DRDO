"""
hybrid_labeling.py

Implements the HYBRID approach for labeling each PRI folder as
ship-present (1) or ship-absent (0):

    Step 1: Load all 32 channels for each PRI
    Step 2: Compute a detection metric (peak-to-average ratio) per PRI
    Step 3: Auto-classify into: confident-present / confident-absent / borderline
    Step 4: Generate a plot for borderline (and a sample of confident) PRIs
            for manual visual verification
    Step 5: Save final labels to a CSV

USAGE
-----
1) Fully automatic mode (no plots, uses only the threshold - fast, for testing):
       python3 hybrid_labeling.py --mode auto

2) Interactive mode (recommended - matches what your guide wants):
       python3 hybrid_labeling.py --mode interactive
   -> For each PRI needing manual review, a plot window / image is shown,
      and you type 1 (ship) or 0 (no ship) in the terminal.

3) Generate review images to disk instead of interactive popups
   (useful if you're on a machine without a display):
       python3 hybrid_labeling.py --mode save_images
"""

import numpy as np
import pandas as pd
import os
import glob
import argparse
import matplotlib.pyplot as plt
from scipy import signal as sp_signal

DATASET_DIR = "Dataset"
OUTPUT_CSV = "final_labels.csv"
REVIEW_IMG_DIR = "review_plots"


# ----------------------------
# Step 1: Loading
# ----------------------------
def load_pri(pri_path, n_channels=32):
    """Load all channel_XX.txt files for one PRI folder into a (32, N) array."""
    channel_files = sorted(glob.glob(os.path.join(pri_path, "channel_*.txt")))
    if len(channel_files) != n_channels:
        print(f"Warning: expected {n_channels} channels in {pri_path}, "
              f"found {len(channel_files)}")
    data = [np.loadtxt(f) for f in channel_files]
    return np.array(data)  # shape: (n_channels, n_samples)


# ----------------------------
# Step 2: Detection metric
# ----------------------------
def compute_detection_metric(pri_data):
    """
    Combine the 32 channels (simple energy stacking - summing power across
    channels boosts a real echo, which is coherent across channels, while
    random per-channel noise partially averages out).

    Returns a peak-to-average ratio: how much the loudest moment in the
    combined signal stands out above the typical background level.
    A real echo -> high ratio. Pure noise -> ratio close to 1.
    """
    combined_power = np.sum(pri_data ** 2, axis=0)  # sum of squares across channels
    peak = np.max(combined_power)
    average = np.mean(combined_power)
    ratio = peak / average if average > 0 else 0
    return ratio, combined_power


# ----------------------------
# Step 3: Auto-classification into 3 buckets
# ----------------------------
def classify_metric(ratio, low_thresh, high_thresh):
    if ratio >= high_thresh:
        return "confident_present"
    elif ratio <= low_thresh:
        return "confident_absent"
    else:
        return "borderline"


def calibrate_thresholds(metrics, low_percentile=40, high_percentile=75):
    """
    Auto-pick thresholds from the DATA ITSELF rather than a hardcoded number,
    since noise floor will vary by recording setup. You can also just
    hardcode thresholds once you've learned what 'normal' looks like for
    your real dataset.
    """
    low_thresh = np.percentile(metrics, low_percentile)
    high_thresh = np.percentile(metrics, high_percentile)
    return low_thresh, high_thresh


# ----------------------------
# Step 4: Visualization for manual review
# ----------------------------
def plot_pri(pri_id, pri_data, combined_power, ratio, save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # Top: a few representative channel waveforms
    for ch in [0, 8, 16, 24]:
        axes[0].plot(pri_data[ch], alpha=0.7, label=f"ch{ch}")
    axes[0].set_title(f"{pri_id} - sample channel waveforms")
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(fontsize=8)

    # Bottom: combined power across all 32 channels (this is what the
    # detection metric is computed from - the clearest view of an echo)
    axes[1].plot(combined_power)
    axes[1].set_title(f"Combined power across 32 channels "
                       f"(peak-to-avg ratio = {ratio:.2f})")
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Power")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
    else:
        plt.show()


# ----------------------------
# Main pipeline
# ----------------------------
def main(mode="interactive", spot_check_fraction=0.15):
    pri_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "PRI_*")))
    if not pri_folders:
        raise SystemExit(f"No PRI folders found in '{DATASET_DIR}/'. "
                          f"Run generate_synthetic_dataset.py first, or point "
                          f"DATASET_DIR to your real data folder.")

    print(f"Found {len(pri_folders)} PRI folders. Loading and scoring all of them...")

    records = []
    for pri_path in pri_folders:
        pri_id = os.path.basename(pri_path)
        pri_data = load_pri(pri_path)
        ratio, combined_power = compute_detection_metric(pri_data)
        records.append({
            "PRI_id": pri_id,
            "path": pri_path,
            "ratio": ratio,
        })

    df = pd.DataFrame(records)

    # Step 3: calibrate thresholds from the metric distribution
    low_thresh, high_thresh = calibrate_thresholds(df["ratio"].values)
    print(f"Auto-calibrated thresholds -> low: {low_thresh:.2f}, high: {high_thresh:.2f}")

    df["bucket"] = df["ratio"].apply(lambda r: classify_metric(r, low_thresh, high_thresh))
    print(df["bucket"].value_counts())

    # Provisional auto-labels
    df["auto_label"] = df["bucket"].map({
        "confident_present": 1,
        "confident_absent": 0,
        "borderline": None,
    })

    final_labels = {}

    if mode == "auto":
        # Fully automatic: borderline cases default to a coin's-throw via
        # threshold midpoint comparison (NOT recommended for real work -
        # only useful to quickly test the pipeline runs end-to-end)
        for _, row in df.iterrows():
            if row["auto_label"] is not None:
                final_labels[row["PRI_id"]] = int(row["auto_label"])
            else:
                final_labels[row["PRI_id"]] = int(row["ratio"] >= (low_thresh + high_thresh) / 2)

    elif mode in ("interactive", "save_images"):
        if mode == "save_images":
            os.makedirs(REVIEW_IMG_DIR, exist_ok=True)

        # Which PRIs need a human look:
        # 1) ALL borderline cases
        # 2) a random spot-check sample of confident cases (sanity check on thresholds)
        borderline = df[df["bucket"] == "borderline"]
        confident = df[df["bucket"] != "borderline"]
        spot_check = confident.sample(frac=spot_check_fraction, random_state=1) \
            if len(confident) > 0 else confident

        to_review = pd.concat([borderline, spot_check]).drop_duplicates(subset="PRI_id")
        print(f"\n{len(to_review)} PRIs selected for manual review "
              f"({len(borderline)} borderline + {len(spot_check)} spot-checked confident)")

        # Auto-accept everything NOT selected for review
        for _, row in df.iterrows():
            if row["PRI_id"] not in to_review["PRI_id"].values:
                final_labels[row["PRI_id"]] = int(row["auto_label"])

        # Manually review the selected ones
        for _, row in to_review.iterrows():
            pri_data = load_pri(row["path"])
            _, combined_power = compute_detection_metric(pri_data)

            if mode == "save_images":
                save_path = os.path.join(REVIEW_IMG_DIR, f"{row['PRI_id']}.png")
                plot_pri(row["PRI_id"], pri_data, combined_power, row["ratio"], save_path)
                print(f"Saved review plot: {save_path} "
                      f"(bucket={row['bucket']}, ratio={row['ratio']:.2f})")
                # default to auto suggestion; you can hand-edit the CSV after review
                suggestion = row["auto_label"] if pd.notna(row["auto_label"]) else \
                    int(row["ratio"] >= (low_thresh + high_thresh) / 2)
                final_labels[row["PRI_id"]] = int(suggestion)
            else:
                plot_pri(row["PRI_id"], pri_data, combined_power, row["ratio"])
                suggestion = "?" if pd.isna(row["auto_label"]) else int(row["auto_label"])
                ans = input(f"{row['PRI_id']} (bucket={row['bucket']}, "
                            f"ratio={row['ratio']:.2f}, auto-suggestion={suggestion}) "
                            f"-> Ship present? [1=yes, 0=no]: ").strip()
                fallback = 0 if pd.isna(row["auto_label"]) else int(row["auto_label"])
                final_labels[row["PRI_id"]] = int(ans) if ans in ("0", "1") else fallback

    else:
        raise ValueError("mode must be one of: auto, interactive, save_images")

    # Step 5: Save final CSV
    out_df = pd.DataFrame({
        "PRI_id": list(final_labels.keys()),
        "label": list(final_labels.values()),
    }).sort_values("PRI_id")
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFinal labels saved to {OUTPUT_CSV}")
    print(out_df["label"].value_counts())

    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "interactive", "save_images"],
                         default="save_images",
                         help="auto = fully automatic (fast test); "
                              "interactive = pop-up plots + terminal input; "
                              "save_images = writes review plots to disk instead of popups")
    args = parser.parse_args()
    main(mode=args.mode)
