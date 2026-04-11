"""
OrbitMLP Error Analysis
Analyzes prediction errors across:
- Per-axis (x, y, z)
- Error vs time
- Error vs orbital radius (r0)
- Error vs displacement magnitude
- Worst-case stars
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os

from orbit_mlp import (
    load_norm_stats,
    load_model_from_file,
    predict_batch,
    DEVICE,
    flogger
)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "orbit_mlp_7.8.pt"
# MODEL_PATH    = "./prev_models/" + MODEL_NAME
MODEL_PATH    = MODEL_NAME
NORM_PATH     = "orbit_norm_6.json"
VAL_DATA_PATH = "validation_data_3"
OUTPUT_DIR    = "error_analysis_output/" + MODEL_NAME

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load val data ─────────────────────────────────────────────────────────────

def load_val_data(folder_path):
    files = sorted(glob.glob(folder_path + "/orbit_train_*.npy"))
    print(f"Loading {len(files)} val files...")
    return np.concatenate([np.load(f) for f in files], axis=0)

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_error_vs_binned(values, errors_pc, xlabel, title, filename, n_bins=30):
    bins = np.percentile(values, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_means, bin_p50, bin_p90 = [], [], []

    for i in range(len(bins) - 1):
        mask = (values >= bins[i]) & (values < bins[i+1])
        if mask.sum() > 0:
            bin_means.append(errors_pc[mask].mean())
            bin_p50.append(np.percentile(errors_pc[mask], 50))
            bin_p90.append(np.percentile(errors_pc[mask], 90))
        else:
            bin_means.append(np.nan)
            bin_p50.append(np.nan)
            bin_p90.append(np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bin_centers, bin_means, label="Mean error",      marker='o', ms=4)
    ax.plot(bin_centers, bin_p50,   label="Median error",    marker='s', ms=4)
    ax.plot(bin_centers, bin_p90,   label="90th percentile", marker='^', ms=4, linestyle='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMSE (parsecs)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_per_axis(errors_xyz, filename):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = ['x error (pc)', 'y error (pc)', 'z error (pc)']
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.hist(errors_xyz[:, i], bins=100, log=True)
        ax.set_xlabel(label)
        ax.set_ylabel("Count (log)")
        ax.set_title(f"{label}\nMean={errors_xyz[:,i].mean():.2f} pc  "
                     f"Std={errors_xyz[:,i].std():.2f} pc")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def print_worst_cases(errors_pc, t, r0, displacement_mag, n=20):
    idx = np.argsort(errors_pc)[-n:][::-1]
    print(f"\nTop {n} worst predictions:")
    print(f"{'Rank':>4}  {'RMSE (pc)':>10}  {'t (Gyr)':>10}  {'r0 (pc)':>10}  {'disp_mag (pc)':>14}")
    print("-" * 55)
    for rank, i in enumerate(idx):
        print(f"{rank+1:>4}  {errors_pc[i]:>10.2f}  {t[i]:>10.4f}  {r0[i]:>10.2f}  {displacement_mag[i]:>14.2f}")


def print_summary(errors_pc, errors_xyz):
    print("\n── Summary ──────────────────────────────────────────")
    print(f"  N samples:        {len(errors_pc):,}")
    print(f"  Overall RMSE:     {errors_pc.mean():.4f} pc")
    print(f"  Median error:     {np.median(errors_pc):.4f} pc")
    print(f"  90th percentile:  {np.percentile(errors_pc, 90):.4f} pc")
    print(f"  99th percentile:  {np.percentile(errors_pc, 99):.4f} pc")
    print(f"  Max error:        {errors_pc.max():.4f} pc")
    print()
    for i, axis in enumerate(['x', 'y', 'z']):
        print(f"  {axis} RMSE: {np.abs(errors_xyz[:, i]).mean():.4f} pc")
    print("─────────────────────────────────────────────────────\n")



# ── Main ──────────────────────────────────────────────────────────────────────

def predict_in_chunks(model, norm_stats, inputs, chunk_size=500_000):
    preds = []
    for i in range(0, len(inputs), chunk_size):
        chunk = inputs[i:i+chunk_size]
        preds.append(predict_batch(model, norm_stats, chunk))
    return np.concatenate(preds, axis=0)

def main():
    print(f"Device: {DEVICE}")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    flogger.set_write_to_file(False)

    norm_stats = load_norm_stats(NORM_PATH)
    model      = load_model_from_file(MODEL_PATH)

    data = load_val_data(VAL_DATA_PATH)

    inputs = data[:, :7].astype(np.float32)   # x0, y0, z0, vx0, vy0, vz0, t
    y_true = data[:, 7:].astype(np.float32)   # heliocentric x, y, z positions

    t                = data[:, 6].astype(np.float32)
    r0               = np.sqrt(data[:, 0]**2 + data[:, 1]**2) - 8000

    initial_pos_helio = data[:, :3].copy().astype(np.float32)
    initial_pos_helio[:, 0] -= 8000

    displacement_mag = np.sqrt(np.sum((y_true - initial_pos_helio[:, :3])**2, axis=1))

    print(f"displacement_mag min:  {displacement_mag.min():.1f} pc")
    print(f"displacement_mag max:  {displacement_mag.max():.1f} pc")
    print(f"displacement_mag mean: {displacement_mag.mean():.1f} pc")

    print("Running inference...")
    y_pred = predict_in_chunks(model, norm_stats, inputs)

    errors_xyz = y_pred - y_true
    errors_pc  = np.sqrt(np.mean(errors_xyz**2, axis=1))

    print_summary(errors_pc, errors_xyz)
    print_worst_cases(errors_pc, t, r0, displacement_mag)

    plot_per_axis(np.abs(errors_xyz), "per_axis_error.png")

    plot_error_vs_binned(
        t, errors_pc,
        xlabel="Time t (Gyr)",
        title="Error vs Time",
        filename="error_vs_time.png"
    )
    plot_error_vs_binned(
        np.abs(t), errors_pc,
        xlabel="|t| (Gyr)",
        title="Error vs |Time|",
        filename="error_vs_abs_time.png"
    )
    plot_error_vs_binned(
        r0, errors_pc,
        xlabel="Initial orbital radius r0 (pc)",
        title="Error vs Orbital Radius",
        filename="error_vs_r0.png"
    )
    plot_error_vs_binned(
        displacement_mag, errors_pc,
        xlabel="True displacement magnitude (pc)",
        title="Error vs Displacement Magnitude",
        filename="error_vs_displacement.png"
    )

    print(f"\nAll plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
