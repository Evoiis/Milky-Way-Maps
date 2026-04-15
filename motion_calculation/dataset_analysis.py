"""
Orbit Dataset Distribution Analysis
Analyzes the distribution of input features across a dataset:
- x0, y0, z0, vx0, vy0, vz0, t  (inputs)
- Derived: r0 (orbital radius offset), displacement_mag, speed
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ── Config ────────────────────────────────────────────────────────────────────

DATA_SET_NAME = "13S"
# DATA_PATH  = "training_data_" + DATA_SET_NAME
DATA_PATH = "data/dataset_" + DATA_SET_NAME + "/training_data"
OUTPUT_DIR = "dataset_analysis_output/distribution_output_" + DATA_SET_NAME
N_BINS     = 60

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────

def load_data(folder_path):
    files = sorted(glob.glob(folder_path + "/orbit_train_*.npy"))
    print(f"Loading {len(files)} files...")
    data = np.concatenate([np.load(f) for f in files], axis=0)
    print(f"Loaded {len(data):,} samples")
    return data

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_histograms(values_dict, filename, suptitle, log_y=False):
    n = len(values_dict)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for ax, (label, values) in zip(axes, values_dict.items()):
        ax.hist(values, bins=N_BINS, log=log_y, color='steelblue', edgecolor='none', alpha=0.85)
        ax.set_xlabel(label)
        ax.set_ylabel("Count" + (" (log)" if log_y else ""))
        ax.set_title(
            f"{label}\n"
            f"mean={values.mean():.2f}  std={values.std():.2f}\n"
            f"min={values.min():.2f}  max={values.max():.2f}"
        )
        ax.grid(True, alpha=0.3)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(suptitle, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_2d(x, y, xlabel, ylabel, title, filename, bins=80):
    fig, ax = plt.subplots(figsize=(7, 6))
    h = ax.hist2d(x, y, bins=bins, cmap='viridis')
    fig.colorbar(h[3], ax=ax, label="Count")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_r0_vs_t(r0, t, filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 2D histogram
    axes[0].hist2d(t, r0, bins=80, cmap='viridis')
    axes[0].set_xlabel("t (Gyr)")
    axes[0].set_ylabel("r0 (pc)")
    axes[0].set_title("r0 vs t (2D histogram)")
    axes[0].grid(True, alpha=0.2)

    # r0 split by t sign
    axes[1].hist(r0[t < 0], bins=N_BINS, alpha=0.6, label="t < 0 (backward)", color='steelblue')
    axes[1].hist(r0[t > 0], bins=N_BINS, alpha=0.6, label="t > 0 (forward)",  color='coral')
    axes[1].set_xlabel("r0 (pc)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("r0 distribution by time direction")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def print_summary(label, values):
    p = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    print(f"\n  {label}:")
    print(f"    mean={values.mean():.3f}  std={values.std():.3f}")
    print(f"    min={values.min():.3f}  max={values.max():.3f}")
    print(f"    p1={p[0]:.3f}  p5={p[1]:.3f}  p25={p[2]:.3f}  "
          f"p50={p[3]:.3f}  p75={p[4]:.3f}  p95={p[5]:.3f}  p99={p[6]:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    data = load_data(DATA_PATH)

    # Raw columns
    x0  = data[:, 0].astype(np.float32)
    y0  = data[:, 1].astype(np.float32)
    z0  = data[:, 2].astype(np.float32)
    vx0 = data[:, 3].astype(np.float32)
    vy0 = data[:, 4].astype(np.float32)
    vz0 = data[:, 5].astype(np.float32)
    t   = data[:, 6].astype(np.float32)

    # Derived
    r0          = np.sqrt(x0**2 + y0**2) - 8000
    speed       = np.sqrt(vx0**2 + vy0**2 + vz0**2)
    r_gal       = np.sqrt(x0**2 + y0**2)           # galactocentric radius
    r_helio     = np.sqrt((x0 - 8000)**2 + y0**2 + z0**2)  # heliocentric distance

    # ── Summary stats ─────────────────────────────────────────────────────────
    print("\n── Dataset Distribution Summary ─────────────────────────────────────")
    print(f"  N samples: {len(data):,}")
    for label, arr in [
        ("x0 (pc)",        x0),
        ("y0 (pc)",        y0),
        ("z0 (pc)",        z0),
        ("vx0 (pc/Gyr)",   vx0),
        ("vy0 (pc/Gyr)",   vy0),
        ("vz0 (pc/Gyr)",   vz0),
        ("t (Gyr)",        t),
        ("r0 (pc)",        r0),
        ("speed (pc/Gyr)", speed),
        ("r_gal (pc)",     r_gal),
        ("r_helio (pc)",   r_helio),
    ]:
        print_summary(label, arr)

    # t sign balance
    n_fwd = (t > 0).sum()
    n_bwd = (t < 0).sum()
    print(f"\n  t > 0 (forward):  {n_fwd:,} ({100*n_fwd/len(t):.1f}%)")
    print(f"  t < 0 (backward): {n_bwd:,} ({100*n_bwd/len(t):.1f}%)")
    print("─────────────────────────────────────────────────────────────────────\n")

    # ── r0 coverage ───────────────────────────────────────────────────────────
    print("r0 coverage buckets:")
    for lo, hi in [(-3000, -1000), (-1000, -200), (-200, 200), (200, 1000), (1000, 3000)]:
        n = ((r0 >= lo) & (r0 < hi)).sum()
        print(f"  r0 in [{lo:>6}, {hi:>5}): {n:>10,}  ({100*n/len(r0):.2f}%)")

    # ── Plots ─────────────────────────────────────────────────────────────────

    # Raw inputs
    plot_histograms(
        {"x0 (pc)": x0, "y0 (pc)": y0, "z0 (pc)": z0,
         "vx0 (pc/Gyr)": vx0, "vy0 (pc/Gyr)": vy0, "vz0 (pc/Gyr)": vz0,
         "t (Gyr)": t},
        filename="inputs_distribution.png",
        suptitle="Input Feature Distributions"
    )

    # Derived quantities
    plot_histograms(
        {"r0 (pc)": r0, "speed (pc/Gyr)": speed,
         "r_gal (pc)": r_gal, "r_helio (pc)": r_helio,
         "|t| (Gyr)": np.abs(t)},
        filename="derived_distribution.png",
        suptitle="Derived Quantity Distributions"
    )

    # Log-scale versions to expose tails
    plot_histograms(
        {"x0 (pc)": x0, "y0 (pc)": y0, "z0 (pc)": z0,
         "vx0 (pc/Gyr)": vx0, "vy0 (pc/Gyr)": vy0, "vz0 (pc/Gyr)": vz0,
         "t (Gyr)": t},
        filename="inputs_distribution_log.png",
        suptitle="Input Feature Distributions (log y)",
        log_y=True
    )

    # 2D plots
    plot_2d(x0, y0, "x0 (pc)", "y0 (pc)",
            "Initial position in x-y plane", "xy_positions.png")

    plot_2d(r_gal, z0, "r_gal (pc)", "z0 (pc)",
            "Galactocentric radius vs z", "rgal_vs_z.png")

    plot_2d(r0, speed, "r0 (pc)", "speed (pc/Gyr)",
            "Orbital radius offset vs speed", "r0_vs_speed.png")

    plot_r0_vs_t(r0, t, "r0_vs_t.png")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()