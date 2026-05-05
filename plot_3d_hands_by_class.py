"""
plot_3d_hands_by_class.py
─────────────────────────
Plots overlaid 3-D hand skeletons for each predicted class (excluding
"unknown"), drawing a random sample of frames per class so the cloud of
poses shows the true spread rather than a single average.

Two layout modes
----------------
  all    (default) – one panel per class in a single row
  single           – one large panel for a specific class (--class)

Usage:
    python plot_3d_hands_by_class.py                          # all classes
    python plot_3d_hands_by_class.py --class "open_hand"      # one class, large
    python plot_3d_hands_by_class.py --samples 40 --save
    python plot_3d_hands_by_class.py --pred path/to/preds.csv --data path/to/data.csv
    python plot_3d_hands_by_class.py --label-col my_col --unknown unknown
    python plot_3d_hands_by_class.py --elev 20 --azim -60     # change viewing angle
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── Output folders ─────────────────────────────────────────────────────────────
OUT_DIR   = "outputs/final outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
BG_COLOR    = "#0d1117"
PANEL_COLOR = "#0d1117"   # keep 3-D panes dark too

FINGER_PREFIXES = {
    "Thumb":  "TH",
    "Pinky":  "F1",
    "Ring":   "F2",
    "Middle": "F3",
    "Index":  "F4",
}
FINGER_JOINTS = ["KNU1_B", "KNU1_A", "KNU2_A", "KNU3_A"]
FINGER_COLORS = ["#e63946", "#2a9d8f", "#e9c46a", "#f4a261", "#457b9d"]

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────
def get_coord_cols():
    cols = ["PALM_POSITION_X", "PALM_POSITION_Y", "PALM_POSITION_Z"]
    for prefix in FINGER_PREFIXES.values():
        for suffix in FINGER_JOINTS:
            cols += [f"{prefix}_{suffix}_X", f"{prefix}_{suffix}_Y", f"{prefix}_{suffix}_Z"]
    return cols


def row_to_points(row):
    coords = row[get_coord_cols()].to_numpy(dtype=float)
    return coords.reshape(-1, 3)   # (21, 3)


def center_hand(points):
    return points - points[0]


# ──────────────────────────────────────────────────────────────────────────────
# 3-D drawing
# ──────────────────────────────────────────────────────────────────────────────
def plot_hand_3d(points, ax, alpha=0.35, lw=1.2):
    """Draw one hand skeleton in 3-D on *ax*."""
    base = 1
    for f in range(5):
        color = FINGER_COLORS[f]
        fp = points[base: base + 4]

        # Palm → base knuckle (dashed)
        ax.plot(
            [points[0, 0], fp[0, 0]],
            [points[0, 1], fp[0, 1]],
            [points[0, 2], fp[0, 2]],
            color=color, alpha=alpha * 0.6, lw=lw, linestyle="--",
        )
        # Knuckle chain
        ax.plot(fp[:, 0], fp[:, 1], fp[:, 2],
                color=color, alpha=alpha, lw=lw)
        ax.scatter(fp[:, 0], fp[:, 1], fp[:, 2],
                   c=color, s=6, alpha=alpha, zorder=3)
        base += 4

    ax.scatter([points[0, 0]], [points[0, 1]], [points[0, 2]],
               c="tomato", s=25, zorder=4)


def _style_ax3d(ax, title: str, n: int, elev: float, azim: float):
    ax.set_facecolor(PANEL_COLOR)
    ax.tick_params(colors="#3a4a5a", labelsize=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#1e2a36")
    ax.yaxis.pane.set_edgecolor("#1e2a36")
    ax.zaxis.pane.set_edgecolor("#1e2a36")
    ax.set_xlabel("X", color="#3a4a5a", fontsize=6, labelpad=2)
    ax.set_ylabel("Y", color="#3a4a5a", fontsize=6, labelpad=2)
    ax.set_zlabel("Z", color="#3a4a5a", fontsize=6, labelpad=2)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"{title}\nn = {n:,}", color="white", fontsize=10,
                 fontweight="bold", pad=6)


# ──────────────────────────────────────────────────────────────────────────────
# Shared legend / finish helpers
# ──────────────────────────────────────────────────────────────────────────────
def _add_legend(fig, n_chosen: int):
    x0 = 0.18
    for f, (name, color) in enumerate(zip(FINGER_PREFIXES.keys(), FINGER_COLORS)):
        fig.text(x0 + f * 0.12, 0.025, f"● {name}",
                 color=color, fontsize=8, ha="center", va="bottom")
    fig.text(x0 + 5 * 0.12, 0.025, "● Palm",
             color="tomato", fontsize=8, ha="center", va="bottom")
    fig.text(0.5, 0.013,
             f"({n_chosen} random samples per class  ·  centred on palm)",
             color="#445566", fontsize=7.5, ha="center", va="bottom")


def _finish(fig, save: bool, out_path: str):
    if save:
        plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG_COLOR)
        print(f"Saved → {out_path}")
    else:
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Mode: all  — one panel per class in a single row
# ──────────────────────────────────────────────────────────────────────────────
def plot_all_classes(
    df: pd.DataFrame,
    label_col: str,
    samples_per_class: int = 30,
    elev: float = 25,
    azim: float = -60,
    save: bool = False,
    out_path: str = "3d_hands_by_class.png",
    seed: int = 42,
):
    path = os.path.join(OUT_DIR, out_path)
    rng     = np.random.default_rng(seed)
    classes = sorted(df[label_col].dropna().unique())
    n_cls   = len(classes)

    panel_w = max(3.8, min(5.0, 24 / n_cls))
    fig = plt.figure(figsize=(panel_w * n_cls, panel_w * 1.05), facecolor=BG_COLOR)
    gs  = gridspec.GridSpec(
        1, n_cls,
        wspace=0.02,
        left=0.01, right=0.99,
        top=0.88, bottom=0.08,
    )

    n_chosen_last = samples_per_class
    for c_idx, cls in enumerate(classes):
        subset      = df[df[label_col] == cls]
        n_total     = len(subset)
        n_chosen    = min(samples_per_class, n_total)
        n_chosen_last = n_chosen
        chosen_idx  = rng.choice(subset.index, size=n_chosen, replace=False)

        ax = fig.add_subplot(gs[0, c_idx], projection="3d")
        _style_ax3d(ax, str(cls), n_total, elev=elev, azim=azim)

        for row_idx in chosen_idx:
            pts = center_hand(row_to_points(df.loc[row_idx]))
            plot_hand_3d(pts, ax, alpha=0.22, lw=1.0)

    _add_legend(fig, n_chosen_last)
    fig.suptitle("Hand Pose Distribution per Class  (3-D overlay)",
                 color="white", fontsize=13, fontweight="bold", y=0.96)
    _finish(fig, save, path)


# ──────────────────────────────────────────────────────────────────────────────
# Mode: single  — one large panel for a specific class
# ──────────────────────────────────────────────────────────────────────────────
def plot_single_class(
    df: pd.DataFrame,
    label_col: str,
    class_name: str,
    samples_per_class: int = 30,
    elev: float = 25,
    azim: float = -60,
    save: bool = False,
    out_path: str = "3d_hands_single.png",
    seed: int = 42,
):
    path = os.path.join(OUT_DIR, out_path)
    subset = df[df[label_col].astype(str) == class_name]
    if subset.empty:
        available = sorted(df[label_col].unique())
        raise ValueError(
            f"Class '{class_name}' not found.\nAvailable classes: {available}"
        )

    rng      = np.random.default_rng(seed)
    n_total  = len(subset)
    n_chosen = min(samples_per_class, n_total)
    chosen_idx = rng.choice(subset.index, size=n_chosen, replace=False)

    fig = plt.figure(figsize=(8, 7.5), facecolor=BG_COLOR)
    ax  = fig.add_subplot(111, projection="3d")
    _style_ax3d(ax, class_name, n_total, elev=elev, azim=azim)

    for row_idx in chosen_idx:
        pts = center_hand(row_to_points(df.loc[row_idx]))
        plot_hand_3d(pts, ax, alpha=0.25, lw=1.1)

    _add_legend(fig, n_chosen)
    fig.suptitle(f"Hand Pose Distribution — '{class_name}'  (3-D overlay)",
                 color="white", fontsize=13, fontweight="bold", y=0.97)
    _finish(fig, save, path)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot overlaid 3-D hand skeletons per classification label"
    )
    parser.add_argument("--pred",      default=r"outputs\final outputs\predictions_open_set.csv")
    parser.add_argument("--data",      default=r"data\cleaned_normalised_data_NOdata18.csv")
    parser.add_argument("--label-col", default="ensemble_open_set",
                        help="Column in predictions CSV with class labels")
    parser.add_argument("--unknown",   default="unknown",
                        help="Label value to exclude")
    parser.add_argument("--class",     dest="single_class", default=None,
                        help="Plot only this class (large single panel). "
                             "Omit to plot all classes side-by-side.")
    parser.add_argument("--samples",   type=int, default=30,
                        help="Max random samples drawn per class (default 30)")
    parser.add_argument("--elev",      type=float, default=25,
                        help="3-D view elevation angle in degrees (default 25)")
    parser.add_argument("--azim",      type=float, default=-60,
                        help="3-D view azimuth angle in degrees (default -60)")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed for sample selection")
    parser.add_argument("--save",      action="store_true",
                        help="Save to PNG instead of opening interactive window")
    parser.add_argument("--out",       default=None,
                        help="Output PNG path (auto-named if omitted)")
    args = parser.parse_args()

    # ── Load & merge ──────────────────────────────────────────────────────────
    preds = pd.read_csv(args.pred)
    data  = pd.read_csv(args.data)
    df    = pd.merge(preds, data, on=["video_id", "frame_id"])

    before = len(df)
    df = df[df[args.label_col].astype(str).str.lower() != args.unknown.lower()].copy()
    print(f"Loaded {before:,} rows → {len(df):,} after removing '{args.unknown}'")
    print(f"Classes: {sorted(df[args.label_col].unique())}")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    common = dict(
        label_col=args.label_col,
        samples_per_class=args.samples,
        elev=args.elev,
        azim=args.azim,
        save=True,
        seed=args.seed,
    )

    if args.single_class:
        out = args.out or f"3d_hands_{args.single_class}.png"
        plot_single_class(df, class_name=args.single_class, out_path=out, **common)
    else:
        out = args.out or "3d_hands_by_class.png"
        plot_all_classes(df, out_path=out, **common)