"""
plot_avg_hands_by_class.py
──────────────────────────
Plots the average hand pose (3 views: front, side, top) for each
predicted class in predictions_open_set.csv, excluding "unknown".

Usage:
    python plot_avg_hands_by_class.py
    python plot_avg_hands_by_class.py --pred path/to/preds.csv --data path/to/data.csv
    python plot_avg_hands_by_class.py --save                   # save to PNG instead of showing
    python plot_avg_hands_by_class.py --label-col my_col       # custom prediction column name
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
# Constants (kept in sync with original viewer)
# ──────────────────────────────────────────────────────────────────────────────
BG_COLOR = "#0d1117"
PANEL_COLOR = "#12181f"

FINGER_PREFIXES = {
    "Thumb":  "TH",
    "Pinky":  "F1",
    "Ring":   "F2",
    "Middle": "F3",
    "Index":  "F4",
}
FINGER_JOINTS = ["KNU1_B", "KNU1_A", "KNU2_A", "KNU3_A"]
FINGER_COLORS = ["#e63946", "#2a9d8f", "#e9c46a", "#f4a261", "#457b9d"]

VIEWS = {
    "Front": (1, 2),  # YZ
    "Side":  (0, 2),  # XZ
    "Top":   (1, 0),  # YX
}
AXIS_LABELS = {
    "Front": ("−Z →", "Y ↑"),
    "Side":  ("X →",  "Y ↑"),
    "Top":   ("Z →",  "X ↑"),
}

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers (identical logic to viewer)
# ──────────────────────────────────────────────────────────────────────────────
def get_coord_cols():
    cols = ["PALM_POSITION_X", "PALM_POSITION_Y", "PALM_POSITION_Z"]
    for prefix in FINGER_PREFIXES.values():
        for suffix in FINGER_JOINTS:
            cols += [f"{prefix}_{suffix}_X", f"{prefix}_{suffix}_Y", f"{prefix}_{suffix}_Z"]
    return cols


def row_to_points(row):
    cols = get_coord_cols()
    coords = row[cols].to_numpy(dtype=float)
    return coords.reshape(-1, 3)  # (21, 3)


def center_hand(points):
    """Translate so palm (index 0) is at origin."""
    return points - points[0]


def mean_hand(rows_df: pd.DataFrame) -> np.ndarray:
    """
    Average hand shape across a set of DataFrame rows.
    Each hand is centred before averaging so palm offsets don't blur the shape.
    Returns (21, 3) array.
    """
    centered = np.stack([center_hand(row_to_points(row)) for _, row in rows_df.iterrows()])
    return centered.mean(axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────────────────────────────────────
def plot_hand_2d_proj(points, ax, axes=(0, 1), alpha=0.95, lw=2.5):
    """Draw one hand skeleton on *ax* using the two coordinate axes given."""
    i, j = axes
    base = 1  # first joint column index (after palm at 0)

    for f in range(5):
        color = FINGER_COLORS[f]
        finger_pts = points[base: base + 4]

        if f == 0:  # Thumb: skip KNU1_B (base knuckle) for the solid segment
            visible = finger_pts[1:]
            ax.plot([points[0, i], visible[0, i]],
                    [points[0, j], visible[0, j]],
                    color=color, alpha=alpha * 0.55, lw=lw, linestyle="--")
            ax.plot(visible[:, i], visible[:, j], color=color, alpha=alpha, lw=lw)
            ax.scatter(visible[:, i], visible[:, j], c=color, s=28, alpha=alpha, zorder=3)
        else:
            ax.plot([points[0, i], finger_pts[0, i]],
                    [points[0, j], finger_pts[0, j]],
                    color=color, alpha=alpha * 0.55, lw=lw, linestyle="--")
            ax.plot(finger_pts[:, i], finger_pts[:, j], color=color, alpha=alpha, lw=lw)
            ax.scatter(finger_pts[:, i], finger_pts[:, j], c=color, s=28, alpha=alpha, zorder=3)

        base += 4

    # Palm marker
    ax.scatter(points[0, i], points[0, j], c="tomato", s=55, zorder=4)


def _style_ax(ax, view_name, invert_x=False):
    ax.set_facecolor(PANEL_COLOR)
    ax.set_aspect("equal")
    ax.axis("off")
    xl, yl = AXIS_LABELS[view_name]
    ax.set_title(f"{view_name}\n{xl}  {yl}", color="#8899aa", fontsize=7.5, pad=4)
    if invert_x:
        ax.invert_xaxis()


# ──────────────────────────────────────────────────────────────────────────────
# Main plot
# ──────────────────────────────────────────────────────────────────────────────
def plot_avg_by_class(df: pd.DataFrame, label_col: str, save: bool = True,
                      out_path: str = "avg_hands_by_class.png"):
    
    path = os.path.join(OUT_DIR, out_path)

    classes = sorted(df[label_col].dropna().unique())
    n_classes = len(classes)
    n_views = len(VIEWS)  # 3

    # ── Layout ────────────────────────────────────────────────────────────────
    # One column per class, 3 rows (one per view) + a header row for the label
    fig = plt.figure(figsize=(3.4 * n_classes, 10), facecolor=BG_COLOR)

    # outer grid: header row (tiny) + 3 view rows
    outer = gridspec.GridSpec(
        4, n_classes,
        height_ratios=[0.18, 1, 1, 1],
        hspace=0.08,
        wspace=0.04,
        left=0.02, right=0.98, top=0.93, bottom=0.06,
    )

    view_names = list(VIEWS.keys())

    for c_idx, cls in enumerate(classes):
        subset = df[df[label_col] == cls]
        avg_pts = mean_hand(subset)

        # Class label header
        ax_label = fig.add_subplot(outer[0, c_idx])
        ax_label.set_facecolor(BG_COLOR)
        ax_label.axis("off")
        ax_label.text(
            0.5, 0.5, str(cls),
            color="white", fontsize=12, fontweight="bold",
            ha="center", va="center", transform=ax_label.transAxes,
        )
        ax_label.text(
            0.5, -0.15, f"n = {len(subset):,}",
            color="#556677", fontsize=8,
            ha="center", va="center", transform=ax_label.transAxes,
        )

        # Three view panels
        for v_idx, view_name in enumerate(view_names):
            ax = fig.add_subplot(outer[v_idx + 1, c_idx])
            plot_hand_2d_proj(avg_pts, ax, axes=VIEWS[view_name])
            _style_ax(ax, view_name, invert_x=(view_name == "Top"))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_y = 0.025
    for f, (name, color) in enumerate(zip(FINGER_PREFIXES.keys(), FINGER_COLORS)):
        fig.text(0.12 + f * 0.13, legend_y, f"● {name}", color=color,
                 fontsize=8, ha="center", va="bottom")
    fig.text(0.12 + 5 * 0.13, legend_y, "● Palm", color="tomato",
             fontsize=8, ha="center", va="bottom")

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle("Average Hand Pose per Class", color="white", fontsize=14,
                 fontweight="bold", y=0.975)

    if save:
        plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG_COLOR)
        print(f"Saved → {path}")
    else:
        plt.tight_layout()
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot average hand per classification label")
    parser.add_argument("--pred",      default=r"outputs\final outputs\predictions_open_set.csv",
                        help="Path to predictions CSV")
    parser.add_argument("--data",      default=r"data\cleaned_normalised_data_NOdata18.csv",
                        help="Path to joint coordinate CSV")
    parser.add_argument("--label-col", default="ensemble_open_set",
                        help="Column name in predictions CSV containing the class labels")
    parser.add_argument("--unknown",   default="unknown",
                        help="Label value to exclude (default: 'unknown')")
    parser.add_argument("--save",      action="store_true",
                        help="Save figure to PNG instead of opening interactive window")
    parser.add_argument("--out",       default="avg_hands_by_class.png",
                        help="Output PNG path (only used with --save)")
    args = parser.parse_args()

    # ── Load & merge ──────────────────────────────────────────────────────────
    preds = pd.read_csv(args.pred)
    data  = pd.read_csv(args.data)
    df    = pd.merge(preds, data, on=["video_id", "frame_id"])

    # ── Filter unknowns ───────────────────────────────────────────────────────
    before = len(df)
    df = df[df[args.label_col].astype(str).str.lower() != args.unknown.lower()].copy()
    print(f"Loaded {before:,} rows → {len(df):,} after removing '{args.unknown}'")
    print(f"Classes: {sorted(df[args.label_col].unique())}")

    plot_avg_by_class(df, label_col=args.label_col, out_path=args.out)