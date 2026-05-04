"""
plot_hand_joints_3d.py  v2
==========================
Extracts hand joint keypoints from cleaned_hand_data_raw.csv and renders
an interactive 3D skeleton plot.

NEW in v2
---------
  • In-app selector  – navigate rows with Prev / Next buttons or type an
    index / frame_id into the text box and press Enter or click "Go".
  • Equal axes       – all three axes share the same scale so the hand
    shape is not distorted.

Usage
-----
  python plot_hand_joints_3d.py                          # opens GUI at row 0
  python plot_hand_joints_3d.py --index 5               # start at row 5
  python plot_hand_joints_3d.py --frame 101_joints       # start at frame_id
  python plot_hand_joints_3d.py --index 0 --save hand.png
"""

import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, TextBox
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Joint definitions
# ---------------------------------------------------------------------------
# FINGER_PREFIXES = {
#     "Thumb":   "TH",
#     "Index":   "F1",
#     "Middle":  "F2",
#     "Ring":    "F3",
#     "Pinky":   "F4",
# }
FINGER_PREFIXES = {
    "Thumb":   "TH",
    "Pinky":   "F1",
    "Ring":     "F2",
    "Middle":    "F3",
    "Index":   "F4",
}
FINGER_JOINTS = ["KNU1_B", "KNU1_A", "KNU2_A", "KNU3_A"]

FINGER_COLORS = {
    "Thumb":  "#e63946",
    "Index":  "#457b9d",
    "Middle": "#e9c46a",
    "Ring":   "#f4a261",
    "Pinky":  "#2a9d8f",
}
PALM_COLOR = "#a8dadc"
BG_COLOR   = "#0d1117"


def col(prefix, suffix, axis):
    return f"{prefix}_{suffix}_{axis}"


def extract_joints(row):
    joints = {}
    joints["Palm"] = (row["PALM_POSITION_X"],
                      row["PALM_POSITION_Y"],
                      row["PALM_POSITION_Z"])
    for finger, prefix in FINGER_PREFIXES.items():
        for suffix in FINGER_JOINTS:
            joints[f"{finger}_{suffix}"] = (
                row[col(prefix, suffix, "X")],
                row[col(prefix, suffix, "Y")],
                row[col(prefix, suffix, "Z")],
            )
    return joints


def build_skeleton():
    # Each bone: (joint_a, joint_b, color, dotted)
    # dotted=True  → dashed line (palm→knuckle connector, hidden KNU1_B segment)
    # KNU1_B (base knuckle) is hidden: not rendered as a point, connections dotted
    bones = []
    for finger in FINGER_PREFIXES:
        color = FINGER_COLORS[finger]
        # chain: KNU1_B, KNU1_A, KNU2_A, KNU3_A
        chain = [f"{finger}_{s}" for s in FINGER_JOINTS]
        knu1b, knu1a, knu2a, knu3a = chain

        if finger == "Thumb":
            # Thumb: skip KNU1_B entirely — dotted line goes Palm → KNU1_A
            bones.append(("Palm", knu1a, color, True))
            bones.append((knu1a, knu2a, color, False))
            bones.append((knu2a, knu3a, color, False))
        else:
            # Other fingers: Palm→KNU1_B dotted (KNU1_B hidden),
            #                KNU1_B→KNU1_A dotted (still hidden node),
            #                KNU1_A→KNU2_A→KNU3_A solid
            bones.append(("Palm", knu1b, color, True))
            bones.append((knu1b,  knu1a, color, False))
            bones.append((knu1a,  knu2a, color, False))
            bones.append((knu2a,  knu3a, color, False))
    return bones


BONES = build_skeleton()

# Joints to hide (not rendered as scatter points)
HIDDEN_JOINTS = {f"{finger}_KNU1_B" for finger in FINGER_PREFIXES if finger == "Thumb"}


def set_equal_axes(ax, joints):
    """Force all three axes to the same scale (cube bounding box)."""
    xs = [v[0] for v in joints.values()]
    ys = [v[1] for v in joints.values()]
    zs = [v[2] for v in joints.values()]
    max_range = max(
        max(xs) - min(xs),
        max(ys) - min(ys),
        max(zs) - min(zs),
    ) / 2.0
    mid_x = (max(xs) + min(xs)) / 2
    mid_y = (max(ys) + min(ys)) / 2
    mid_z = (max(zs) + min(zs)) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def draw_hand(ax, joints):
    """Clear axes and redraw bones, joints, and fingertip labels."""
    ax.cla()

    # Bones — dotted for palm connectors / hidden KNU1_B segments, solid otherwise
    for a, b, color, dotted in BONES:
        ls = (0, (4, 4))  # custom dash: 4pt on, 4pt off — matches reference image
        ax.plot([joints[a][0], joints[b][0]],
                [joints[a][1], joints[b][1]],
                [joints[a][2], joints[b][2]],
                color=color, linewidth=2.5, alpha=0.85,
                linestyle=ls if dotted else "-")

    # Palm point
    px, py, pz = joints["Palm"]
    ax.scatter(px, py, pz, color=PALM_COLOR, s=120, zorder=5,
               edgecolors="white", linewidths=0.8)

    # Finger knuckles + fingertip labels — skip hidden KNU1_B joints
    for finger in FINGER_PREFIXES:
        color = FINGER_COLORS[finger]
        xs_f, ys_f, zs_f = [], [], []
        for suffix in FINGER_JOINTS:
            label = f"{finger}_{suffix}"
            if label in HIDDEN_JOINTS:
                continue          # KNU1_B: skip scatter point
            x, y, z = joints[label]
            xs_f.append(x); ys_f.append(y); zs_f.append(z)
        ax.scatter(xs_f, ys_f, zs_f, color=color, s=60, zorder=5,
                   edgecolors="white", linewidths=0.6)
        tx, ty, tz = joints[f"{finger}_KNU3_A"]
        ax.text(tx, ty, tz + 1.5, finger, fontsize=7.5, color="white",
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2",
                          fc=color, ec="none", alpha=0.75))

    set_equal_axes(ax, joints)

    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors="gray", labelsize=7)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_xlabel("X", color="gray", fontsize=9)
    ax.set_ylabel("Y", color="gray", fontsize=9)
    ax.set_zlabel("Z", color="gray", fontsize=9)


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------

# Load Data
@st.cache_data  # This prevents reloading the CSV on every click
def load_data():
    return pd.read_csv("data/normalised_hand_data_DATA18REMOVED.csv")

df = load_data()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Navigation")

# 1. Search by Video/Frame
search_vid = st.sidebar.text_input("Video ID (e.g. 4)", help="Enter number only")
search_frm = st.sidebar.text_input("Frame ID (e.g. 948)", help="Enter number only")

# Logic to find index based on search
start_idx = 0
if search_vid or search_frm:
    vid_full = f"data_{search_vid}" if search_vid.isdigit() else search_vid
    frm_full = f"{search_frm}_joints" if search_frm.isdigit() else search_frm
    
    mask = pd.Series([True] * len(df))
    if search_vid: mask &= (df["video_id"] == vid_full)
    if search_frm: mask &= (df["frame_id"] == frm_full)
    
    matches = df[mask]
    if not matches.empty:
        start_idx = int(matches.index[0])
    else:
        st.sidebar.error("No match found. Showing row 0.")

# 2. Row Slider
idx = st.sidebar.slider("Select Row Index", 0, len(df)-1, start_idx)

# --- MAIN DISPLAY ---
st.title("3D Hand Joint Visualizer")
row = df.iloc[idx]
st.write(f"**Current Video:** {row['video_id']} | **Frame:** {row['frame_id']} | **Row:** {idx}")

# Create the plot
fig = plt.figure(figsize=(10, 8))
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_subplot(111, projection="3d")

# Draw the hand using your existing function
joints = extract_joints(row)
draw_hand(ax, joints)

# Add Legend
patches = [mpatches.Patch(color=c, label=f) for f, c in FINGER_COLORS.items()]
patches.append(mpatches.Patch(color=PALM_COLOR, label="Palm"))
ax.legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.35, labelcolor="white")

# Render to Streamlit
st.pyplot(fig)


# ---------------------------------------------------------------------------
# Static save helper
# ---------------------------------------------------------------------------

def save_plot(df, idx, save_path):
    row    = df.iloc[idx]
    joints = extract_joints(row)

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax  = fig.add_subplot(111, projection="3d")
    draw_hand(ax, joints)

    patches = [mpatches.Patch(color=c, label=f)
               for f, c in FINGER_COLORS.items()]
    patches.append(mpatches.Patch(color=PALM_COLOR, label="Palm"))
    ax.legend(handles=patches, loc="upper left", fontsize=8,
              framealpha=0.35, labelcolor="white")
    ax.set_title(
        f"video: {row['video_id']}   frame: {row['frame_id']}   (row {idx})",
        color="white", fontsize=11, pad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------