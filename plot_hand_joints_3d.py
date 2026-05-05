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

class HandViewer:
    def __init__(self, df: pd.DataFrame, start_index: int = 0):
        self.df  = df
        self.idx = max(0, min(start_index, len(df) - 1))

        self.fig = plt.figure(figsize=(11, 8.5))
        self.fig.patch.set_facecolor(BG_COLOR)

        # 3-D axes – leave room at bottom for controls
        self.ax = self.fig.add_axes([0.0, 0.13, 1.0, 0.87], projection="3d")

        # Static legend
        patches = [mpatches.Patch(color=c, label=f)
                   for f, c in FINGER_COLORS.items()]
        patches.append(mpatches.Patch(color=PALM_COLOR, label="Palm"))
        self.ax.legend(handles=patches, loc="upper left", fontsize=8,
                       framealpha=0.35, labelcolor="white")

        # ── Widgets ──────────────────────────────────────────────────────────
        # Layout:  [◀ Prev]  [video N]  [frame N]  [Go]  [Next ▶]  [info]
        btn_bg = "#1e2532"

        ax_prev = self.fig.add_axes([0.02,  0.03, 0.09, 0.06])
        ax_vid  = self.fig.add_axes([0.13,  0.03, 0.20, 0.06])
        ax_frm  = self.fig.add_axes([0.35,  0.03, 0.20, 0.06])
        ax_go   = self.fig.add_axes([0.57,  0.03, 0.07, 0.06])
        ax_next = self.fig.add_axes([0.66,  0.03, 0.09, 0.06])
        ax_info = self.fig.add_axes([0.77,  0.03, 0.21, 0.06])

        self.btn_prev = Button(ax_prev, "◀  Prev", color=btn_bg, hovercolor="#2e3a50")
        self.btn_next = Button(ax_next, "Next  ▶", color=btn_bg, hovercolor="#2e3a50")
        self.btn_go   = Button(ax_go,   "Go",      color="#2a9d8f", hovercolor="#21867a")

        # Two textboxes: bare numbers only (4 → data_4, 948 → 948_joints)
        self.tb_vid = TextBox(ax_vid, "video: ", color=btn_bg, hovercolor="#2e3a50")
        self.tb_frm = TextBox(ax_frm, "frame: ", color=btn_bg, hovercolor="#2e3a50")

        for btn in (self.btn_prev, self.btn_next, self.btn_go):
            btn.label.set_color("white")
            btn.label.set_fontsize(10)
        for tb in (self.tb_vid, self.tb_frm):
            tb.text_disp.set_color("white")
            tb.text_disp.set_fontsize(9)
            tb.label.set_color("#7a8a9a")
            tb.label.set_fontsize(8)

        ax_info.set_axis_off()
        ax_info.set_facecolor(BG_COLOR)
        self.info_text = ax_info.text(0.0, 0.5, "", color="gray", fontsize=8,
                                      va="center", transform=ax_info.transAxes)

        ax_hint = self.fig.add_axes([0.13, 0.005, 0.74, 0.022])
        ax_hint.set_axis_off()
        ax_hint.text(0.0, 0.5,
                     "video: number only  (e.g. 4 = data_4)      "
                     "frame: number only  (e.g. 948 = 948_joints)      "
                     "press Enter in either box or click Go",
                     color="#444e5e", fontsize=7, va="center",
                     transform=ax_hint.transAxes)

        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.btn_go.on_clicked(self._on_go)
        self.tb_vid.on_submit(self._on_vid_submit)
        self.tb_frm.on_submit(self._on_frm_submit)

        self._refresh()

    # ── Helpers: expand bare numbers to full ids ──────────────────────────────

    @staticmethod
    def _expand_video(text):
        t = text.strip()
        return f"data_{t}" if t.isdigit() else t

    @staticmethod
    def _expand_frame(text):
        t = text.strip()
        return f"{t}_joints" if t.isdigit() else t

    @staticmethod
    def _num_video(video_id):
        """'data_4' → '4'  (bare number for display)."""
        return video_id.replace("data_", "")

    @staticmethod
    def _num_frame(frame_id):
        """'948_joints' → '948'  (bare number for display)."""
        return frame_id.replace("_joints", "")

    # ── Navigation ────────────────────────────────────────────────────────────

    def _on_prev(self, _):
        self.idx = max(0, self.idx - 1)
        self._refresh()

    def _on_next(self, _):
        self.idx = min(len(self.df) - 1, self.idx + 1)
        self._refresh()

    def _on_go(self, _):
        self._lookup(self.tb_vid.text, self.tb_frm.text)

    def _on_vid_submit(self, text):
        """Enter in video box: jump to first frame of that video."""
        vid = self._expand_video(text)
        matches = self.df[self.df["video_id"] == vid]
        if not matches.empty:
            self.idx = int(matches.index[0])
            self._refresh()
        else:
            self._show_error(f"video '{vid}' not found")

    def _on_frm_submit(self, text):
        """Enter in frame box: find within current video."""
        current_vid = self.df.iloc[self.idx]["video_id"]
        self._lookup(current_vid, text)

    def _lookup(self, raw_vid, raw_frm):
        vid = self._expand_video(raw_vid)
        frm = self._expand_frame(raw_frm)
        mask = (self.df["video_id"] == vid) & (self.df["frame_id"] == frm)
        matches = self.df[mask]
        if not matches.empty:
            self.idx = int(matches.index[0])
            self._refresh()
        else:
            elsewhere = self.df[self.df["frame_id"] == frm]["video_id"].unique()
            if len(elsewhere):
                self._show_error(
                    f"frame {frm!r} not in video {vid!r} — found in: {list(elsewhere)}")
            else:
                self._show_error(f"Not found: video={vid!r}  frame={frm!r}")

    def _show_error(self, msg):
        self.info_text.set_text(msg)
        self.info_text.set_color("#e63946")
        self.fig.canvas.draw_idle()

    # ── Redraw ───────────────────────────────────────────────────────────────

    def _refresh(self):
        row    = self.df.iloc[self.idx]
        joints = extract_joints(row)
        draw_hand(self.ax, joints)

        vid = row["video_id"]
        frm = row["frame_id"]

        self.ax.set_title(
            f"video: {vid}   frame: {frm}   (row {self.idx})",
            color="white", fontsize=11, pad=10)

        patches = [mpatches.Patch(color=c, label=f)
                   for f, c in FINGER_COLORS.items()]
        patches.append(mpatches.Patch(color=PALM_COLOR, label="Palm"))
        self.ax.legend(handles=patches, loc="upper left", fontsize=8,
                       framealpha=0.35, labelcolor="white")

        # Show bare numbers in the textboxes
        self.tb_vid.eventson = False
        self.tb_frm.eventson = False
        self.tb_vid.set_val(self._num_video(vid))
        self.tb_frm.set_val(self._num_frame(frm))
        self.tb_vid.eventson = True
        self.tb_frm.eventson = True

        self.info_text.set_text(f"row {self.idx + 1} / {len(self.df)}")
        self.info_text.set_color("gray")
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()



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

def main():
    parser = argparse.ArgumentParser(description="Plot hand joints in 3D.")
    parser.add_argument("--csv",   default="data/cleaned_normalised_data_NOdata18.csv")
    parser.add_argument("--index", type=int, default=None,
                        help="Starting row index (0-based)")
    parser.add_argument("--video", type=str, default=None,
                        help="Starting video_id — bare number ok: 4 = data_4")
    parser.add_argument("--frame", type=str, default=None,
                        help="Starting frame_id — bare number ok: 948 = 948_joints")
    parser.add_argument("--save",  type=str, default=None,
                        help="Save static plot to file instead of opening GUI")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        sys.exit(f"ERROR: File not found → {args.csv}")

    # Expand bare numbers to full ids
    def expand_vid(v): return f"data_{v}" if v and v.isdigit() else v
    def expand_frm(f): return f"{f}_joints" if f and f.isdigit() else f

    start = 0
    if args.video or args.frame:
        vid = expand_vid(args.video) if args.video else None
        frm = expand_frm(args.frame) if args.frame else None
        mask = pd.Series([True] * len(df))
        if vid:
            mask &= df["video_id"] == vid
        if frm:
            mask &= df["frame_id"] == frm
        matches = df[mask]
        if matches.empty:
            sys.exit(f"ERROR: video={vid!r} frame={frm!r} not found.")
        start = int(matches.index[0])
    elif args.index is not None:
        if args.index >= len(df):
            sys.exit(f"ERROR: --index {args.index} out of range ({len(df)} rows).")
        start = args.index

    if args.save:
        save_plot(df, start, args.save)
    else:
        viewer = HandViewer(df, start_index=start)
        viewer.show()


if __name__ == "__main__":
    main()