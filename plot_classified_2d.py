import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, TextBox

# ---------------------------------------------------------------------------
# Definitions & Constants
# ---------------------------------------------------------------------------
BG_COLOR = "#0d1117"

FINGER_PREFIXES = {
    "Thumb":   "TH",
    "Pinky":   "F1",
    "Ring":    "F2",
    "Middle":  "F3",
    "Index":   "F4",
}
FINGER_JOINTS = ["KNU1_B", "KNU1_A", "KNU2_A", "KNU3_A"]

FINGER_COLORS = ["#e63946", "#2a9d8f", "#e9c46a", "#f4a261", "#457b9d"]

VIEWS = {
    "front": (1, 2),   # YZ
    "side":  (0, 2),   # XZ
    "top":   (1, 0),   # YX
}

# Updated top view label to point +Z to the right
AXIS_LABELS = {
    "front": ("-Z →", "Y ↑"),
    "side":  ("X →", "Y ↑"),
    "top":   ("Z →", "X ↑"),
}

# ---------------------------------------------------------------------------
# Data Processing Logic
# ---------------------------------------------------------------------------
def get_coord_cols():
    cols = ["PALM_POSITION_X", "PALM_POSITION_Y", "PALM_POSITION_Z"]
    for finger, prefix in FINGER_PREFIXES.items():
        for suffix in FINGER_JOINTS:
            cols += [f"{prefix}_{suffix}_X", f"{prefix}_{suffix}_Y", f"{prefix}_{suffix}_Z"]
    return cols

def row_to_points(row):
    cols = get_coord_cols()
    coords = row[cols].to_numpy(dtype=float)
    return coords.reshape(-1, 3)

def center_hand(points):
    palm = points[0]
    return points - palm

def plot_hand_2d_proj(points, ax, axes=(0, 1), alpha=0.95, lw=2.0):
    i, j = axes
    base = 1
    for f in range(5):
        color = FINGER_COLORS[f]
        finger_pts = points[base : base + 4] 

        if f == 0: # Thumb logic from features.py
            thumb_visible = finger_pts[1:]
            ax.plot([points[0, i], thumb_visible[0, i]],
                    [points[0, j], thumb_visible[0, j]],
                    color=color, alpha=alpha * 0.6, lw=lw, linestyle="--")
            ax.plot(thumb_visible[:, i], thumb_visible[:, j],
                    color=color, alpha=alpha, lw=lw)
            ax.scatter(thumb_visible[:, i], thumb_visible[:, j],
                       c=color, s=20, alpha=alpha, zorder=3)
        else:
            ax.plot([points[0, i], finger_pts[0, i]],
                    [points[0, j], finger_pts[0, j]],
                    color=color, alpha=alpha * 0.6, lw=lw, linestyle="--")
            ax.plot(finger_pts[:, i], finger_pts[:, j],
                    color=color, alpha=alpha, lw=lw)
            ax.scatter(finger_pts[:, i], finger_pts[:, j],
                       c=color, s=20, alpha=alpha, zorder=3)
        base += 4
    ax.scatter(points[0, i], points[0, j], c="tomato", s=40, zorder=4)

# ---------------------------------------------------------------------------
# Interactive Viewer
# ---------------------------------------------------------------------------
class HandViewerMultiview:
    def __init__(self, df: pd.DataFrame, start_index: int = 0):
        self.df = df
        self.idx = max(0, min(start_index, len(df) - 1))

        self.fig, self.axes = plt.subplots(1, 3, figsize=(14, 5.5))
        self.fig.patch.set_facecolor(BG_COLOR)
        plt.subplots_adjust(bottom=0.25)

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

        self.tb_vid = TextBox(ax_vid, "video: ", color=btn_bg, hovercolor="#2e3a50")
        self.tb_frm = TextBox(ax_frm, "frame: ", color=btn_bg, hovercolor="#2e3a50")

        # Styling
        for btn in (self.btn_prev, self.btn_next, self.btn_go):
            btn.label.set_color("white")
            btn.label.set_fontsize(10)
        for tb in (self.tb_vid, self.tb_frm):
            tb.text_disp.set_color("white")
            tb.text_disp.set_fontsize(9)
            tb.label.set_color("#7a8a9a")
            tb.label.set_fontsize(8)

        # Info text
        ax_info.set_axis_off()
        ax_info.set_facecolor(BG_COLOR)
        self.info_text = ax_info.text(0.0, 0.5, "", color="gray", fontsize=8,
                                      va="center", transform=ax_info.transAxes)

        # Hint text
        ax_hint = self.fig.add_axes([0.13, 0.005, 0.74, 0.022])
        ax_hint.set_axis_off()
        ax_hint.text(0.0, 0.5,
                     "video: number only  (e.g. 4 = data_4)      "
                     "frame: number only  (e.g. 948 = 948_joints)      "
                     "press Enter in either box or click Go",
                     color="#444e5e", fontsize=7, va="center",
                     transform=ax_hint.transAxes)

        # Events
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.btn_go.on_clicked(self._on_go)
        self.tb_vid.on_submit(self._on_vid_submit)
        self.tb_frm.on_submit(self._on_frm_submit)

        self._refresh()

    # ── Helpers ──────────────────────────────────────────────────────────────
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
        return video_id.replace("data_", "")

    @staticmethod
    def _num_frame(frame_id):
        return frame_id.replace("_joints", "")

    # ── Navigation ───────────────────────────────────────────────────────────
    def _on_prev(self, _):
        self.idx = max(0, self.idx - 1)
        self._refresh()

    def _on_next(self, _):
        self.idx = min(len(self.df) - 1, self.idx + 1)
        self._refresh()

    def _on_go(self, _):
        self._lookup(self.tb_vid.text, self.tb_frm.text)

    def _on_vid_submit(self, text):
        vid = self._expand_video(text)
        matches = self.df[self.df["video_id"] == vid]
        if not matches.empty:
            self.idx = int(matches.index[0])
            self._refresh()
        else:
            self._show_error(f"video '{vid}' not found")

    def _on_frm_submit(self, text):
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
                self._show_error(f"frame {frm!r} not in video {vid!r} — found in: {list(elsewhere)}")
            else:
                self._show_error(f"Not found: video={vid!r}  frame={frm!r}")

    def _show_error(self, msg):
        self.info_text.set_text(msg)
        self.info_text.set_color("#e63946")
        self.fig.canvas.draw_idle()

    # ── Redraw ───────────────────────────────────────────────────────────────
    def _refresh(self):
        row = self.df.iloc[self.idx]
        pts = center_hand(row_to_points(row))
        pred = row.get("ensemble_open_set", "N/A")
        vid = row["video_id"]
        frm = row["frame_id"]

        for v, view_name in enumerate(VIEWS.keys()):
            ax = self.axes[v]
            ax.cla()
            ax.set_facecolor(BG_COLOR)
            ax.set_aspect("equal")
            ax.axis("off")
            
            plot_hand_2d_proj(pts, ax, axes=VIEWS[view_name])
            
            # Mirror the 'top' view horizontally so +Z points right
            if view_name == "top":
                ax.invert_xaxis()
                
            xl, yl = AXIS_LABELS[view_name]
            ax.set_title(f"{view_name.capitalize()} view\n{xl}  {yl}", color="white", fontsize=10)

        self.fig.suptitle(f"Prediction: {pred} | {vid} | {frm}", color="white")
        
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

# ---------------------------------------------------------------------------
# Main Loader
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Use raw strings (r"") to avoid "invalid escape sequence" errors
    pred_path = r"final outputs\predictions_open_set.csv"
    data_path = r"data\normalised_hand_data_DATA18REMOVED.csv"

    try:
        preds = pd.read_csv(pred_path)
        data = pd.read_csv(data_path)
        
        # Merge datasets to get coordinates for each prediction
        df = pd.merge(preds, data, on=["video_id", "frame_id"])
        
        viewer = HandViewerMultiview(df)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")