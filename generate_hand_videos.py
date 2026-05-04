"""
generate_hand_videos.py
=======================
For every unique video_id in the dataset, generates an MP4 animation
showing the 21 hand joint points moving frame-by-frame exactly as they
were captured.

Each frame is rendered as a 3-panel plot (Top / Front / Side view) so
the full 3-D motion is visible without needing a rotating 3-D axis.

Layout per animation frame
--------------------------
  [ Top view: -Z → / X ↑ ]  [ Front view: -Z → / Y ↑ ]  [ Side view: X → / Y ↑ ]

Output
------
  videos/<video_id>.mp4   – one file per video_id

Usage
-----
  python generate_hand_videos.py
  python generate_hand_videos.py --input  data/normalised_hand_data.csv
                                  --outdir videos
                                  --fps    30
                                  --dpi    120
  python generate_hand_videos.py --video  data_17    # render only one video
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FFMpegWriter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FINGER_PREFIXES = {
    "Thumb":  "TH",
    "Pinky":  "F1",
    "Ring":   "F2",
    "Middle": "F3",
    "Index":  "F4",
}
FINGER_JOINTS = ["KNU1_B", "KNU1_A", "KNU2_A", "KNU3_A"]

FINGER_COLORS = ["#e63946", "#2a9d8f", "#e9c46a", "#f4a261", "#457b9d"]
PALM_COLOR    = "tomato"
BG_COLOR      = "#0d1117"

# View definitions: (horiz_axis, vert_axis, negate_horiz, negate_vert)
# Axes: 0=X  1=Y  2=Z
VIEWS = {
    "Top\n−Z →   X ↑":   (2, 0, True,  False),
    "Front\n−Z →   Y ↑": (2, 1, True,  False),
    "Side\nX →   Y ↑":   (0, 1, False, False),
}

def col(prefix, suffix, axis):
    return f"{prefix}_{suffix}_{axis}"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def coord_columns():
    cols = ["PALM_POSITION_X", "PALM_POSITION_Y", "PALM_POSITION_Z"]
    for prefix in FINGER_PREFIXES.values():
        for suffix in FINGER_JOINTS:
            cols += [col(prefix, suffix, "X"),
                     col(prefix, suffix, "Y"),
                     col(prefix, suffix, "Z")]
    return cols


def row_to_points(row_values: np.ndarray) -> np.ndarray:
    """Convert flat (63,) coordinate array → (21, 3) array."""
    return row_values.reshape(-1, 3)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _proj(pts, cfg):
    """Return (horiz, vert) arrays for a single view config."""
    i, j, ni, nj = cfg
    si = -1 if ni else 1
    sj = -1 if nj else 1
    return si * pts[:, i], sj * pts[:, j]


def draw_hand_on_axes(axes_list, pts, lw=1.8, ms=25):
    """
    Draw the hand skeleton onto three pre-created matplotlib Axes
    (one per view). Returns all artist objects so they can be updated
    or cleared.
    """
    artists = []
    for ax, (_, cfg) in zip(axes_list, VIEWS.items()):
        base = 1
        for f, prefix in enumerate(FINGER_PREFIXES.values()):
            color = FINGER_COLORS[f]
            finger_pts = pts[base:base + 4]         # (4, 3)

            if f == 0:                               # thumb — skip KNU1_B
                vis = finger_pts[1:]                 # KNU1_A…KNU3_A
                hx_p, vy_p = _proj(pts[[0]], cfg)
                hx_v, vy_v = _proj(vis, cfg)
                ln, = ax.plot(
                    [hx_p[0], hx_v[0]], [vy_p[0], vy_v[0]],
                    color=color, lw=lw * 0.6, ls="--", alpha=0.7)
                artists.append(ln)
                ln, = ax.plot(hx_v, vy_v, color=color, lw=lw, alpha=0.9)
                artists.append(ln)
                sc = ax.scatter(hx_v, vy_v, c=color, s=ms, zorder=4)
                artists.append(sc)
            else:
                hx_p, vy_p = _proj(pts[[0]], cfg)
                hx_f, vy_f = _proj(finger_pts, cfg)
                ln, = ax.plot(
                    [hx_p[0], hx_f[0]], [vy_p[0], vy_f[0]],
                    color=color, lw=lw * 0.6, ls="--", alpha=0.7)
                artists.append(ln)
                ln, = ax.plot(hx_f, vy_f, color=color, lw=lw, alpha=0.9)
                artists.append(ln)
                sc = ax.scatter(hx_f, vy_f, c=color, s=ms, zorder=4)
                artists.append(sc)

            base += 4

        hx_p, vy_p = _proj(pts[[0]], cfg)
        sc = ax.scatter(hx_p, vy_p, c=PALM_COLOR, s=ms * 1.8, zorder=5)
        artists.append(sc)

    return artists


def compute_axis_limits(D_vid: np.ndarray, margin: float = 0.05):
    """
    Compute consistent per-view axis limits across all frames of a video
    so the hand doesn't jump around as limits auto-rescale.
    """
    pts_all = D_vid.reshape(-1, 3)               # (N*21, 3)
    limits  = {}
    for view_name, cfg in VIEWS.items():
        i, j, ni, nj = cfg
        si = -1 if ni else 1
        sj = -1 if nj else 1
        hvals = si * pts_all[:, i]
        vvals = sj * pts_all[:, j]
        hrange = hvals.max() - hvals.min()
        vrange = vvals.max() - vvals.min()
        hpad   = max(hrange * margin, 1e-3)
        vpad   = max(vrange * margin, 1e-3)
        limits[view_name] = (
            hvals.min() - hpad, hvals.max() + hpad,
            vvals.min() - vpad, vvals.max() + vpad,
        )
    return limits


# ---------------------------------------------------------------------------
# Per-video renderer
# ---------------------------------------------------------------------------

def render_video(video_id: str,
                 frames: pd.DataFrame,
                 coord_cols: list,
                 out_path: str,
                 fps: int = 30,
                 dpi: int = 120):
    """Render one MP4 for a single video_id."""

    # Sort frames by frame number
    frames = frames.copy()
    frames["_fn"] = frames["frame_id"].str.extract(r"(\d+)").astype(int)
    frames = frames.sort_values("_fn").reset_index(drop=True)

    D_vid = frames[coord_cols].to_numpy(dtype=float).reshape(len(frames), 21, 3)
    n_frames = len(frames)

    limits = compute_axis_limits(D_vid)

    # ── Figure setup ─────────────────────────────────────────────────────────
    n_views = len(VIEWS)
    fig = plt.figure(figsize=(4.5 * n_views, 4.5), facecolor=BG_COLOR)

    gs = gridspec.GridSpec(
        1, n_views,
        figure=fig,
        wspace=0.06,
        left=0.01, right=0.99, top=0.88, bottom=0.02,
    )

    view_axes = []
    for v, (view_name, cfg) in enumerate(VIEWS.items()):
        ax = fig.add_subplot(gs[0, v])
        ax.set_facecolor(BG_COLOR)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(view_name, color="white", fontsize=9, pad=4)
        lims = limits[view_name]
        ax.set_xlim(lims[0], lims[1])
        ax.set_ylim(lims[2], lims[3])
        view_axes.append(ax)

    # Title and frame counter text
    title_txt = fig.text(
        0.5, 0.97,
        f"{video_id}   frame 0 / {n_frames - 1}",
        color="white", fontsize=11, ha="center", va="top",
    )

    # Finger legend
    legend_items = list(FINGER_PREFIXES.keys()) + ["Palm"]
    legend_colors = FINGER_COLORS + [PALM_COLOR]
    for fi, (name, color) in enumerate(zip(legend_items, legend_colors)):
        fig.text(
            0.01 + fi * 0.13, 0.005, name,
            color=color, fontsize=7.5, ha="left", va="bottom",
        )

    # ── Writer ───────────────────────────────────────────────────────────────
    writer = FFMpegWriter(fps=fps, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p"])

    with writer.saving(fig, out_path, dpi=dpi):
        for fi in range(n_frames):
            pts = D_vid[fi]                         # (21, 3)

            # Clear only the data artists, keep axes/titles
            for ax in view_axes:
                for art in list(ax.lines) + list(ax.collections):
                    art.remove()

            draw_hand_on_axes(view_axes, pts)

            frame_num = int(frames["_fn"].iloc[fi])
            title_txt.set_text(
                f"{video_id}   frame {frame_num} / {int(frames['_fn'].iloc[-1])}"
            )

            writer.grab_frame()

            if fi % 100 == 0:
                pct = 100 * fi / n_frames
                print(f"    [{video_id}] {fi}/{n_frames} frames ({pct:.0f}%)",
                      end="\r", flush=True)

    plt.close(fig)
    print(f"    [{video_id}] done → {out_path}            ")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate per-video MP4 animations of hand joint motion.")
    parser.add_argument("--input",  default="raw data/normalised_hand_data.csv",
                        help="Normalised hand data CSV")
    parser.add_argument("--outdir", default="videos",
                        help="Output directory for MP4 files (default: videos/)")
    parser.add_argument("--fps",    type=int,   default=30,
                        help="Frames per second (default: 30)")
    parser.add_argument("--dpi",    type=int,   default=120,
                        help="Output resolution in DPI (default: 120)")
    parser.add_argument("--video",  type=str,   default=None,
                        help="Render only this video_id (e.g. data_17). "
                             "Omit to render all videos.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"Loading {args.input} …", end=" ", flush=True)
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        sys.exit(f"\nERROR: File not found → {args.input}")

    # Drop exact duplicate rows (same video_id + frame_id)
    before = len(df)
    df = df.drop_duplicates(subset=["video_id", "frame_id"])
    print(f"{len(df):,} unique rows ({before - len(df):,} duplicates dropped)")

    ccols = coord_columns()

    # Select videos to render
    all_videos = sorted(df["video_id"].unique())
    if args.video:
        if args.video not in all_videos:
            sys.exit(f"ERROR: video_id '{args.video}' not found. "
                     f"Available: {all_videos}")
        videos = [args.video]
    else:
        videos = all_videos

    print(f"Rendering {len(videos)} video(s) at {args.fps} fps, {args.dpi} dpi\n")

    for vi, video_id in enumerate(videos):
        frames  = df[df["video_id"] == video_id]
        out_path = os.path.join(args.outdir, f"{video_id}.mp4")
        n = len(frames)
        print(f"[{vi+1}/{len(videos)}] {video_id}  ({n} frames) → {out_path}")
        render_video(
            video_id=video_id,
            frames=frames,
            coord_cols=ccols,
            out_path=out_path,
            fps=args.fps,
            dpi=args.dpi,
        )

    print(f"\nAll done. Videos saved to: {args.outdir}/")


if __name__ == "__main__":
    main()