import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

FIG_DIR = "figures"
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR, exist_ok=True)

FINGER_PREFIXES = {
    "Thumb":   "TH",
    "Pinky":   "F1",
    "Ring":     "F2",
    "Middle":   "F3",
    "Index":   "F4",
}

FINGER_JOINTS = ["KNU1_B", "KNU1_A", "KNU2_A", "KNU3_A"]

FINGER_COLORS = [
    "#e63946",  # Thumb
    "#2a9d8f",  # Index
    "#e9c46a",  # Middle
    "#f4a261",  # Ring
    "#457b9d",  # Pinky
]

def col(prefix, suffix, axis):
    return f"{prefix}_{suffix}_{axis}"

def load_data():
    df = pd.read_csv("data/normalised_hand_data.csv")

    coord_cols = [
        "PALM_POSITION_X",
        "PALM_POSITION_Y",
        "PALM_POSITION_Z",
    ]

    for finger, prefix in FINGER_PREFIXES.items():
        for suffix in FINGER_JOINTS:
            coord_cols += [
                col(prefix, suffix, "X"),
                col(prefix, suffix, "Y"),
                col(prefix, suffix, "Z"),
            ]

    D = df[coord_cols].to_numpy(dtype=float)
    return D

D = load_data()
N = len(D)
P = np.size(D[0])

## Redefine the features ##
def row_to_points(row):
    return row.reshape(-1, 3)  # (21, 3)
def center_hand(points):
    palm = points[0]
    return points - palm
def normalize_scale(points):
    scale = np.linalg.norm(points)
    return points / scale if scale > 0 else points
def radial_distances(points):
    return np.linalg.norm(points, axis=1)  # (21,)
# palm = index 0
# then fingers in order, 4 joints each

def get_fingertip_indices():
    tips = []
    base = 1
    for i in range(5):  # 5 fingers
        tips.append(base + 3)  # last joint of each finger
        base += 4
    return tips

from itertools import combinations

def fingertip_distances(points):
    tips = get_fingertip_indices()
    dists = []
    for i, j in combinations(tips, 2):
        d = np.linalg.norm(points[i] - points[j])
        dists.append(d)
    return np.array(dists)  # (10,)

def fingertip_angles(points):
    tips = get_fingertip_indices()
    angles = []

    for i, j in combinations(tips, 2):
        v1 = points[i]
        v2 = points[j]

        cos_theta = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        )
        angles.append(np.arccos(np.clip(cos_theta, -1, 1)))

    return np.array(angles)  # (10,)

def finger_lengths(points):
    lengths = []
    base = 1
    for i in range(5):
        finger_joints = points[base:base+4]
        length = np.sum(np.linalg.norm(np.diff(finger_joints, axis=0), axis=1))
        lengths.append(length)
        base += 4
    return np.array(lengths)  # (5,)

def extract_features_from_row(row):
    points = row_to_points(row)
    points = center_hand(points)
    points = normalize_scale(points)

    radial = radial_distances(points)          # 21
    tip_dists = fingertip_distances(points)    # 10
    tip_angles = fingertip_angles(points)      # 10
    lengths = finger_lengths(points)           # 5

    return np.concatenate([radial, tip_dists, tip_angles, lengths])

def build_feature_matrix(D):
    return np.array([extract_features_from_row(row) for row in D])
X = build_feature_matrix(D)

# optional normalization feature matrix
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

# ## Finding Kideal using the elbow method

# costs = []
# K_range = range(1, 10)

# for K in K_range:
#     kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
#     kmeans.fit(X)
#     costs.append(kmeans.inertia_)  # same as your cost

# plt.figure(figsize=(6,4))
# plt.plot(K_range, costs, marker='o')
# plt.xlabel("K")
# plt.ylabel("Cost (inertia)")
# plt.title("Elbow Method")
# plt.grid()
# plt.show()

# # 'silhouette_score' can also be used to evaluate clustering quality for different K values, but it requires K >= 2.
# from sklearn.metrics import silhouette_score

# scores = []

# for K in range(2, 10):  # silhouette needs K >= 2
#     kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
#     labels = kmeans.fit_predict(X)
#     score = silhouette_score(X, labels)
#     scores.append(score)

# plt.figure(figsize=(6,4))
# plt.plot(range(2, 10), scores, marker='o')
# plt.xlabel("K")
# plt.ylabel("Silhouette Score")
# plt.title("Silhouette Method")
# plt.grid()
# plt.show()


Kideal = 5
kmeans = KMeans(n_clusters=Kideal, n_init=10, random_state=0)
labels = kmeans.fit_predict(X)


## K custers visualization ##
# # PCA projection (your new “main plot”)

# pca = PCA(n_components=2)
# X_2d = pca.fit_transform(X)

# plt.figure(figsize=(7,6))
# plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)

# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("Gestures in feature space (PCA)")
# plt.grid()
# plt.show()

# centroids_2d = pca.transform(kmeans.cluster_centers_)

# plt.figure(figsize=(7,6))
# plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
# plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='black', s=150, marker='x')

# plt.title("Clusters + centroids, PCA space K={}".format(Kideal))
# plt.savefig(os.path.join(FIG_DIR, "kmeans_visual_feature.png"), dpi=300, bbox_inches='tight')


# ── Plotting utilities ─────────────────────────────────────────────────────

FINGER_COLORS = ["#e63946", "#2a9d8f", "#e9c46a", "#f4a261", "#457b9d"]

def plot_hand_2d(points, ax, alpha=0.35, lw=1.2):
    base = 1
    for f in range(5):
        color = FINGER_COLORS[f]
        finger_pts = points[base : base + 4]
        ax.plot(
            [points[0, 0], finger_pts[0, 0]],
            [points[0, 1], finger_pts[0, 1]],
            color=color, alpha=alpha * 0.6, lw=lw, linestyle="--",
        )
        ax.plot(finger_pts[:, 0], finger_pts[:, 1],
                color=color, alpha=alpha, lw=lw)
        ax.scatter(finger_pts[:, 0], finger_pts[:, 1],
                   c=color, s=6, alpha=alpha, zorder=3)
        base += 4

    # palm
    ax.scatter(points[0, 0], points[0, 1], c="tomato", s=18, zorder=4)


def plot_hand_3d(points, ax, alpha=0.35, lw=1.2):
    base = 1
    for f in range(5):
        color = FINGER_COLORS[f]
        finger_pts = points[base : base + 4]
        ax.plot(
            [points[0, 0], finger_pts[0, 0]],
            [points[0, 1], finger_pts[0, 1]],
            [points[0, 2], finger_pts[0, 2]],
            color=color, alpha=alpha * 0.6, lw=lw, linestyle="--",
        )
        ax.plot(finger_pts[:, 0], finger_pts[:, 1], finger_pts[:, 2],
                color=color, alpha=alpha, lw=lw)
        ax.scatter(finger_pts[:, 0], finger_pts[:, 1], finger_pts[:, 2],
                   c=color, s=6, alpha=alpha, zorder=3)
        base += 4

    ax.scatter([points[0, 0]], [points[0, 1]], [points[0, 2]],
               c="tomato", s=25, zorder=4)


def plot_clusters_as_hands(D, labels, K, samples_per_cluster=30, mode="2d"):
    show_both = (mode == "both")
    n_rows = 2 if show_both else 1

    fig = plt.figure(figsize=(4 * K, 4 * n_rows), facecolor="#0d1117")

    for k in range(K):
        idx = np.where(labels == k)[0]
        n_chosen = min(samples_per_cluster, len(idx))
        chosen = np.random.choice(idx, size=n_chosen, replace=False)

        if mode in ("2d", "both"):
            ax2 = fig.add_subplot(n_rows, K, k + 1)
            ax2.set_facecolor("#0d1117")
            for i in chosen:
                pts = center_hand(row_to_points(D[i]))
                plot_hand_2d(pts, ax2, alpha=0.25)
            ax2.set_title(f"Cluster {k}  (n={len(idx)})", color="white", fontsize=10)
            ax2.set_aspect("equal")
            ax2.axis("off")

        if mode in ("3d", "both"):
            row_offset = 1 if show_both else 0
            ax3 = fig.add_subplot(n_rows, K, row_offset * K + k + 1, projection="3d")
            ax3.set_facecolor("#0d1117")
            ax3.tick_params(colors="gray", labelsize=6)
            for i in chosen:
                pts = center_hand(row_to_points(D[i]))
                plot_hand_3d(pts, ax3, alpha=0.20)
            if mode == "3d":
                ax3.set_title(f"Cluster {k}  (n={len(idx)})", color="white", fontsize=10)

    # legend for finger colours
    for f, (name, color) in enumerate(zip(FINGER_PREFIXES.keys(), FINGER_COLORS)):
        fig.text(0.01 + f * 0.08, 0.01, name, color=color, fontsize=8,
                 ha="left", va="bottom")
    fig.text(0.01 + 5 * 0.08, 0.01, "Palm", color="tomato", fontsize=8)

    plt.suptitle("Hand gesture clusters", color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/clusters_hands_{mode}.png",
                dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()

def mean_hand(D, idx):
    """
    Average hand shape for a cluster.
    Centers each hand before averaging so palm offsets don't blur the shape.
    Returns (21, 3) array.
    """
    centered = np.array([center_hand(row_to_points(D[i])) for i in idx])
    return centered.mean(axis=0)  # (21, 3)


def plot_average_gestures(D, labels, K, mode="2d"):
    show_both = (mode == "both")
    n_rows = 2 if show_both else 1

    fig = plt.figure(figsize=(4 * K, 4 * n_rows), facecolor="#0d1117")

    for k in range(K):
        idx = np.where(labels == k)[0]
        avg_pts = mean_hand(D, idx)  # (21, 3)

        # ── 2D ──────────────────────────────────────────────────────────
        if mode in ("2d", "both"):
            ax2 = fig.add_subplot(n_rows, K, k + 1)
            ax2.set_facecolor("#0d1117")
            plot_hand_2d(avg_pts, ax2, alpha=0.95, lw=2.0)
            ax2.set_title(f"Cluster {k}  (n={len(idx)})", color="white", fontsize=10)
            ax2.set_aspect("equal")
            ax2.axis("off")

        # ── 3D ──────────────────────────────────────────────────────────
        if mode in ("3d", "both"):
            row_offset = 1 if show_both else 0
            ax3 = fig.add_subplot(n_rows, K, row_offset * K + k + 1, projection="3d")
            ax3.set_facecolor("#0d1117")
            ax3.tick_params(colors="gray", labelsize=6)
            plot_hand_3d(avg_pts, ax3, alpha=0.95, lw=2.0)
            if mode == "3d":
                ax3.set_title(f"Cluster {k}  (n={len(idx)})", color="white", fontsize=10)

    # finger colour legend
    for f, (name, color) in enumerate(zip(FINGER_PREFIXES.keys(), FINGER_COLORS)):
        fig.text(0.01 + f * 0.08, 0.01, name, color=color, fontsize=8, ha="left", va="bottom")
    fig.text(0.01 + 5 * 0.08, 0.01, "Palm", color="tomato", fontsize=8)

    plt.suptitle("Average gesture per cluster", color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/clusters_average_{mode}.png",
                dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()

# Axis index pairs for each view
VIEWS = {
    "top": (1, 0),   # YX
    "front":  (1, 2),   # YZ
    "side":   (0, 2),   # XZ
}

def plot_hand_2d_proj(points, ax, axes=(0, 1), alpha=0.95, lw=2.0):
    i, j = axes
    base = 1
    for f in range(5):
        color = FINGER_COLORS[f]
        finger_pts = points[base : base + 4]  # KNU1_B, KNU1_A, KNU2_A, KNU3_A

        if f == 0:
            # Thumb: skip KNU1_B, treat KNU1_A as the base knuckle
            thumb_visible = finger_pts[1:]   # KNU1_A, KNU2_A, KNU3_A — (3, 3)
            # dashed: palm → KNU1_A
            ax.plot(
                [points[0, i], thumb_visible[0, i]],
                [points[0, j], thumb_visible[0, j]],
                color=color, alpha=alpha * 0.6, lw=lw, linestyle="--",
            )
            # solid: KNU1_A → KNU2_A → KNU3_A
            ax.plot(thumb_visible[:, i], thumb_visible[:, j],
                    color=color, alpha=alpha, lw=lw)
            ax.scatter(thumb_visible[:, i], thumb_visible[:, j],
                       c=color, s=20, alpha=alpha, zorder=3)
        else:
            ax.plot(
                [points[0, i], finger_pts[0, i]],
                [points[0, j], finger_pts[0, j]],
                color=color, alpha=alpha * 0.6, lw=lw, linestyle="--",
            )
            ax.plot(finger_pts[:, i], finger_pts[:, j],
                    color=color, alpha=alpha, lw=lw)
            ax.scatter(finger_pts[:, i], finger_pts[:, j],
                       c=color, s=20, alpha=alpha, zorder=3)

        base += 4

    ax.scatter(points[0, i], points[0, j], c="tomato", s=40, zorder=4)

def plot_average_gestures_multiview(D, labels, K):
    """
    For each cluster: 3 columns (front/side/top), one row per cluster.
    """
    view_names = list(VIEWS.keys())
    n_views = len(view_names)

    fig, axes = plt.subplots(
        K, n_views,
        figsize=(3.5 * n_views, 3.5 * K),
        facecolor="#0d1117"
    )

    # ensure 2D indexing even for K=1
    if K == 1:
        axes = axes[np.newaxis, :]

    axis_labels = {
        "front": ("-Z →", "Y ↑"),
        "side":  ("X →", "Y ↑"),
        "top":   ("-Z →", "X ↑"),
    }

    for k in range(K):
        idx = np.where(labels == k)[0]
        avg_pts = mean_hand(D, idx)

        for v, view_name in enumerate(view_names):
            ax = axes[k, v]
            ax.set_facecolor("#0d1117")
            ax.set_aspect("equal")
            ax.axis("off")

            plot_hand_2d_proj(avg_pts, ax, axes=VIEWS[view_name])

            # column header on top row only
            if k == 0:
                xl, yl = axis_labels[view_name]
                ax.set_title(
                    f"{view_name.capitalize()} view\n{xl}  {yl}",
                    color="white", fontsize=10, pad=6
                )

            # cluster label on left column only
            if v == 0:
                ax.set_ylabel(
                    f"Cluster {k}\n(n={len(idx)})",
                    color="white", fontsize=9, rotation=0,
                    labelpad=50, va="center"
                )
                ax.yaxis.set_label_position("left")
                ax.axis("off")  # re-disable after ylabel

    # finger legend
    for f, (name, color) in enumerate(zip(FINGER_PREFIXES.keys(), FINGER_COLORS)):
        fig.text(0.01 + f * 0.09, 0.005, name, color=color,
                 fontsize=8, ha="left", va="bottom")
    fig.text(0.01 + 5 * 0.09, 0.005, "Palm", color="tomato",
             fontsize=8, ha="left", va="bottom")

    plt.suptitle("Average gesture per cluster — multi-view",
                 color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/clusters_average_multiview.png",
                dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()

# Pick whichever view you want:
# plot_clusters_as_hands(D, labels, Kideal, samples_per_cluster=30, mode="2d")
# plot_clusters_as_hands(D, labels, Kideal, samples_per_cluster=30, mode="3d")
# plot_clusters_as_hands(D, labels, Kideal, samples_per_cluster=5, mode="both")  # side-by-side rows

# plot_average_gestures(D, labels, Kideal, mode="2d")
plot_average_gestures(D, labels, Kideal, mode="3d")
# plot_average_gestures(D, labels, Kideal, mode="both")  # 2D top row, 3D bottom row

plot_average_gestures_multiview(D, labels, Kideal)

df = pd.read_csv("data/normalised_hand_data.csv")

df['Cluster_Number'] = labels

df.to_csv("data/normalised_hand_data_cluster", index=False)