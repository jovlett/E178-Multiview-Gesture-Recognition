import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import os
from sklearn.cluster import KMeans

FIG_DIR = "figures"
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR, exist_ok=True)

FINGER_PREFIXES = {
    "Thumb":   "TH",
    "Index":   "F1",
    "Middle":  "F2",
    "Ring":    "F3",
    "Pinky":   "F4",
}

# FINGER_PREFIXES = {
#     "Thumb":   "TH",
#     "Pinky":   "F1",
#     "Ring":  "F2",
#     "Middle":    "F3",
#     "Index":   "F4",
# }

FINGER_JOINTS = ["KNU1_B", "KNU1_A", "KNU2_A", "KNU3_A"]

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

### FUNCTIONS FOR K-MEANS CLUSTERING ###


def initialize_centroids(K, D, random_seed=None):
    if random_seed is not None:
        np.random.seed(seed=random_seed)
    P = D.shape[1]
    C = np.empty((K, P))
    xmin = D.min(axis=0)
    xmax = D.max(axis=0)
    for p in range(P):
        C[:, p] = stats.uniform.rvs(loc=xmin[p], scale=xmax[p] - xmin[p], size=K)
    return C


def assign_samples_to_nearest_centroid(D, C):
    '''
    N = len(D)
    K = len(C)
    gamma = np.zeros((N, K), dtype=bool)
    for i in range(N):
        dist = np.array(np.sqrt(np.sum((C - D[i, :])**2, axis=1)))
        gamma[i, np.argmin(dist)] = True
    return gamma
    '''

    diffs = D[:, np.newaxis, :] - C[np.newaxis, :, :]  # (N, K, P)
    dists = np.sqrt(np.sum(diffs**2, axis=2))           # (N, K)
    gamma = np.zeros((len(D), len(C)), dtype=bool)
    gamma[np.arange(len(D)), np.argmin(dists, axis=1)] = True
    return gamma


def place_centroids(gamma, D):
    K = gamma.shape[1]
    P = D.shape[1]
    C = np.empty((K, P))
    for i in range(K):
        if np.sum(gamma[:, i]) == 0:
            return None
        C[i, :] = np.mean(D[gamma[:, i]], axis=0)
    return C


def run_kmeans(D, C):

    done = False
    first = True

    while not done:

        gamma = assign_samples_to_nearest_centroid(D, C)

        C = place_centroids(gamma, D)

        if C is None:
            return None, None

        if first == False:
            if np.array_equal(gamma, gammaold):
                break

        gammaold = gamma
        first = False

    return C, gamma


def eval_cost(D, C, gamma):
    if C is None:
        return np.nan
    K = len(C)
    cost = 0
    for i in range(K):
        cost += np.sum((D[gamma[:, i]] - C[i, :])**2)
    return cost


def ensemble_run(K, D, ensemblesize, random_seed=None):
    if random_seed is not None:
        np.random.seed(seed=random_seed)

    best_run = {"C": None, "gamma": None, "cost": np.inf}
    '''
    for e in range(ensemblesize):

        C = initialize_centroids(K, D)
        C, gamma = run_kmeans(D, C)

        while C is None:
            C = initialize_centroids(K, D)
            C, gamma = run_kmeans(D, C)

        cost = eval_cost(D, C, gamma)

        if e == 0 or cost < best_run["cost"]:
            best_run["C"] = C
            best_run["gamma"] = gamma
            best_run["cost"] = cost'''
    
    for e in range(ensemblesize):
        print(f"K = {K}, run {e+1}/{ensemblesize}")

        C = initialize_centroids(K, D)
        C, gamma = run_kmeans(D, C)

        retry = 0
        max_retry = 10

        while C is None and retry < max_retry:
            retry += 1
            print(f"  empty cluster, retry {retry}/{max_retry}")
            C = initialize_centroids(K, D)
            C, gamma = run_kmeans(D, C)

        if C is None:
            print("  skipped this run")
            continue

        cost = eval_cost(D, C, gamma)

        if cost < best_run["cost"]:
            best_run["C"] = C
            best_run["gamma"] = gamma
            best_run["cost"] = cost

    if best_run["C"] is None:
        return None

    return best_run


def K_iteration(D, ensemblesize, maxK, random_seed=None):
    if random_seed is not None:
        np.random.seed(seed=random_seed)

    best_cost = np.empty(maxK)
    '''
    for ind in range(maxK):
        best_cost[ind] = ensemble_run(ind + 1, D, ensemblesize, random_seed)["cost"]'''
    
    for ind in range(maxK):
        K = ind + 1
        best_run = ensemble_run(K, D, ensemblesize)

        if best_run is None:
            best_cost[ind] = np.nan
        else:
            best_cost[ind] = best_run["cost"]

    return best_cost

### Function features of the hand ###

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


##### Actual execution ####
X = build_feature_matrix(D)
# optional normalization feature matrix
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)
ensemblesize = 20
maxK = 20
best_cost = K_iteration(X, ensemblesize, maxK, random_seed=452)


fig, ax = plt.subplots(figsize=(8, 5), nrows=2, sharex=True)

ax[0].plot(
    range(1, maxK + 1),
    best_cost,
    linewidth=3,
    marker="o",
    markersize=8,
)
ax[0].grid()
ax[0].set_ylabel("Cost")

ax[1].plot(
    range(2, maxK + 1),
    100 * np.abs(np.diff(best_cost)) / best_cost[1:],
    linewidth=3,
    marker="o",
    markersize=8,
)
ax[1].grid()
ax[1].set_ylabel("Cost improvement")
ax[1].set_xlabel("K")

plt.title(f"K-means clustering Cost vs K (ensemblesize={ensemblesize})")
plt.savefig(os.path.join(FIG_DIR, "kmeans_cost.png"), dpi=300, bbox_inches='tight')
plt.show()

## From the plot, we can choose K=9 as a good balance between cost and simplicity
Kideal = 9

best_run = ensemble_run(Kideal, X, ensemblesize=20, random_seed=5423)

optC = best_run["C"]
optgamma = best_run["gamma"]

labels = np.argmax(optgamma, axis=1)

## Visualize the clusters in 2D using PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
C_2d = pca.transform(optC)

plt.figure(figsize=(7,6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
plt.scatter(C_2d[:, 0], C_2d[:, 1], c='black', marker='x', s=150)

plt.title(f"K-means clustering (K={Kideal})")
plt.savefig(os.path.join(FIG_DIR, "kmeans_visual.png"), dpi=300, bbox_inches='tight')
# plt.show()

for k in range(Kideal):
    indices = np.where(labels == k)[0]
    print(f"\nCluster {k}:")
    print(indices[:5])  # print 5 example rows

print("N =", N)
print("P =", P)
print("Kideal =", Kideal)
print("Final cost =", best_run["cost"])
print("Cluster labels:")
print(labels)


results = {
    "C": optC,
    "gamma": optgamma,
    "labels": labels,
    "cost": best_run["cost"],
    "Kideal": Kideal,
}

with open("data/kmeans_gesture_results.pickle", "wb") as f:
    pickle.dump(results, f)

print(f'kideal = {Kideal}, cost = {best_run["cost"]}')



## Finding Kideal using the elbow method

costs = []
K_range = range(5, 20)

for K in K_range:
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
    kmeans.fit(X)
    costs.append(kmeans.inertia_)  # same as your cost

plt.figure(figsize=(6,4))
plt.plot(K_range, costs, marker='o')
plt.xlabel("K")
plt.ylabel("Cost (inertia)")
plt.title("Elbow Method")
plt.grid()
plt.show()

# 'silhouette_score' can also be used to evaluate clustering quality for different K values, but it requires K >= 2.
from sklearn.metrics import silhouette_score

scores = []

for K in range(2, 20):  # silhouette needs K >= 2
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    scores.append(score)

plt.figure(figsize=(6,4))
plt.plot(range(2, 20), scores, marker='o')
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.grid()
plt.show()


# PCA projection (your new “main plot”)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

plt.figure(figsize=(7,6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Gestures in feature space (PCA)")
plt.grid()
plt.show()

# pca = PCA(n_components=2)
# D_2d = pca.fit_transform(D)

# plt.figure(figsize=(7,6))
# plt.scatter(D_2d[:, 0], D_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)

# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("Gestures in feature space (PCA)")
# plt.grid()
# plt.show()


centroids_2d = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(7,6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='black', s=150, marker='x')

plt.title("Clusters + centroids, PCA space K={}".format(Kideal))
plt.savefig(os.path.join(FIG_DIR, "kmeans_visual_feature.png"), dpi=300, bbox_inches='tight')