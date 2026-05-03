import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

FINGER_PREFIXES = {
    "Thumb":   "TH",
    "Index":   "F1",
    "Middle":  "F2",
    "Ring":    "F3",
    "Pinky":   "F4",
}

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

ensemblesize = 20
maxK = 10
best_cost = K_iteration(D, ensemblesize, maxK, random_seed=452)


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

plt.show()

Kideal = 4

best_run = ensemble_run(Kideal, D, ensemblesize=20, random_seed=5423)

optC = best_run["C"]
optgamma = best_run["gamma"]

labels = np.argmax(optgamma, axis=1)

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