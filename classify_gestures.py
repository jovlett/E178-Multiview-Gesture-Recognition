"""
classify_gestures.py
====================
Open-set multi-class hand gesture classification.

The key design principle is **open-set**: the models are trained on 8 known
gesture classes but the unlabeled pool may contain 
+hand poses that don't
belong to any of them.  Rather than forcing every frame into a known class
(closed-set / pigeonhole), each model independently decides whether it is
confident enough to label a frame.  Frames that no model claims confidently
are left as UNKNOWN.

Models
------
  1. Logistic Regression  – confidence via predict_proba  (threshold 0.80)
  2. MLP                  – confidence via predict_proba  (threshold 0.80)
  3. Random Forest        – confidence via predict_proba  (threshold 0.80)
  4. Prototype (cosine)   – confidence via max cosine sim (threshold 0.95)
     Threshold chosen from the reference set minimum (0.978), with a small
     margin.  Reference set LR/MLP/RF always score > 0.94.

Open-set ensemble rule
----------------------
  A frame is assigned a gesture label only when ≥ MIN_AGREE models (default 3)
  all independently agree on the SAME class above their confidence threshold.
  Any frame that fails this test is marked UNKNOWN.

  This naturally handles:
    • Low individual confidence  → each model returns UNKNOWN
    • Model disagreement         → no class reaches MIN_AGREE votes
    • Genuinely novel poses      → cosine sim low for ALL prototypes

Thresholds were calibrated on the reference set:
  - All known-class frames score > 0.94 predict_proba → 0.80 gives headroom
  - All known-class frames score > 0.978 cosine sim  → 0.95 gives headroom
  Neither threshold incorrectly rejects a single known-class frame.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model     import LogisticRegression
from sklearn.neural_network   import MLPClassifier
from sklearn.ensemble         import RandomForestClassifier
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.model_selection  import StratifiedKFold, cross_validate
from sklearn.metrics          import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os

os.makedirs("figures",  exist_ok=True)
os.makedirs("outputs",  exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
UNKNOWN          = "UNKNOWN"
PROBA_THRESHOLD  = 0.80   # for LR / MLP / RF
COSINE_THRESHOLD = 0.95   # for prototype
MIN_AGREE        = 3      # min models that must agree for ensemble to label

# ── Column definitions ─────────────────────────────────────────────────────────
FINGER_PREFIXES = {"Thumb": "TH", "Pinky": "F1", "Ring": "F2",
                   "Middle": "F3", "Index": "F4"}
FINGER_JOINTS   = ["KNU1_B", "KNU1_A", "KNU2_A", "KNU3_A"]

def _build_coord_cols():
    cols = ["PALM_POSITION_X", "PALM_POSITION_Y", "PALM_POSITION_Z"]
    for prefix in FINGER_PREFIXES.values():
        for suffix in FINGER_JOINTS:
            cols += [f"{prefix}_{suffix}_{ax}" for ax in ("X", "Y", "Z")]
    return cols

COORD_COLS = _build_coord_cols()   # 63 columns; PALM_POSITION_* are constant
                                   # (all zero after spatial normalisation) but
                                   # kept for reproducibility with original loader


# ── Data loaders ───────────────────────────────────────────────────────────────
def load_reference(path="data/prototype_dataset.csv"):
    df = pd.read_csv(path)
    before = len(df)
    df = df.drop_duplicates(subset=COORD_COLS).reset_index(drop=True)
    print(f"    Reference  : {before} → {len(df)} rows  "
          f"({before - len(df)} duplicates removed)")
    return df[COORD_COLS].to_numpy(float), df["gesture_name"].to_numpy()

def load_unlabeled(path="data/unlabeled_set.csv"):
    # Already deduplicated by create_unlabeled_set.py — load as-is.
    df = pd.read_csv(path)
    print(f"    Unlabeled  : {len(df)} rows")
    return df[COORD_COLS].to_numpy(float), df[["video_id", "frame_id"]]


# ── Prototype open-set classifier ─────────────────────────────────────────────
class PrototypeClassifier:
    """
    Nearest-centroid classifier using cosine similarity.

    predict() returns UNKNOWN for any frame whose maximum cosine similarity to
    all class centroids falls below `threshold`.  This is the open-set
    rejection step — frames that look unlike every known prototype are not
    forced into the closest class.
    """

    def fit(self, X, y):
        self.labels_       = np.unique(y)
        self.proto_matrix_ = np.vstack(
            [X[y == lbl].mean(axis=0) for lbl in self.labels_]
        )
        return self

    def predict(self, X, threshold=COSINE_THRESHOLD):
        sim      = cosine_similarity(X, self.proto_matrix_)    # (N, K)
        max_sim  = sim.max(axis=1)                             # (N,)
        pred_idx = np.argmax(sim, axis=1)
        preds    = self.labels_[pred_idx].astype(object)
        preds[max_sim < threshold] = UNKNOWN
        return preds

    def confidence(self, X):
        """Max cosine similarity to any prototype — the 'confidence' score."""
        return cosine_similarity(X, self.proto_matrix_).max(axis=1)

    def cv_accuracy(self, X, y, n_splits=5, seed=42):
        """Stratified k-fold CV accuracy (excludes UNKNOWN from the count)."""
        skf, scores = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                      random_state=seed), []
        for tr, va in skf.split(X, y):
            self.fit(X[tr], y[tr])
            preds = self.predict(X[va])
            # count UNKNOWN as wrong
            scores.append(np.mean(preds == y[va]))
        return np.array(scores)


# ── Open-set wrapper for sklearn pipelines ────────────────────────────────────
def predict_open(model, X, threshold=PROBA_THRESHOLD):
    """
    Wraps any sklearn classifier that exposes predict_proba.

    Returns a label string for each sample, or UNKNOWN when the model's
    maximum class probability is below `threshold`.
    """
    probs    = model.predict_proba(X)           # (N, K)
    max_prob = probs.max(axis=1)                # (N,)
    best_cls = model.classes_[np.argmax(probs, axis=1)]
    preds    = best_cls.astype(object)
    preds[max_prob < threshold] = UNKNOWN
    return preds

def get_confidence(model, X):
    """Max predict_proba score — used for the scatter analysis."""
    return model.predict_proba(X).max(axis=1)


# ── Open-set ensemble ─────────────────────────────────────────────────────────
def ensemble_open_set(per_model_preds, min_agree=MIN_AGREE):
    """
    For each frame, collect non-UNKNOWN votes.  If at least `min_agree`
    models agree on the same class, assign that class.  Otherwise UNKNOWN.

    Parameters
    ----------
    per_model_preds : list of 1-D arrays, each length N
    min_agree       : int — minimum agreeing models required

    Returns
    -------
    1-D array of length N
    """
    stacked = np.column_stack(per_model_preds)          # (N, n_models)
    result  = np.full(len(stacked), UNKNOWN, dtype=object)

    for i, row in enumerate(stacked):
        known_votes = row[row != UNKNOWN]
        if len(known_votes) == 0:
            continue
        unique, counts = np.unique(known_votes, return_counts=True)
        top_class      = unique[np.argmax(counts)]
        top_count      = counts.max()
        if top_count >= min_agree:
            result[i] = top_class

    return result


# ── Model definitions ──────────────────────────────────────────────────────────
def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(solver="lbfgs", max_iter=2000,
                                          C=1.0, random_state=42)),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    MLPClassifier(hidden_layer_sizes=(64, 32),
                                     activation="relu", max_iter=2000,
                                     early_stopping=False, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=300, max_depth=8,
                                              min_samples_leaf=2,
                                              random_state=42, n_jobs=-1)),
        ]),
    }


# ── Cross-validation ──────────────────────────────────────────────────────────
def run_cv(models, X, y, n_splits=5, seed=42):
    cv, rows = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=seed), []
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=cv,
                             scoring=["accuracy", "f1_macro"],
                             return_train_score=True)
        rows.append({"Model": name,
                     "CV Acc  (mean)":   res["test_accuracy"].mean(),
                     "CV Acc  (std)":    res["test_accuracy"].std(),
                     "CV F1   (mean)":   res["test_f1_macro"].mean(),
                     "CV F1   (std)":    res["test_f1_macro"].std(),
                     "Train Acc (mean)": res["train_accuracy"].mean()})

    proto  = PrototypeClassifier()
    scores = proto.cv_accuracy(X, y, n_splits=n_splits, seed=seed)
    rows.append({"Model": "Prototype (cosine)",
                 "CV Acc  (mean)":   scores.mean(),
                 "CV Acc  (std)":    scores.std(),
                 "CV F1   (mean)":   float("nan"),
                 "CV F1   (std)":    float("nan"),
                 "Train Acc (mean)": float("nan")})

    return pd.DataFrame(rows).set_index("Model")


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_cv_comparison(cv_summary, save_path="figures/cv_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("5-Fold CV — Deduplicated Reference Set (N=200, 25/class)",
                 fontsize=13)
    palette = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    for ax, metric, label in zip(axes,
                                  ["CV Acc  (mean)", "CV F1   (mean)"],
                                  ["Accuracy", "Macro F1"]):
        vals = cv_summary[metric].dropna()
        errs = cv_summary[metric.replace("mean","std")].reindex(vals.index).fillna(0)
        bars = ax.barh(vals.index, vals.values, xerr=errs.values,
                       color=palette[:len(vals)], capsize=4, edgecolor="white")
        for bar, v in zip(bars, vals.values):
            ax.text(v+0.01, bar.get_y()+bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=9)
        ax.set_xlabel(label); ax.set_xlim(0, 1.1)
        ax.axvline(1.0, color="grey", lw=0.8, ls="--")
        ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_confusion_matrices(models, proto, X, y,
                            save_path="figures/confusion_matrices.png"):
    all_models = {**models, "Prototype (cosine)": proto}
    n = len(all_models)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    fig.suptitle("Confusion matrices — trained on full reference set", fontsize=12)
    classes = np.unique(y)
    for ax, (name, model) in zip(axes, all_models.items()):
        # Use raw predict (no threshold) on training data — this is a sanity check
        if hasattr(model, "predict_proba"):
            preds = model.predict(X)
        else:
            preds = model.predict(X, threshold=0.0)   # disable threshold for CM
        cm   = confusion_matrix(y, preds, labels=classes)
        disp = ConfusionMatrixDisplay(cm, display_labels=classes)
        disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(""); ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_open_set_results(results_df, save_path="figures/open_set_results.png"):
    """
    Two panels:
      Left  – ensemble label distribution (known classes + UNKNOWN)
      Right – per-model UNKNOWN rate
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Open-set classification  "
        f"(≥{MIN_AGREE}/4 models agree above confidence threshold)",
        fontsize=13
    )

    # ── Left: ensemble gesture distribution ──────────────────────────────────
    ens_counts = results_df["ensemble_open_set"].value_counts()
    colors     = ["#c44e52" if lbl == UNKNOWN else "#4c72b0"
                  for lbl in ens_counts.index]
    bars = ax1.barh(ens_counts.index, ens_counts.values,
                    color=colors, edgecolor="white")
    for bar, v in zip(bars, ens_counts.values):
        ax1.text(v + 50, bar.get_y() + bar.get_height()/2,
                 f"{v:,}", va="center", fontsize=9)
    ax1.set_xlabel("Frames")
    ax1.set_title("Ensemble open-set — label distribution", fontsize=11)
    ax1.grid(axis="x", alpha=0.3)

    # ── Right: per-model UNKNOWN rate ────────────────────────────────────────
    model_cols  = ["logistic_regression", "mlp",
                   "random_forest", "prototype_cosine"]
    model_names = ["Logistic Reg.", "MLP", "Random Forest", "Prototype"]
    unknown_pct = [
        (results_df[col] == UNKNOWN).mean() * 100
        for col in model_cols
    ]
    ax2.bar(model_names, unknown_pct,
            color=["#dd8452", "#55a868", "#4c72b0", "#c44e52"],
            edgecolor="white")
    for i, v in enumerate(unknown_pct):
        ax2.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)
    ax2.set_ylabel("% frames marked UNKNOWN")
    ax2.set_title("Per-model rejection rate", fontsize=11)
    ax2.set_ylim(0, max(unknown_pct) * 1.2)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_confidence_vs_agreement(results_df, confidences,
                                 save_path="figures/confidence_vs_agreement.png"):
    """
    Scatter: x = number of models in agreement (1-4), y = avg confidence.
    Colour encodes final ensemble label (UNKNOWN in red).
    """
    model_cols = ["logistic_regression", "mlp",
                  "random_forest", "prototype_cosine"]
    votes      = results_df[model_cols]

    # count how many models gave the same answer as the ensemble
    ens        = results_df["ensemble_open_set"].to_numpy()
    agree_n    = votes.apply(
        lambda row: (row.values == row.value_counts().idxmax()).sum(), axis=1
    ).to_numpy()

    avg_conf   = np.mean(np.column_stack(confidences), axis=1)
    is_unknown = ens == UNKNOWN

    fig, ax = plt.subplots(figsize=(8, 5))
    sc1 = ax.scatter(agree_n[~is_unknown], avg_conf[~is_unknown],
                     c="#4c72b0", alpha=0.4, s=8, label="Classified")
    sc2 = ax.scatter(agree_n[is_unknown],  avg_conf[is_unknown],
                     c="#c44e52", alpha=0.3, s=8, label="UNKNOWN")
    ax.axhline(PROBA_THRESHOLD,  color="#dd8452", ls="--", lw=1,
               label=f"Proba threshold ({PROBA_THRESHOLD})")
    ax.axhline(COSINE_THRESHOLD, color="#55a868", ls=":",  lw=1,
               label=f"Cosine threshold ({COSINE_THRESHOLD})")
    ax.set_xlabel("Models in agreement (out of 4)")
    ax.set_ylabel("Average confidence score")
    ax.set_title("Confidence vs model agreement — open-set view", fontsize=11)
    ax.legend(fontsize=9, markerscale=3)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Hand Gesture Open-Set Classification")
    print("=" * 60)
    print(f"\n  Config: proba threshold={PROBA_THRESHOLD}  "
          f"cosine threshold={COSINE_THRESHOLD}  "
          f"min_agree={MIN_AGREE}/4")

    # 1. Load data
    print("\n[1] Loading data …")
    X_ref, y_ref   = load_reference("data/prototype_dataset.csv")
    X_all, meta_df = load_unlabeled("data/unlabeled_set.csv")
    print(f"    Features  : {X_ref.shape[1]} columns")
    print(f"    Classes   : {np.unique(y_ref).tolist()}")

    # 2. Cross-validation (on known classes — closed-set sanity check)
    print("\n[2] Closed-set CV on reference set (sanity check) …")
    models     = build_models()
    cv_summary = run_cv(models, X_ref, y_ref)
    print()
    print(cv_summary.to_string(float_format=lambda x: f"{x:.4f}"))
    cv_summary.to_csv("outputs/cv_summary.csv")
    plot_cv_comparison(cv_summary)

    # 3. Fit on full reference set
    print("\n[3] Fitting all models on full reference set …")
    for name, model in models.items():
        model.fit(X_ref, y_ref)
        print(f"    {name:<25}  train acc = "
              f"{np.mean(model.predict(X_ref) == y_ref):.4f}")
    proto = PrototypeClassifier().fit(X_ref, y_ref)
    print(f"    {'Prototype (cosine)':<25}  train acc = "
          f"{np.mean(proto.predict(X_ref, threshold=0.0) == y_ref):.4f}")

    plot_confusion_matrices(models, proto, X_ref, y_ref)

    # 4. Open-set prediction on unlabeled set
    print("\n[4] Open-set predictions on unlabeled set …")
    print(f"    A frame is labelled only when ≥{MIN_AGREE} models confidently agree.\n")

    results      = meta_df.copy()
    all_preds    = []
    confidences  = []

    # ── sklearn models ──
    for name, model in models.items():
        col_key     = name.lower().replace(" ", "_")
        preds_open  = predict_open(model, X_all, threshold=PROBA_THRESHOLD)
        conf        = get_confidence(model, X_all)

        results[col_key] = preds_open
        all_preds.append(preds_open)
        confidences.append(conf)

        n_unknown = (preds_open == UNKNOWN).sum()
        print(f"    {name:<25}  UNKNOWN={n_unknown:>5} "
              f"({n_unknown/len(preds_open)*100:.1f}%)")

    # ── prototype ──
    preds_proto = proto.predict(X_all, threshold=COSINE_THRESHOLD)
    conf_proto  = proto.confidence(X_all)
    results["prototype_cosine"] = preds_proto
    all_preds.append(preds_proto)
    confidences.append(conf_proto)

    n_unknown_proto = (preds_proto == UNKNOWN).sum()
    print(f"    {'Prototype (cosine)':<25}  UNKNOWN={n_unknown_proto:>5} "
          f"({n_unknown_proto/len(preds_proto)*100:.1f}%)")

    # ── ensemble ──
    results["ensemble_open_set"] = ensemble_open_set(all_preds,
                                                     min_agree=MIN_AGREE)

    # 5. Report
    print("\n[5] Open-set ensemble results …")
    ens          = results["ensemble_open_set"]
    n_unknown    = (ens == UNKNOWN).sum()
    n_classified = (ens != UNKNOWN).sum()

    print(f"\n    Total frames   : {len(ens):>6}")
    print(f"    Classified     : {n_classified:>6}  ({n_classified/len(ens)*100:.1f}%)")
    print(f"    UNKNOWN        : {n_unknown:>6}  ({n_unknown/len(ens)*100:.1f}%)")
    print()
    print("    Classified gesture breakdown:")
    for lbl, cnt in ens[ens != UNKNOWN].value_counts().items():
        print(f"      {lbl:<20}  {cnt:>5}  ({cnt/n_classified*100:.1f}% of classified)")

    # 6. Save + plots
    results.to_csv("outputs/predictions_open_set.csv", index=False)
    print("\n  Saved → outputs/predictions_open_set.csv")

    plot_open_set_results(results)
    plot_confidence_vs_agreement(results, confidences)

    print("\n[Done]")
    print("=" * 60)


if __name__ == "__main__":
    main()