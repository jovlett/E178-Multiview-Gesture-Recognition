"""
classify_gestures.py
====================
Multi-class hand gesture classification using:
  1. Logistic Regression
  2. Multi-Layer Perceptron  (MLP)
  3. Random Forest
  4. Prototype Cosine Similarity  (reference-set centroids, no training needed)

Workflow
--------
  • X_ref / y_ref  – deduplicated prototype_dataset.csv  (200 rows → 25 / class)
  • X_all          – deduplicated unlabeled_set.csv       (20 490 unique rows)

NOTE ON DUPLICATES
  The raw data contains every frame exactly twice, and the prototype set
  inherits the same duplication (320 rows → 200 unique).  Running CV on
  the raw prototype rows causes data leakage: identical feature vectors
  split across train/test folds give artificially perfect scores.
  Both datasets are deduplicated before any fitting or evaluation.

Each supervised model is evaluated with stratified 5-fold CV on the reference
set, then retrained on the full deduplicated reference set and used to label
X_all.  Predictions are saved to  predictions_all_models.csv.
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model    import LogisticRegression
from sklearn.neural_network  import MLPClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics         import (classification_report,
                                     ConfusionMatrixDisplay,
                                     confusion_matrix)
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ── Column definitions (unchanged from original loader) ───────────────────────
FINGER_PREFIXES = {
    "Thumb":  "TH",
    "Pinky":  "F1",
    "Ring":   "F2",
    "Middle": "F3",
    "Index":  "F4",
}

FINGER_JOINTS = ["KNU1_B", "KNU1_A", "KNU2_A", "KNU3_A"]

def _col(prefix, suffix, axis):
    return f"{prefix}_{suffix}_{axis}"

def _build_coord_cols():
    cols = ["PALM_POSITION_X", "PALM_POSITION_Y", "PALM_POSITION_Z"]
    for prefix in FINGER_PREFIXES.values():
        for suffix in FINGER_JOINTS:
            cols += [_col(prefix, suffix, ax) for ax in ("X", "Y", "Z")]
    return cols

COORD_COLS = _build_coord_cols()   # 63 feature columns


# ── Data loaders ───────────────────────────────────────────────────────────────
def load_reference(path: str = "data/prototype_dataset.csv"):
    """
    Load the labelled prototype set and deduplicate on feature columns.
    Returns (X_ref, y_ref) with 200 rows (25 per class).
    """
    df = pd.read_csv(path)
    before = len(df)
    df = df.drop_duplicates(subset=COORD_COLS).reset_index(drop=True)
    print(f"    Deduped reference  : {before} → {len(df)} rows "
          f"({before - len(df)} duplicates removed)")
    X = df[COORD_COLS].to_numpy(dtype=float)
    y = df["gesture_name"].to_numpy()
    return X, y


def load_unlabeled(path: str = "data/unlabeled_set.csv"):
    """
    Load the unlabelled set and deduplicate.
    Returns (X_all, meta_df) with unique rows only.
    """
    df = pd.read_csv(path)
    before = len(df)
    df = df.drop_duplicates(subset=COORD_COLS).reset_index(drop=True)
    print(f"    Deduped unlabeled  : {before} → {len(df)} rows "
          f"({before - len(df)} duplicates removed)")
    X   = df[COORD_COLS].to_numpy(dtype=float)
    meta = df[["video_id", "frame_id"]]
    return X, meta


# ── Prototype / cosine-similarity classifier ──────────────────────────────────
class PrototypeClassifier:
    """
    Nearest-centroid classifier using cosine similarity.
    One prototype per class = mean of all reference vectors for that class.
    """

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.labels_       = np.unique(y)
        self.proto_matrix_ = np.vstack(
            [X[y == label].mean(axis=0) for label in self.labels_]
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        sim      = cosine_similarity(X, self.proto_matrix_)   # (N, K)
        pred_idx = np.argmax(sim, axis=1)
        return self.labels_[pred_idx]

    def cv_accuracy(self, X: np.ndarray, y: np.ndarray,
                    n_splits: int = 5, seed: int = 42) -> np.ndarray:
        """Manual stratified k-fold (needed because PrototypeClassifier is not
        a sklearn estimator and cross_validate can't be used directly)."""
        skf    = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                 random_state=seed)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            self.fit(X[train_idx], y[train_idx])
            preds = self.predict(X[val_idx])
            scores.append(np.mean(preds == y[val_idx]))
        return np.array(scores)


# ── Model definitions ──────────────────────────────────────────────────────────
def build_models():
    """
    Three sklearn pipelines.  All use StandardScaler — essential for LR and MLP.
    Hyper-parameters tuned for the small reference set (N=200, 8 classes,
    25 samples per class).
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                solver="lbfgs",
                max_iter=2000,
                C=1.0,
                random_state=42,
            )),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    MLPClassifier(
                hidden_layer_sizes=(64, 32),  # smaller net for N=200
                activation="relu",
                max_iter=2000,
                early_stopping=False,         # sklearn >=1.8 bug w/ string targets in CV
                random_state=42,
            )),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=300,
                max_depth=8,          # cap depth; 25 samples/class makes deep trees overfit
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


# ── Cross-validation ──────────────────────────────────────────────────────────
def run_cv(models: dict, X: np.ndarray, y: np.ndarray,
           n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Stratified k-fold CV on the deduplicated reference set."""
    cv   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []

    for name, model in models.items():
        res = cross_validate(
            model, X, y,
            cv=cv,
            scoring=["accuracy", "f1_macro"],
            return_train_score=True,
        )
        rows.append({
            "Model":            name,
            "CV Acc  (mean)":   res["test_accuracy"].mean(),
            "CV Acc  (std)":    res["test_accuracy"].std(),
            "CV F1   (mean)":   res["test_f1_macro"].mean(),
            "CV F1   (std)":    res["test_f1_macro"].std(),
            "Train Acc (mean)": res["train_accuracy"].mean(),
        })

    proto  = PrototypeClassifier()
    scores = proto.cv_accuracy(X, y, n_splits=n_splits, seed=seed)
    rows.append({
        "Model":            "Prototype (cosine)",
        "CV Acc  (mean)":   scores.mean(),
        "CV Acc  (std)":    scores.std(),
        "CV F1   (mean)":   float("nan"),
        "CV F1   (std)":    float("nan"),
        "Train Acc (mean)": float("nan"),
    })

    return pd.DataFrame(rows).set_index("Model")


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_cv_comparison(cv_summary: pd.DataFrame,
                       save_path: str = "figures/cv_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "5-Fold CV on Deduplicated Reference Set  "
        "(N=200, 25 samples/class)",
        fontsize=13,
    )
    palette = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    for ax, metric, label in zip(
        axes,
        ["CV Acc  (mean)", "CV F1   (mean)"],
        ["Accuracy", "Macro F1"],
    ):
        vals = cv_summary[metric].dropna()
        errs = cv_summary[
            metric.replace("mean", "std")
        ].reindex(vals.index).fillna(0)
        bars = ax.barh(vals.index, vals.values, xerr=errs.values,
                       color=palette[:len(vals)], capsize=4,
                       edgecolor="white")
        for bar, v in zip(bars, vals.values):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=9)
        ax.set_xlabel(label)
        ax.set_xlim(0, 1.1)
        ax.axvline(1.0, color="grey", lw=0.8, ls="--")
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_confusion_matrices(models: dict, proto: PrototypeClassifier,
                            X: np.ndarray, y: np.ndarray,
                            save_path: str = "figuresconfusion_matrices.png"):
    all_models = {**models, "Prototype (cosine)": proto}
    n = len(all_models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle(
        "Confusion Matrices — retrained on full deduplicated reference set",
        fontsize=12,
    )
    classes = np.unique(y)
    for ax, (name, model) in zip(axes, all_models.items()):
        preds = model.predict(X)
        cm    = confusion_matrix(y, preds, labels=classes)
        disp  = ConfusionMatrixDisplay(cm, display_labels=classes)
        disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Hand Gesture Classification")
    print("=" * 60)

    # 1. Load + deduplicate
    print("\n[1] Loading data …")
    X_ref, y_ref   = load_reference("data/prototype_dataset.csv")
    X_all, meta_df = load_unlabeled("data/unlabeled_set.csv")
    print(f"    Features           : {X_ref.shape[1]} columns")
    print(f"    Classes            : {np.unique(y_ref).tolist()}")

    # 2. Cross-validation on deduplicated reference set
    print("\n[2] Running 5-fold stratified CV on deduplicated reference set …")
    models     = build_models()
    cv_summary = run_cv(models, X_ref, y_ref)

    print()
    print(cv_summary.to_string(float_format=lambda x: f"{x:.4f}"))
    cv_summary.to_csv("outputs/cv_summary.csv")
    print("\n  Saved → outputs/cv_summary.csv")
    plot_cv_comparison(cv_summary)

    # 3. Refit on full deduplicated reference set
    print("\n[3] Fitting all models on full deduplicated reference set …")
    for name, model in models.items():
        model.fit(X_ref, y_ref)
        train_acc = np.mean(model.predict(X_ref) == y_ref)
        print(f"    {name:<25}  train acc = {train_acc:.4f}")

    proto = PrototypeClassifier().fit(X_ref, y_ref)
    print(f"    {'Prototype (cosine)':<25}  train acc = "
          f"{np.mean(proto.predict(X_ref) == y_ref):.4f}")

    # 4. Per-class report on reference set (train-set sanity check)
    print("\n[4] Classification reports on reference set (train — sanity check) …")
    for name, model in {**models, "Prototype (cosine)": proto}.items():
        print(f"\n  ── {name} ──")
        print(classification_report(y_ref, model.predict(X_ref),
                                    target_names=np.unique(y_ref)))

    plot_confusion_matrices(models, proto, X_ref, y_ref)

    # 5. Predict on deduplicated unlabeled set
    print("\n[5] Predicting labels for deduplicated unlabeled set …")
    results = meta_df.copy()
    col_names = {}
    for name, model in {**models, "Prototype (cosine)": proto}.items():
        col_key = (name.lower()
                   .replace(" ", "_")
                   .replace("(", "").replace(")", ""))
        col_names[name] = col_key
        results[col_key] = model.predict(X_all)
        vc = pd.Series(results[col_key]).value_counts()
        print(f"\n  {name}:")
        print(vc.to_string())

    # Majority vote across all 4 methods
    pred_cols = list(col_names.values())
    results["ensemble_vote"] = results[pred_cols].apply(
        lambda row: row.value_counts().idxmax(), axis=1
    )

    out_path = "outputs/predictions_all_models.csv"
    results.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")

    print("\n[Done]")
    print("=" * 60)


if __name__ == "__main__":
    main()