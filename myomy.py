"""
Hand Pose Multi-Class Predictor Pipeline
=========================================
Reference set : 10 pose classes × p samples each (default p=10)
Unlabeled set : N samples to predict
Models        : Logistic Regression, MLP, Random Forest
Output        : Full model comparison + predictions on unlabeled data
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1.  DATA LOADING HELPERS
# ---------------------------------------------------------------------------

def load_reference_set(keypoints: np.ndarray, labels) -> tuple[np.ndarray, np.ndarray]:
    """
    Accepts keypoints of shape (N, J, D) or (N, J*D) and a label array of
    length N, where J = number of joints and D = 2 or 3.

    Returns
    -------
    X : np.ndarray, shape (N, J*D)   — flattened, float32
    y : np.ndarray, shape (N,)       — integer-encoded class labels
    """
    X = _flatten_keypoints(keypoints)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return X, y, le


def load_unlabeled_set(keypoints: np.ndarray) -> np.ndarray:
    """Flatten unlabeled keypoints to (N, J*D)."""
    return _flatten_keypoints(keypoints)


def _flatten_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Flatten (N, J, D) → (N, J*D) if needed; otherwise pass (N, F) through."""
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim == 3:
        # (N, J, D) → (N, J*D)
        return arr.reshape(arr.shape[0], -1)
    elif arr.ndim == 2:
        return arr
    else:
        raise ValueError(f"Expected 2-D or 3-D array, got shape {arr.shape}")


# ---------------------------------------------------------------------------
# 2.  PIPELINE DEFINITIONS
# ---------------------------------------------------------------------------

def build_pipelines() -> dict[str, Pipeline]:
    """
    Returns a dict of named sklearn Pipelines, each with:
      - StandardScaler  (z-score per feature)
      - Classifier
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                multi_class="multinomial",
                solver="lbfgs",
                C=1.0,
                random_state=42,
            )),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            )),
        ]),
        "Random Forest": Pipeline([
            # RF is scale-invariant but we keep scaler for consistency
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=1,
                random_state=42,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# 3.  CROSS-VALIDATED EVALUATION
# ---------------------------------------------------------------------------

SCORING = ["accuracy", "f1_macro", "f1_weighted"]


def evaluate_all(
    pipelines: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> dict:
    """
    Runs StratifiedKFold CV on every pipeline and collects metrics.

    With p=10 samples/class and 10 classes (N=100) a 5-fold CV gives
    8 train / 2 test samples per class — use n_splits=5 (or lower for
    very small p; n_splits=p gives leave-one-out-per-class).

    Returns
    -------
    results : dict  {model_name: {metric: mean±std, ...}}
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}

    print(f"\n{'='*60}")
    print(f"  Cross-Validation Summary  ({n_splits}-fold StratifiedKFold)")
    print(f"{'='*60}")

    for name, pipe in pipelines.items():
        cv_scores = cross_validate(
            pipe, X, y,
            cv=cv,
            scoring=SCORING,
            return_train_score=True,
        )
        results[name] = {
            metric: {
                "mean": cv_scores[f"test_{metric}"].mean(),
                "std":  cv_scores[f"test_{metric}"].std(),
                "train_mean": cv_scores[f"train_{metric}"].mean(),
            }
            for metric in SCORING
        }

        print(f"\n  [{name}]")
        for metric in SCORING:
            m = results[name][metric]
            print(
                f"    {metric:<16}  "
                f"test {m['mean']:.3f} ± {m['std']:.3f}  |  "
                f"train {m['train_mean']:.3f}"
            )

    print(f"\n{'='*60}\n")
    return results


# ---------------------------------------------------------------------------
# 4.  FINAL FIT + PREDICT ON UNLABELED DATA
# ---------------------------------------------------------------------------

def fit_and_predict(
    pipelines: dict,
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    X_unlabeled: np.ndarray,
    label_encoder: LabelEncoder,
) -> dict:
    """
    Fits each pipeline on the full reference set, then predicts the
    unlabeled set.

    Returns
    -------
    predictions : dict  {model_name: {"labels": [...], "proba": ndarray}}
    """
    predictions = {}
    for name, pipe in pipelines.items():
        pipe.fit(X_ref, y_ref)
        pred_int = pipe.predict(X_unlabeled)
        pred_labels = label_encoder.inverse_transform(pred_int)
        proba = pipe.predict_proba(X_unlabeled) if hasattr(pipe["clf"], "predict_proba") else None
        predictions[name] = {"labels": pred_labels, "proba": proba}
    return predictions


# ---------------------------------------------------------------------------
# 5.  VISUALISATION
# ---------------------------------------------------------------------------

def plot_cv_comparison(results: dict, save_path: str = None):
    """Bar chart comparing test accuracy / F1-macro / F1-weighted."""
    model_names = list(results.keys())
    metrics = SCORING
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(model_names):
        means = [results[name][m]["mean"] for m in metrics]
        stds  = [results[name][m]["std"]  for m in metrics]
        bars  = ax.bar(x + i * width, means, width, yerr=stds,
                       label=name, capsize=4, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison — Cross-Validated Scores (mean ± std)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion_matrices(
    pipelines: dict,
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    save_path: str = None,
):
    """
    Fits each model on the full reference set and shows its confusion matrix.
    (For small datasets this is complementary to CV — not a replacement.)
    """
    class_names = label_encoder.classes_.astype(str)
    n = len(pipelines)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, pipe) in zip(axes, pipelines.items()):
        pipe.fit(X, y)
        y_pred = pipe.predict(X)
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
        ax.set_title(name)

    fig.suptitle("Confusion Matrices (trained on full reference set)", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_classification_reports(
    pipelines: dict,
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
):
    """Prints per-class precision / recall / F1 for each model."""
    class_names = label_encoder.classes_.astype(str)
    for name, pipe in pipelines.items():
        pipe.fit(X, y)
        y_pred = pipe.predict(X)
        print(f"\n--- {name} ---")
        print(classification_report(y, y_pred, target_names=class_names))


# ---------------------------------------------------------------------------
# 6.  MAIN  (replace the synthetic data block with your real data)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── Synthetic data for demonstration ────────────────────────────────────
    NUM_CLASSES    = 10
    SAMPLES_CLASS  = 10        # p — your reference samples per pose
    NUM_JOINTS     = 21        # e.g. MediaPipe hand: 21 joints
    DIMS           = 3         # 3-D keypoints (x, y, z)
    NUM_UNLABELED  = 30        # size of your unlabeled prediction set

    rng = np.random.default_rng(0)
    pose_names = [f"pose_{i:02d}" for i in range(NUM_CLASSES)]

    # Reference set: (N, J, D) keypoints
    ref_keypoints = rng.standard_normal(
        (NUM_CLASSES * SAMPLES_CLASS, NUM_JOINTS, DIMS)
    ).astype(np.float32)
    ref_labels = np.repeat(pose_names, SAMPLES_CLASS)

    # Unlabeled set: (M, J, D)
    unlabeled_keypoints = rng.standard_normal(
        (NUM_UNLABELED, NUM_JOINTS, DIMS)
    ).astype(np.float32)
    # ── End synthetic data ───────────────────────────────────────────────────

    # 1. Load + preprocess
    X_ref, y_ref, le = load_reference_set(ref_keypoints, ref_labels)
    X_unlabeled       = load_unlabeled_set(unlabeled_keypoints)

    print(f"Reference set  : {X_ref.shape}  ({NUM_CLASSES} classes × {SAMPLES_CLASS} samples)")
    print(f"Unlabeled set  : {X_unlabeled.shape}")
    print(f"Feature dim    : {X_ref.shape[1]}  ({NUM_JOINTS} joints × {DIMS} dims)")

    # 2. Build pipelines
    pipes = build_pipelines()

    # 3. Cross-validated evaluation
    cv_results = evaluate_all(pipes, X_ref, y_ref, n_splits=5)

    # 4. Visualise comparison
    plot_cv_comparison(cv_results)
    plot_confusion_matrices(pipes, X_ref, y_ref, le)
    print_classification_reports(pipes, X_ref, y_ref, le)

    # 5. Predict unlabeled set with all models
    predictions = fit_and_predict(pipes, X_ref, y_ref, X_unlabeled, le)

    print("\n── Predictions on unlabeled set ──")
    print(f"{'Sample':<8}", *[f"{n:<22}" for n in predictions])
    for i in range(NUM_UNLABELED):
        row = [predictions[n]["labels"][i] for n in predictions]
        print(f"{i:<8}", *[f"{r:<22}" for r in row])