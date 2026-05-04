"""
classify_gestures.py
====================
Open-set multi-class hand gesture classification using three models.

Design principle: OPEN-SET
--------------------------
The models are trained on 8 known gesture classes, but the unlabeled pool
may contain hand poses that don't belong to any of them. Rather than forcing
every frame into a known class, each model independently decides whether it
is confident enough to label a frame. Frames that no model claims confidently
are left as UNKNOWN.

Models
------
  1. Logistic Regression  – confidence via predict_proba  (threshold 0.80)
  2. MLP                  – confidence via predict_proba  (threshold 0.80)
  3. Random Forest        – confidence via predict_proba  (threshold 0.80)

Open-set ensemble rule
----------------------
  A frame is assigned a gesture label only when ALL 3 models independently
  agree on the SAME class AND each exceeds the 0.80 confidence threshold.
  Any frame that fails this test is marked UNKNOWN.

  This naturally handles:
    - Low individual confidence  → model returns UNKNOWN
    - Model disagreement         → classes don't all match
    - Genuinely novel poses      → all models are uncertain

Why three models instead of four?
----------------------------------
  Each of the three sklearn models uses a fundamentally different learning
  strategy (linear boundaries, neural net, decision trees). That diversity
  captures different signals from the data. Requiring unanimous agreement
  across all three is actually more conservative than the original 3-of-4
  rule, meaning every labeled frame is highly trustworthy. The cosine/
  prototype model was omitted as its geometric similarity approach is largely
  redundant when well-calibrated probability scores are already available.

Threshold calibration
---------------------
  On the reference set, all known-class frames score > 0.94 predict_proba.
  The 0.80 threshold gives comfortable headroom without rejecting real gestures.

Output folder
-------------
  All figures, CSVs, and feature summaries are saved to: exp_data_outputs/
"""

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
from sklearn.metrics         import (classification_report, confusion_matrix,
                                     ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import os

# ── Output folder (single location for everything) ────────────────────────────
OUT_DIR = "outputs/final outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
UNKNOWN         = "UNKNOWN"
PROBA_THRESHOLD = 0.80   # minimum confidence for any model to claim a label
MIN_AGREE       = 3      # all three models must agree (unanimous)

# ── Column definitions ─────────────────────────────────────────────────────────
FINGER_PREFIXES = {"Thumb": "TH", "Pinky": "F1", "Ring": "F2",
                   "Middle": "F3", "Index": "F4"}
FINGER_JOINTS   = ["KNU1_B", "KNU1_A", "KNU2_A", "KNU3_A"]

def _build_coord_cols():
    """Build the list of 63 coordinate feature column names."""
    cols = ["PALM_POSITION_X", "PALM_POSITION_Y", "PALM_POSITION_Z"]
    for prefix in FINGER_PREFIXES.values():
        for suffix in FINGER_JOINTS:
            cols += [f"{prefix}_{suffix}_{ax}" for ax in ("X", "Y", "Z")]
    return cols

COORD_COLS = _build_coord_cols()   # 63 columns total


# ── Data loaders ───────────────────────────────────────────────────────────────
def load_reference(path="data/prototype_dataset.csv"):
    """
    Load the labeled reference set and remove duplicate rows.
    Returns:
        X (np.ndarray): shape (N, 63) — joint coordinate features
        y (np.ndarray): shape (N,)    — gesture name labels
    """
    df = pd.read_csv(path)
    before = len(df)
    df = df.drop_duplicates(subset=COORD_COLS).reset_index(drop=True)
    print(f"    Reference  : {before} -> {len(df)} rows  "
          f"({before - len(df)} duplicates removed)")
    return df[COORD_COLS].to_numpy(float), df["gesture_name"].to_numpy()


def load_unlabeled(path="data/unlabeled_set.csv"):
    """
    Load the unlabeled set for open-set classification.
    Returns:
        X    (np.ndarray):    shape (N, 63) — joint coordinate features
        meta (pd.DataFrame):  video_id and frame_id columns for tracking
    """
    df = pd.read_csv(path)
    print(f"    Unlabeled  : {len(df)} rows")
    return df[COORD_COLS].to_numpy(float), df[["video_id", "frame_id"]]


# ── Open-set prediction wrapper ───────────────────────────────────────────────
def predict_open(model, X, threshold=PROBA_THRESHOLD):
    """
    Run open-set prediction for any sklearn classifier with predict_proba.

    Each frame gets the class with the highest probability. If that
    probability is below `threshold`, the frame is labeled UNKNOWN instead.

    Parameters
    ----------
    model     : fitted sklearn Pipeline (must expose predict_proba)
    X         : np.ndarray of shape (N, features)
    threshold : float — minimum confidence to assign a class label

    Returns
    -------
    np.ndarray of shape (N,) — class strings or UNKNOWN
    """
    probs    = model.predict_proba(X)           # shape: (N, num_classes)
    max_prob = probs.max(axis=1)                # shape: (N,)
    best_cls = model.classes_[np.argmax(probs, axis=1)]
    preds    = best_cls.astype(object)
    preds[max_prob < threshold] = UNKNOWN
    return preds


def get_confidence(model, X):
    """Return the max predict_proba score per frame (used for plotting)."""
    return model.predict_proba(X).max(axis=1)


# ── Ensemble: majority vote (2 out of 3 models must agree) ───────────────────
def ensemble_majority(per_model_preds, min_agree=2):
    """
    Assign a gesture when at least `min_agree` models confidently agree on
    the same class. With 3 models and min_agree=2, one model can be uncertain
    or disagree without blocking the label.

    How it works step by step:
      1. For each frame, collect the prediction from every model.
      2. Filter out UNKNOWN votes — only confident predictions count.
      3. Find the most common class among those confident predictions.
      4. If that class appears at least `min_agree` times, assign it.
      5. Otherwise mark the frame UNKNOWN.

    Example:
      Model votes: ["thumbs_up", "thumbs_up", UNKNOWN]
      Known votes: ["thumbs_up", "thumbs_up"]
      Top class "thumbs_up" has count 2 >= min_agree 2  →  label = "thumbs_up"

      Model votes: ["thumbs_up", "fist", UNKNOWN]
      Known votes: ["thumbs_up", "fist"]
      Top class has count 1 < min_agree 2               →  label = UNKNOWN

    Parameters
    ----------
    per_model_preds : list of 1-D arrays, each length N
                      (one array per model, in the same frame order)
    min_agree       : int — minimum number of models that must agree (default 2)

    Returns
    -------
    np.ndarray of shape (N,) — class strings or UNKNOWN
    """
    stacked = np.column_stack(per_model_preds)          # shape: (N, n_models)
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
    """
    Build the three sklearn pipelines. Each pipeline:
      1. Scales features to zero mean / unit variance (StandardScaler)
      2. Runs the classifier on the scaled features

    Scaling matters because the three models are sensitive to feature magnitude
    in different ways — doing it inside the pipeline ensures CV is clean
    (the scaler is re-fit inside each fold, not on the full dataset first).
    """
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
    """
    Run stratified k-fold CV on all models and return a summary DataFrame.

    'Stratified' means each fold preserves the original class proportions —
    important here because we have exactly 25 frames per gesture class.

    Metrics reported:
      - CV Accuracy (mean and std across folds)
      - CV Macro F1  (mean and std across folds) — macro = equal weight per class
      - Train Accuracy (mean) — useful to spot overfitting
    """
    cv   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=cv,
                             scoring=["accuracy", "f1_macro"],
                             return_train_score=True)
        rows.append({
            "Model":            name,
            "CV Acc  (mean)":   res["test_accuracy"].mean(),
            "CV Acc  (std)":    res["test_accuracy"].std(),
            "CV F1   (mean)":   res["test_f1_macro"].mean(),
            "CV F1   (std)":    res["test_f1_macro"].std(),
            "Train Acc (mean)": res["train_accuracy"].mean(),
        })
    return pd.DataFrame(rows).set_index("Model")


# ── Feature summary ───────────────────────────────────────────────────────────
def save_feature_summary(X_ref, y_ref, X_all):
    """
    Save a human-readable summary of the feature space to a CSV.
    Includes per-feature mean and std across both datasets.
    """
    ref_df = pd.DataFrame(X_ref, columns=COORD_COLS)
    ref_df["split"] = "reference"
    all_df = pd.DataFrame(X_all, columns=COORD_COLS)
    all_df["split"] = "unlabeled"
    combined = pd.concat([ref_df, all_df], ignore_index=True)

    summary = combined.groupby("split")[COORD_COLS].agg(["mean", "std"])
    path = os.path.join(OUT_DIR, "feature_summary.csv")
    summary.to_csv(path)
    print(f"  Saved -> {path}")


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_cv_comparison(cv_summary):
    """Bar chart comparing CV accuracy and macro F1 across models."""
    path = os.path.join(OUT_DIR, "cv_comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("5-Fold CV — Deduplicated Reference Set (N=200, 25/class)",
                 fontsize=13)
    palette = ["#4c72b0", "#dd8452", "#55a868"]
    for ax, metric, label in zip(axes,
                                  ["CV Acc  (mean)", "CV F1   (mean)"],
                                  ["Accuracy", "Macro F1"]):
        vals = cv_summary[metric].dropna()
        errs = cv_summary[metric.replace("mean", "std")].reindex(vals.index).fillna(0)
        bars = ax.barh(vals.index, vals.values, xerr=errs.values,
                       color=palette[:len(vals)], capsize=4, edgecolor="white")
        for bar, v in zip(bars, vals.values):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=9)
        ax.set_xlabel(label)
        ax.set_xlim(0, 1.1)
        ax.axvline(1.0, color="grey", lw=0.8, ls="--")
        ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def plot_confusion_matrices(models, X, y):
    """
    Confusion matrices for all three models trained on the full reference set.

    Note: these use raw predict() with no threshold — this is a sanity check
    that the models have actually learned the gesture classes correctly before
    we apply open-set rejection on the unlabeled data.
    """
    path = os.path.join(OUT_DIR, "confusion_matrices.png")
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle("Confusion matrices — trained on full reference set", fontsize=12)
    classes = np.unique(y)
    for ax, (name, model) in zip(axes, models.items()):
        preds = model.predict(X)
        cm    = confusion_matrix(y, preds, labels=classes)
        disp  = ConfusionMatrixDisplay(cm, display_labels=classes)
        disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def plot_open_set_results(results_df):
    """
    Two panels:
      Left  - ensemble label distribution (known classes + UNKNOWN)
      Right - per-model UNKNOWN rejection rate
    """
    path = os.path.join(OUT_DIR, "open_set_results.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Open-set classification  (2 out of 3 models must agree)",
        fontsize=13
    )

    # Left: ensemble gesture distribution
    ens_counts = results_df["ensemble_open_set"].value_counts()
    colors = ["#c44e52" if lbl == UNKNOWN else "#4c72b0"
              for lbl in ens_counts.index]
    bars = ax1.barh(ens_counts.index, ens_counts.values,
                    color=colors, edgecolor="white")
    for bar, v in zip(bars, ens_counts.values):
        ax1.text(v + 50, bar.get_y() + bar.get_height() / 2,
                 f"{v:,}", va="center", fontsize=9)
    ax1.set_xlabel("Frames")
    ax1.set_title("Ensemble open-set — label distribution", fontsize=11)
    ax1.grid(axis="x", alpha=0.3)

    # Right: per-model UNKNOWN rate
    model_cols  = ["logistic_regression", "mlp", "random_forest"]
    model_names = ["Logistic Reg.", "MLP", "Random Forest"]
    unknown_pct = [
        (results_df[col] == UNKNOWN).mean() * 100
        for col in model_cols
    ]
    ax2.bar(model_names, unknown_pct,
            color=["#4c72b0", "#dd8452", "#55a868"],
            edgecolor="white")
    for i, v in enumerate(unknown_pct):
        ax2.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)
    ax2.set_ylabel("% frames marked UNKNOWN")
    ax2.set_title("Per-model rejection rate", fontsize=11)
    ax2.set_ylim(0, max(unknown_pct) * 1.2 if max(unknown_pct) > 0 else 10)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def plot_confidence_distribution(results_df, confidences):
    """
    Violin plot of per-model confidence scores, split by whether the
    ensemble classified the frame or left it as UNKNOWN.

    A healthy plot shows:
      - Classified frames clustered near high confidence (close to 1.0)
      - UNKNOWN frames spread lower or below the threshold line
    """
    path = os.path.join(OUT_DIR, "confidence_distribution.png")
    model_cols  = ["logistic_regression", "mlp", "random_forest"]
    model_names = ["Logistic Reg.", "MLP", "Random Forest"]
    ens         = results_df["ensemble_open_set"].to_numpy()
    is_unknown  = ens == UNKNOWN

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Confidence score distribution — classified vs UNKNOWN", fontsize=12)

    colors = {"Classified": "#4c72b0", UNKNOWN: "#c44e52"}
    for ax, col_name, conf, display_name in zip(
            axes, model_cols, confidences, model_names):
        data = {
            "Classified": conf[~is_unknown],
            UNKNOWN:      conf[is_unknown],
        }
        parts = ax.violinplot(list(data.values()), positions=[0, 1],
                              showmedians=True, showextrema=True)
        for pc, color in zip(parts["bodies"], colors.values()):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(list(data.keys()))
        ax.set_ylabel("Max predicted probability")
        ax.set_title(display_name, fontsize=10)
        ax.axhline(PROBA_THRESHOLD, color="grey", ls="--", lw=1,
                   label=f"Threshold ({PROBA_THRESHOLD})")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def plot_agreement_scatter(results_df, confidences):
    """
    Scatter plot: x = number of models that agree (0-3),
                  y = average confidence across models.
    Classified frames in blue, UNKNOWN in red.

    A healthy plot should show:
      - Blue dots clustered at x=3 with high confidence
      - Red dots spread across lower x values and/or lower confidence
    """
    path = os.path.join(OUT_DIR, "agreement_scatter.png")
    model_cols = ["logistic_regression", "mlp", "random_forest"]
    votes      = results_df[model_cols]
    ens        = results_df["ensemble_open_set"].to_numpy()
    is_unknown = ens == UNKNOWN

    agree_n = votes.apply(
        lambda row: (row.values == row.value_counts().idxmax()).sum(), axis=1
    ).to_numpy()

    avg_conf = np.mean(np.column_stack(confidences), axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(agree_n[~is_unknown], avg_conf[~is_unknown],
               c="#4c72b0", alpha=0.4, s=8, label="Classified")
    ax.scatter(agree_n[is_unknown], avg_conf[is_unknown],
               c="#c44e52", alpha=0.3, s=8, label="UNKNOWN")
    ax.axhline(PROBA_THRESHOLD, color="#dd8452", ls="--", lw=1,
               label=f"Confidence threshold ({PROBA_THRESHOLD})")
    ax.set_xlabel("Models in agreement (out of 3)")
    ax.set_ylabel("Average confidence score")
    ax.set_title("Confidence vs model agreement — open-set view", fontsize=11)
    ax.legend(fontsize=9, markerscale=3)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Hand Gesture Open-Set Classification (3-Model Ensemble)")
    print("=" * 60)
    print(f"\n  Config: proba threshold={PROBA_THRESHOLD}  "
          f"rule=majority (2/3 models must agree)")
    print(f"  Output folder: {OUT_DIR}/\n")

    # 1. Load data
    print("[1] Loading data ...")
    X_ref, y_ref   = load_reference("data/prototype_dataset.csv")
    X_all, meta_df = load_unlabeled("data/unlabeled_set.csv")
    print(f"    Features  : {X_ref.shape[1]} columns")
    print(f"    Classes   : {np.unique(y_ref).tolist()}")

    # 2. Save feature summary
    print("\n[2] Saving feature summary ...")
    save_feature_summary(X_ref, y_ref, X_all)

    # 3. Cross-validation
    print("\n[3] Closed-set CV on reference set (sanity check) ...")
    models     = build_models()
    cv_summary = run_cv(models, X_ref, y_ref)
    print()
    print(cv_summary.to_string(float_format=lambda x: f"{x:.4f}"))
    cv_path = os.path.join(OUT_DIR, "cv_summary.csv")
    cv_summary.to_csv(cv_path)
    print(f"\n  Saved -> {cv_path}")
    plot_cv_comparison(cv_summary)

    # 4. Fit on full reference set
    print("\n[4] Fitting all models on full reference set ...")
    for name, model in models.items():
        model.fit(X_ref, y_ref)
        train_acc = np.mean(model.predict(X_ref) == y_ref)
        print(f"    {name:<25}  train acc = {train_acc:.4f}")

    plot_confusion_matrices(models, X_ref, y_ref)

    # 5. Open-set prediction on unlabeled set
    print("\n[5] Open-set predictions on unlabeled set ...")
    print(f"    A frame is labelled only when at least 2 models "
          f"confidently agree.\n")

    results     = meta_df.copy()
    all_preds   = []
    confidences = []

    for name, model in models.items():
        col_key    = name.lower().replace(" ", "_")
        preds_open = predict_open(model, X_all, threshold=PROBA_THRESHOLD)
        conf       = get_confidence(model, X_all)

        results[col_key] = preds_open
        all_preds.append(preds_open)
        confidences.append(conf)

        n_unk = (preds_open == UNKNOWN).sum()
        print(f"    {name:<25}  UNKNOWN={n_unk:>5} "
              f"({n_unk / len(preds_open) * 100:.1f}%)")

    # 6. Ensemble
    results["ensemble_open_set"] = ensemble_majority(all_preds, min_agree=2)

    # 7. Report
    print("\n[6] Open-set ensemble results ...")
    ens          = results["ensemble_open_set"]
    n_unknown    = (ens == UNKNOWN).sum()
    n_classified = (ens != UNKNOWN).sum()

    print(f"\n    Total frames   : {len(ens):>6}")
    print(f"    Classified     : {n_classified:>6}  "
          f"({n_classified / len(ens) * 100:.1f}%)")
    print(f"    UNKNOWN        : {n_unknown:>6}  "
          f"({n_unknown / len(ens) * 100:.1f}%)")
    print()
    print("    Classified gesture breakdown:")
    if n_classified > 0:
        for lbl, cnt in ens[ens != UNKNOWN].value_counts().items():
            print(f"      {lbl:<20}  {cnt:>5}  "
                  f"({cnt / n_classified * 100:.1f}% of classified)")
    else:
        print("      (no frames classified — all marked UNKNOWN)")

    # 8. Save predictions and plots
    pred_path = os.path.join(OUT_DIR, "predictions_open_set.csv")
    results.to_csv(pred_path, index=False)
    print(f"\n  Saved -> {pred_path}")

    plot_open_set_results(results)
    plot_confidence_distribution(results, confidences)
    plot_agreement_scatter(results, confidences)

    print("\n[Done]")
    print(f"  All outputs in: {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()