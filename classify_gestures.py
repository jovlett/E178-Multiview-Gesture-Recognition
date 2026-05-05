"""
classify_gestures.py
====================
Open-set multi-class hand gesture classification using three models.
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
from sklearn.metrics         import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os

# ── Output folders ─────────────────────────────────────────────────────────────
OUT_DIR   = "outputs/final outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
UNKNOWN         = "UNKNOWN"
PROBA_THRESHOLD = 0.80
MIN_AGREE       = 2

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

COORD_COLS = _build_coord_cols()


# ── Data loaders ───────────────────────────────────────────────────────────────
def load_reference(path="data/prototype_dataset.csv"):
    df = pd.read_csv(path)
    before = len(df)
    df = df.drop_duplicates(subset=COORD_COLS).reset_index(drop=True)
    print(f"    Reference  : {before} -> {len(df)} rows  "
          f"({before - len(df)} duplicates removed)")
    return df[COORD_COLS].to_numpy(float), df["gesture_name"].to_numpy()

def load_unlabeled(path="data/unlabeled_set.csv"):
    df = pd.read_csv(path)
    print(f"    Unlabeled  : {len(df)} rows")
    return df[COORD_COLS].to_numpy(float), df[["video_id", "frame_id"]]


# ── Open-set prediction wrapper ───────────────────────────────────────────────
def predict_open(model, X, threshold=PROBA_THRESHOLD):
    probs    = model.predict_proba(X)
    max_prob = probs.max(axis=1)
    best_cls = model.classes_[np.argmax(probs, axis=1)]
    preds    = best_cls.astype(object)
    preds[max_prob < threshold] = UNKNOWN
    return preds

def get_confidence(model, X):
    return model.predict_proba(X).max(axis=1)


# ── Ensemble ──────────────────────────────────────────────────────────────────
def ensemble_majority(per_model_preds, min_agree=2):
    stacked = np.column_stack(per_model_preds)
    result  = np.full(len(stacked), UNKNOWN, dtype=object)
    for i, row in enumerate(stacked):
        known_votes = row[row != UNKNOWN]
        if len(known_votes) == 0:
            continue
        unique, counts = np.unique(known_votes, return_counts=True)
        top_class  = unique[np.argmax(counts)]
        top_count  = counts.max()
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
    ref_df = pd.DataFrame(X_ref, columns=COORD_COLS)
    ref_df["split"] = "reference"
    all_df = pd.DataFrame(X_all, columns=COORD_COLS)
    all_df["split"] = "unlabeled"
    combined = pd.concat([ref_df, all_df], ignore_index=True)
    summary  = combined.groupby("split")[COORD_COLS].agg(["mean", "std"])
    path = os.path.join(OUT_DIR, "feature_summary.csv")
    summary.to_csv(path)
    print(f"  Saved -> {path}")


# ── NEW: Confidence extremes per gesture ──────────────────────────────────────
def save_confidence_extremes(results_df, confidences):
    """
    For each ensemble-assigned gesture class (including UNKNOWN), find:
      - The frame with the HIGHEST average confidence across all three models
      - The frame with the LOWEST  average confidence across all three models
        that was still classified into that label (i.e. met the threshold)

    For UNKNOWN frames, confidence = the average score of the three models,
    showing how uncertain those frames truly were.

    Saves:
      outputs/final outputs/confidence_extremes.csv   — machine-readable table
      outputs/final outputs/confidence_extremes.png   — visual summary
    """
    avg_conf = np.mean(np.column_stack(confidences), axis=1)
    results_df = results_df.copy()
    results_df["avg_confidence"] = avg_conf

    rows = []
    for gesture in sorted(results_df["ensemble_open_set"].unique()):
        subset = results_df[results_df["ensemble_open_set"] == gesture]
        if len(subset) == 0:
            continue

        best_idx = subset["avg_confidence"].idxmax()
        worst_idx = subset["avg_confidence"].idxmin()

        best_row  = subset.loc[best_idx]
        worst_row = subset.loc[worst_idx]

        rows.append({
            "gesture":              gesture,
            "total_frames":         len(subset),
            # Best case
            "best_video_id":        best_row["video_id"],
            "best_frame_id":        best_row["frame_id"],
            "best_confidence_pct":  round(best_row["avg_confidence"] * 100, 2),
            "best_lr":              best_row["logistic_regression"],
            "best_mlp":             best_row["mlp"],
            "best_rf":              best_row["random_forest"],
            # Worst (but still classified) case
            "worst_video_id":       worst_row["video_id"],
            "worst_frame_id":       worst_row["frame_id"],
            "worst_confidence_pct": round(worst_row["avg_confidence"] * 100, 2),
            "worst_lr":             worst_row["logistic_regression"],
            "worst_mlp":            worst_row["mlp"],
            "worst_rf":             worst_row["random_forest"],
        })

    extremes_df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "confidence_extremes.csv")
    extremes_df.to_csv(csv_path, index=False)
    print(f"  Saved -> {csv_path}")

    _plot_confidence_extremes(extremes_df)
    return extremes_df


def _plot_confidence_extremes(extremes_df):
    """
    Horizontal dumbbell chart — one row per gesture.
    Each row shows a line from worst to best confidence,
    with coloured dots at each end and the values labelled.
    UNKNOWN gets a distinct muted styling.
    """
    path = os.path.join(OUT_DIR, "confidence_extremes.png")

    # Sort: known gestures alphabetically, UNKNOWN last
    known = extremes_df[extremes_df["gesture"] != UNKNOWN].sort_values("gesture")
    unk   = extremes_df[extremes_df["gesture"] == UNKNOWN]
    df    = pd.concat([known, unk], ignore_index=True)

    n   = len(df)
    fig_h = max(5, n * 0.82 + 2.2)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # ── Colour palette ────────────────────────────────────────────────────────
    BEST_COL    = "#00e5a0"   # vivid mint
    WORST_COL   = "#ff5f6d"   # vivid coral
    UNK_BEST    = "#8888aa"
    UNK_WORST   = "#555577"
    LINE_COL    = "#2a2d3a"
    TEXT_COL    = "#e8eaf0"
    LABEL_COL   = "#9ea3b8"
    GRID_COL    = "#1e2130"

    # Background horizontal bands
    for i in range(n):
        ax.axhspan(i - 0.45, i + 0.45,
                   color="#161924" if i % 2 == 0 else "#0f1117", zorder=0)

    for i, row in df.iterrows():
        is_unk  = row["gesture"] == UNKNOWN
        bc      = UNK_BEST  if is_unk else BEST_COL
        wc      = UNK_WORST if is_unk else WORST_COL
        best_x  = row["best_confidence_pct"]
        worst_x = row["worst_confidence_pct"]

        # connector line
        ax.plot([worst_x, best_x], [i, i],
                color=LINE_COL, lw=2.5, solid_capstyle="round", zorder=1)

        # filled range band
        ax.barh(i, best_x - worst_x, left=worst_x, height=0.18,
                color=bc, alpha=0.12, zorder=1)

        # dots
        ax.scatter(best_x,  i, color=bc, s=110, zorder=3,
                   edgecolors="white", linewidths=0.6)
        ax.scatter(worst_x, i, color=wc, s=110, zorder=3,
                   edgecolors="white", linewidths=0.6)

        # value labels
        ax.text(best_x + 0.8,  i,  f"{best_x:.1f}%",
                va="center", ha="left",  fontsize=8.5,
                color=bc, fontweight="bold")
        ax.text(worst_x - 0.8, i,  f"{worst_x:.1f}%",
                va="center", ha="right", fontsize=8.5,
                color=wc, fontweight="bold")

        # frame count badge
        ax.text(101.5, i, f"n={row['total_frames']:,}",
                va="center", ha="left", fontsize=7.5,
                color=LABEL_COL)

    # ── Y-axis: gesture labels ────────────────────────────────────────────────
    gesture_labels = df["gesture"].tolist()
    ax.set_yticks(range(n))
    ax.set_yticklabels(gesture_labels, fontsize=10.5,
                       color=TEXT_COL, fontfamily="monospace")
    ax.tick_params(axis="y", length=0, pad=8)

    # ── X-axis ────────────────────────────────────────────────────────────────
    ax.set_xlim(0, 110)
    ax.set_xlabel("Average ensemble confidence  (%)",
                  color=LABEL_COL, fontsize=10, labelpad=8)
    ax.tick_params(axis="x", colors=LABEL_COL, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # vertical grid lines
    for xv in range(0, 101, 20):
        ax.axvline(xv, color=GRID_COL, lw=0.8, zorder=0)
    ax.axvline(PROBA_THRESHOLD * 100, color="#ffcc44",
               lw=1.2, ls="--", zorder=2, alpha=0.7)
    ax.text(PROBA_THRESHOLD * 100 + 0.3, n - 0.1,
            f"threshold\n{int(PROBA_THRESHOLD*100)}%",
            color="#ffcc44", fontsize=7.5, va="top", alpha=0.85)

    # ── Title & legend ────────────────────────────────────────────────────────
    ax.set_title("Confidence Extremes per Gesture Class\n"
                 "Best (highest avg confidence)  vs  Worst (lowest avg confidence)",
                 color=TEXT_COL, fontsize=13, fontweight="bold",
                 pad=16, loc="left")

    legend_els = [
        mpatches.Patch(color=BEST_COL,  label="Best  (highest confidence)"),
        mpatches.Patch(color=WORST_COL, label="Worst (lowest confidence still classified)"),
        mpatches.Patch(color=UNK_BEST,  label="UNKNOWN range"),
    ]
    ax.legend(handles=legend_els, loc="lower right",
              framealpha=0.15, edgecolor="#444",
              labelcolor=TEXT_COL, fontsize=8.5)

    ax.invert_yaxis()
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved -> {path}")


# ── Standard plots ─────────────────────────────────────────────────────────────
def plot_cv_comparison(cv_summary):
    path = os.path.join(OUT_DIR, "cv_comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("5-Fold CV — Deduplicated Reference Set (N=200, 25/class)", fontsize=13)
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
        ax.set_xlabel(label); ax.set_xlim(0, 1.1)
        ax.axvline(1.0, color="grey", lw=0.8, ls="--")
        ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {path}")

def plot_confusion_matrices(models, X, y):
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
        ax.set_title(name, fontsize=10); ax.set_xlabel(""); ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {path}")

def plot_open_set_results(results_df):
    path = os.path.join(OUT_DIR, "open_set_results.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Open-set classification  (2 out of 3 models must agree)", fontsize=13)
    ens_counts = results_df["ensemble_open_set"].value_counts()
    colors = ["#c44e52" if lbl == UNKNOWN else "#4c72b0" for lbl in ens_counts.index]
    bars = ax1.barh(ens_counts.index, ens_counts.values, color=colors, edgecolor="white")
    for bar, v in zip(bars, ens_counts.values):
        ax1.text(v + 50, bar.get_y() + bar.get_height() / 2, f"{v:,}", va="center", fontsize=9)
    ax1.set_xlabel("Frames"); ax1.set_title("Ensemble open-set — label distribution", fontsize=11)
    ax1.grid(axis="x", alpha=0.3)
    model_cols  = ["logistic_regression", "mlp", "random_forest"]
    model_names = ["Logistic Reg.", "MLP", "Random Forest"]
    unknown_pct = [(results_df[col] == UNKNOWN).mean() * 100 for col in model_cols]
    ax2.bar(model_names, unknown_pct, color=["#4c72b0", "#dd8452", "#55a868"], edgecolor="white")
    for i, v in enumerate(unknown_pct):
        ax2.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)
    ax2.set_ylabel("% frames marked UNKNOWN"); ax2.set_title("Per-model rejection rate", fontsize=11)
    ax2.set_ylim(0, max(unknown_pct) * 1.2 if max(unknown_pct) > 0 else 10)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {path}")

def plot_confidence_distribution(results_df, confidences):
    path = os.path.join(OUT_DIR, "confidence_distribution.png")
    model_cols  = ["logistic_regression", "mlp", "random_forest"]
    model_names = ["Logistic Reg.", "MLP", "Random Forest"]
    ens        = results_df["ensemble_open_set"].to_numpy()
    is_unknown = ens == UNKNOWN
    fig, axes  = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Confidence score distribution — classified vs UNKNOWN", fontsize=12)
    colors = {"Classified": "#4c72b0", UNKNOWN: "#c44e52"}
    for ax, col_name, conf, display_name in zip(axes, model_cols, confidences, model_names):
        data  = {"Classified": conf[~is_unknown], UNKNOWN: conf[is_unknown]}
        parts = ax.violinplot(list(data.values()), positions=[0, 1],
                              showmedians=True, showextrema=True)
        for pc, color in zip(parts["bodies"], colors.values()):
            pc.set_facecolor(color); pc.set_alpha(0.6)
        ax.set_xticks([0, 1]); ax.set_xticklabels(list(data.keys()))
        ax.set_ylabel("Max predicted probability"); ax.set_title(display_name, fontsize=10)
        ax.axhline(PROBA_THRESHOLD, color="grey", ls="--", lw=1,
                   label=f"Threshold ({PROBA_THRESHOLD})")
        ax.legend(fontsize=8); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {path}")

def plot_agreement_scatter(results_df, confidences):
    path = os.path.join(OUT_DIR, "agreement_scatter.png")
    model_cols = ["logistic_regression", "mlp", "random_forest"]
    votes      = results_df[model_cols]
    ens        = results_df["ensemble_open_set"].to_numpy()
    is_unknown = ens == UNKNOWN
    agree_n    = votes.apply(
        lambda row: (row.values == row.value_counts().idxmax()).sum(), axis=1
    ).to_numpy()
    avg_conf = np.mean(np.column_stack(confidences), axis=1)
    fig, ax  = plt.subplots(figsize=(8, 5))
    ax.scatter(agree_n[~is_unknown], avg_conf[~is_unknown],
               c="#4c72b0", alpha=0.4, s=8, label="Classified")
    ax.scatter(agree_n[is_unknown], avg_conf[is_unknown],
               c="#c44e52", alpha=0.3, s=8, label="UNKNOWN")
    ax.axhline(PROBA_THRESHOLD, color="#dd8452", ls="--", lw=1,
               label=f"Confidence threshold ({PROBA_THRESHOLD})")
    ax.set_xlabel("Models in agreement (out of 3)"); ax.set_ylabel("Average confidence score")
    ax.set_title("Confidence vs model agreement — open-set view", fontsize=11)
    ax.legend(fontsize=9, markerscale=3); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Hand Gesture Open-Set Classification (3-Model Ensemble)")
    print("=" * 60)
    print(f"\n  Config: proba threshold={PROBA_THRESHOLD}  "
          f"rule=majority (2/3 models must agree)")
    print(f"  Output folder: {OUT_DIR}/\n")

    print("[1] Loading data ...")
    X_ref, y_ref   = load_reference("data/prototype_dataset.csv")
    X_all, meta_df = load_unlabeled("data/unlabeled_set.csv")
    print(f"    Features  : {X_ref.shape[1]} columns")
    print(f"    Classes   : {np.unique(y_ref).tolist()}")

    print("\n[2] Saving feature summary ...")
    save_feature_summary(X_ref, y_ref, X_all)

    print("\n[3] Closed-set CV on reference set (sanity check) ...")
    models     = build_models()
    cv_summary = run_cv(models, X_ref, y_ref)
    print()
    print(cv_summary.to_string(float_format=lambda x: f"{x:.4f}"))
    cv_path = os.path.join(OUT_DIR, "cv_summary.csv")
    cv_summary.to_csv(cv_path)
    print(f"\n  Saved -> {cv_path}")
    plot_cv_comparison(cv_summary)

    print("\n[4] Fitting all models on full reference set ...")
    for name, model in models.items():
        model.fit(X_ref, y_ref)
        train_acc = np.mean(model.predict(X_ref) == y_ref)
        print(f"    {name:<25}  train acc = {train_acc:.4f}")
    plot_confusion_matrices(models, X_ref, y_ref)

    print("\n[5] Open-set predictions on unlabeled set ...")
    print(f"    A frame is labelled only when at least 2 models confidently agree.\n")

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

    results["ensemble_open_set"] = ensemble_majority(all_preds, min_agree=MIN_AGREE)

    print("\n[6] Open-set ensemble results ...")
    ens          = results["ensemble_open_set"]
    n_unknown    = (ens == UNKNOWN).sum()
    n_classified = (ens != UNKNOWN).sum()
    print(f"\n    Total frames   : {len(ens):>6}")
    print(f"    Classified     : {n_classified:>6}  ({n_classified / len(ens) * 100:.1f}%)")
    print(f"    UNKNOWN        : {n_unknown:>6}  ({n_unknown / len(ens) * 100:.1f}%)")
    print()
    print("    Classified gesture breakdown:")
    if n_classified > 0:
        for lbl, cnt in ens[ens != UNKNOWN].value_counts().items():
            print(f"      {lbl:<20}  {cnt:>5}  ({cnt / n_classified * 100:.1f}% of classified)")
    else:
        print("      (no frames classified — all marked UNKNOWN)")

    pred_path = os.path.join(OUT_DIR, "predictions_open_set.csv")
    results.to_csv(pred_path, index=False)
    print(f"\n  Saved -> {pred_path}")

    plot_open_set_results(results)
    plot_confidence_distribution(results, confidences)
    plot_agreement_scatter(results, confidences)

    print("\n[7] Computing confidence extremes per gesture ...")
    save_confidence_extremes(results, confidences)

    print("\n[Done]")
    print(f"  All outputs in: {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()