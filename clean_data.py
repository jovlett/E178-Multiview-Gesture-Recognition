"""
clean_data.py
---------------
Cleans the raw normalised hand data by:
  1. Excluding bad video IDs (EXCLUDED_VIDEOS)
  2. Deduplicating on feature columns — the raw data contains every frame
     exactly twice, so dedup halves the row count to unique poses only.

Usage:
    python cleaned_data.py

Input  (same directory):
    normalised_hand_data.csv  – raw dataset (no labels)

Output:
    cleaned_data.csv          – clean, deduplicated dataset
"""

import pandas as pd

# ── Feature columns (used for deduplication) ──────────────────────────────────
FINGER_PREFIXES = {"Thumb": "TH", "Pinky": "F1", "Ring": "F2",
                   "Middle": "F3", "Index": "F4"}
FINGER_JOINTS   = ["KNU1_B", "KNU1_A", "KNU2_A", "KNU3_A"]

COORD_COLS = ["PALM_POSITION_X", "PALM_POSITION_Y", "PALM_POSITION_Z"]
for _p in FINGER_PREFIXES.values():
    for _s in FINGER_JOINTS:
        COORD_COLS += [f"{_p}_{_s}_{ax}" for ax in ("X", "Y", "Z")]

# ── Videos to exclude entirely ────────────────────────────────────────────────
# data_18 was found to be unusable and is excluded before any processing.
EXCLUDED_VIDEOS = {"data_18"}

# ── Load ───────────────────────────────────────────────────────────────────────
raw = pd.read_csv("raw data/normalised_hand_data.csv")

print(f"Raw rows (before excl): {len(raw):>6}")

# ── Exclude bad videos ────────────────────────────────────────────────────────
excluded_mask = raw["video_id"].isin(EXCLUDED_VIDEOS)
print(f"Rows from excluded videos ({', '.join(EXCLUDED_VIDEOS)}): {excluded_mask.sum():>6}")
raw = raw[~excluded_mask].reset_index(drop=True)
print(f"Raw rows (after excl) : {len(raw):>6}")

# ── Deduplicate on feature columns ────────────────────────────────────────────
# The raw data contains every frame exactly twice. Dedup here so the output
# file is clean and downstream scripts don't need to know about this quirk.
before_dedup = len(raw)
cleaned = raw.drop_duplicates(subset=COORD_COLS).reset_index(drop=True)
print(f"Duplicates removed    : {before_dedup - len(cleaned):>6}")
print(f"Final unique rows     : {len(cleaned):>6}")

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = "data/cleaned_normalised_data_NOdata18.csv"
cleaned.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")