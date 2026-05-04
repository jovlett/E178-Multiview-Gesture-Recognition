"""
create_unlabeled_set.py
-----------------------
Removes all rows that appear in the reference prototype dataset from the
raw normalised hand data, producing a clean unlabeled set for training.

Matching strategy: video_id + frame_id
  - The prototype feature values are stored at reduced float precision
    (truncated to ~8 d.p.) vs the full-precision raw data, so exact
    numeric matching fails. video_id + frame_id uniquely identifies each
    hand pose frame and is consistent across both files.

Cleaning steps (all applied here, not downstream):
  1. Exclude bad video IDs (EXCLUDED_VIDEOS)
  2. Remove prototype frames (anti-join)
  3. Deduplicate on feature columns — the raw data contains every frame
     exactly twice, so dedup halves the row count to unique poses only.

Usage:
    python create_unlabeled_set.py

Inputs  (same directory):
    prototype_dataset.csv     – reference set with 'gesture_name' label
    normalised_hand_data.csv  – raw dataset (no labels)

Output:
    unlabeled_set.csv         – clean, deduplicated, ready-to-use unlabeled set
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
# data_18 was found to be unuseable and is excluded before any processing.
EXCLUDED_VIDEOS = {"data_18"}

# ── Load ───────────────────────────────────────────────────────────────────────
prototype = pd.read_csv("data/prototype_dataset.csv")
raw       = pd.read_csv("data/normalised_hand_data.csv")

print(f"Prototype rows        : {len(prototype):>6}")
print(f"Raw rows (before excl): {len(raw):>6}")

# ── Exclude bad videos ────────────────────────────────────────────────────────
excluded_mask = raw["video_id"].isin(EXCLUDED_VIDEOS)
print(f"Rows from excluded videos ({', '.join(EXCLUDED_VIDEOS)}): {excluded_mask.sum():>6}")
raw = raw[~excluded_mask].reset_index(drop=True)
print(f"Raw rows (after excl) : {len(raw):>6}")

# ── Build unique key set from prototype ───────────────────────────────────────
# Each video_id + frame_id maps to exactly one gesture, so dedup is safe.
JOIN_KEYS = ["video_id", "frame_id"]
proto_keys = prototype[JOIN_KEYS].drop_duplicates()
print(f"Unique prototype keys : {len(proto_keys):>6}")

# ── Anti-join: keep raw rows whose key does NOT appear in the prototype ────────
proto_keys = proto_keys.copy()
proto_keys["_in_prototype"] = True

merged    = raw.merge(proto_keys, on=JOIN_KEYS, how="left")
unlabeled = raw[merged["_in_prototype"].isna()].reset_index(drop=True)

print(f"Rows removed          : {len(raw) - len(unlabeled):>6}")
print(f"Unlabeled rows        : {len(unlabeled):>6}")

# ── Deduplicate on feature columns ────────────────────────────────────────────
# The raw data contains every frame exactly twice. Dedup here so the output
# file is clean and downstream scripts don't need to know about this quirk.
before_dedup = len(unlabeled)
unlabeled = unlabeled.drop_duplicates(subset=COORD_COLS).reset_index(drop=True)
print(f"Duplicates removed    : {before_dedup - len(unlabeled):>6}")
print(f"Final unique rows     : {len(unlabeled):>6}")

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = "data/unlabeled_set.csv"
unlabeled.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")