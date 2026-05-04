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

Usage:
    python create_unlabeled_set.py

Inputs  (same directory):
    prototype_dataset.csv     – reference set with 'gesture_name' label
    normalised_hand_data.csv  – raw dataset (no labels)

Output:
    unlabeled_set.csv         – raw data with all prototype rows removed
"""

import pandas as pd

# ── Load ───────────────────────────────────────────────────────────────────────
prototype = pd.read_csv("data/prototype_dataset.csv")
raw       = pd.read_csv("data/normalised_hand_data.csv")

print(f"Prototype rows        : {len(prototype):>6}")
print(f"Raw rows              : {len(raw):>6}")

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

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = "data/unlabeled_set.csv"
unlabeled.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")