"""
Verification script for entity-level train/validation split.

Checks:
- Total Source 1 entities
- Train split count & percentage
- Validation split count & percentage
- Overlap between train and validation sets (must be 0)
- Ground-truth coverage across splits (must be 100%)
"""

import os
import sys
from src.data_loader import DataLoader
from src.validation import ValidationSplitter


def main():
    print("=========================================================")
    print(" Amazon ML Challenge 2026: Validation Split Verification ")
    print("=========================================================\n")

    loader = DataLoader(data_dir="data")
    splitter = ValidationSplitter(data_loader=loader, seed=42)

    splits_dir = "experiments/splits"
    train_path = os.path.join(splits_dir, "train_s1_ids.txt")
    val_path = os.path.join(splits_dir, "val_s1_ids.txt")

    print(f"Generating deterministic entity splits (seed=42, val_ratio=0.20)...")
    splitter.generate_and_save_splits(output_dir=splits_dir, val_ratio=0.20)

    print(f"Saved splits to:")
    print(f"  - Train split:      {train_path}")
    print(f"  - Validation split: {val_path}\n")

    # Load and verify
    print("Verifying split integrity...")
    all_s1_ids = splitter.get_all_source1_entity_ids("train")
    total_s1 = len(all_s1_ids)

    train_ids = splitter.load_split_ids(train_path)
    val_ids = splitter.load_split_ids(val_path)

    train_count = len(train_ids)
    val_count = len(val_ids)

    train_pct = (train_count / total_s1) * 100 if total_s1 else 0
    val_pct = (val_count / total_s1) * 100 if total_s1 else 0

    # Check disjoint overlap
    overlap = train_ids.intersection(val_ids)
    overlap_count = len(overlap)

    # Check partition completeness
    combined_ids = train_ids.union(val_ids)
    sum_matches_total = (train_count + val_count) == total_s1
    all_covered = combined_ids == set(all_s1_ids)

    # Check Ground Truth coverage
    print("Loading ground-truth mapping to verify coverage...")
    gt_mapping = loader.load_ground_truth()
    total_gt_rows = len(gt_mapping)

    gt_train_covered = sum(1 for s1_id in gt_mapping if s1_id in train_ids)
    gt_val_covered = sum(1 for s1_id in gt_mapping if s1_id in val_ids)
    gt_total_covered = gt_train_covered + gt_val_covered
    all_gt_covered = (gt_total_covered == total_gt_rows) and (total_gt_rows == total_s1)

    print("\n------------------- SPLIT METRICS -------------------")
    print(f"Total Source 1 entities:             {total_s1:,}")
    print(f"Train split count:                   {train_count:,}")
    print(f"Validation split count:              {val_count:,}")
    print(f"Train percentage:                    {train_pct:.2f}%")
    print(f"Validation percentage:                {val_pct:.2f}%")
    print(f"Overlap between train and val IDs:   {overlap_count:,}")
    print(f"Ground-truth rows in train split:    {gt_train_covered:,} ({(gt_train_covered/total_gt_rows)*100:.2f}%)")
    print(f"Ground-truth rows in val split:      {gt_val_covered:,} ({(gt_val_covered/total_gt_rows)*100:.2f}%)")
    print(f"Total Ground-truth rows covered:     {gt_total_covered:,} / {total_gt_rows:,}")
    print("-----------------------------------------------------\n")

    # Assertions / Integrity Check Results
    print("---------------- INTEGRITY CHECKS ------------------")
    print(f"[*] Train + Val == Total S1:         {'PASSED' if sum_matches_total else 'FAILED'}")
    print(f"[*] Overlap == 0:                    {'PASSED' if overlap_count == 0 else 'FAILED'}")
    print(f"[*] All S1 entities accounted for:   {'PASSED' if all_covered else 'FAILED'}")
    print(f"[*] All Ground-Truth S1 IDs covered: {'PASSED' if all_gt_covered else 'FAILED'}")
    print("-----------------------------------------------------")

    if sum_matches_total and overlap_count == 0 and all_covered and all_gt_covered:
        print("\nAll validation checks PASSED successfully!")
    else:
        print("\nValidation check FAILED! Please review split parameters.")
        sys.exit(1)


if __name__ == "__main__":
    main()
