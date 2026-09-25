"""
Execution script for Blocking and Candidate Generation Analysis.

Evaluates normalization strategies, token frequencies, block size distributions,
and ground-truth key coverage on the validation split.
"""

import os
import sys
import csv
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

from src.data_loader import DataLoader
from src.validation import ValidationSplitter
from src.blocking_analysis import (
    normalize_text,
    extract_tokens,
    extract_address_components,
    BlockingAnalyzer,
    LEGAL_SUFFIXES
)


def main():
    print("=========================================================")
    print(" Amazon ML Challenge 2026: Blocking Key & Field Analysis ")
    print("=========================================================\n")

    loader = DataLoader(data_dir="data")
    splitter = ValidationSplitter(data_loader=loader, seed=42)
    analyzer = BlockingAnalyzer()

    val_split_path = "experiments/splits/val_s1_ids.txt"
    if not os.path.exists(val_split_path):
        print("Validation split file not found. Running split generation...")
        splitter.generate_and_save_splits(output_dir="experiments/splits")

    print("1. Loading Validation Source 1 Entities...")
    val_s1_ids = splitter.load_split_ids(val_split_path)
    print(f"   Loaded {len(val_s1_ids):,} validation S1 IDs.")

    # Data structures for validation analysis
    val_s1_entities: Dict[str, Tuple[str, str, str]] = {}
    val_name_raw_counter = Counter()
    val_name_norm_counter = Counter()
    val_name_country_counter = Counter()
    val_name_tokens_counter = Counter()
    val_addr_tokens_counter = Counter()
    val_postal_counter = Counter()

    missing_addr_count = 0
    total_val_s1 = 0

    s1_train_file = loader.get_file_path("train", "source1")
    with open(s1_train_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        _ = next(reader, None)
        for row in reader:
            if not row:
                continue
            eid, name, addr, country = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
            if eid in val_s1_ids:
                total_val_s1 += 1
                val_s1_entities[eid] = (name, addr, country)

                # Name normalizations
                raw_lower = name.lower()
                norm_name = normalize_text(name, remove_legal_suffix=True)
                name_country_key = f"{norm_name}___{country.lower()}"

                val_name_raw_counter[raw_lower] += 1
                val_name_norm_counter[norm_name] += 1
                val_name_country_counter[name_country_key] += 1

                # Tokens
                tokens = extract_tokens(name)
                for t in tokens:
                    if t not in LEGAL_SUFFIXES and len(t) > 2:
                        val_name_tokens_counter[t] += 1

                # Address
                if not addr:
                    missing_addr_count += 1
                else:
                    addr_info = extract_address_components(addr)
                    for at in addr_info["tokens"]:
                        if len(at) > 2:
                            val_addr_tokens_counter[at] += 1
                    if addr_info["has_postal"]:
                        val_postal_counter[addr_info["postal_token"]] += 1

    print(f"   Processed {total_val_s1:,} validation S1 records.\n")

    # -------------------------------------------------------------
    # Section 1: Business Name Normalization & Block Size Profiles
    # -------------------------------------------------------------
    print("---------------- SECTION 1: BLOCK SIZES -----------------")
    raw_stats = analyzer.compute_block_stats(val_name_raw_counter)
    norm_stats = analyzer.compute_block_stats(val_name_norm_counter)
    country_stats = analyzer.compute_block_stats(val_name_country_counter)

    print(f"{'Key Strategy':<25} | {'Unique Keys':<12} | {'Max Size':<10} | {'Median':<8} | {'p99 Size':<10} | {'>100 count':<10}")
    print("-" * 85)
    print(f"{'Raw Lowercase Name':<25} | {raw_stats['total_unique_blocks']:<12,} | {raw_stats['max_block_size']:<10,} | {raw_stats['median_block_size']:<8} | {raw_stats['p99_block_size']:<10} | {raw_stats['blocks_over_100']:<10}")
    print(f"{'Normalized Name':<25} | {norm_stats['total_unique_blocks']:<12,} | {norm_stats['max_block_size']:<10,} | {norm_stats['median_block_size']:<8} | {norm_stats['p99_block_size']:<10} | {norm_stats['blocks_over_100']:<10}")
    print(f"{'Norm Name + Country':<25} | {country_stats['total_unique_blocks']:<12,} | {country_stats['max_block_size']:<10,} | {country_stats['median_block_size']:<8} | {country_stats['p99_block_size']:<10} | {country_stats['blocks_over_100']:<10}")
    print("---------------------------------------------------------\n")

    # -------------------------------------------------------------
    # Section 2: High-Frequency (Dangerous) Token Analysis
    # -------------------------------------------------------------
    print("----------- SECTION 2: TOP HIGH-FREQUENCY TOKENS ---------")
    print("Top 10 High-Frequency Name Tokens (Risk of combinatorial explosion if used alone):")
    for tok, count in val_name_tokens_counter.most_common(10):
        pct = (count / total_val_s1) * 100
        print(f"   - '{tok}': {count:,} occurrences ({pct:.2f}% of validation entities)")

    print("\nTop 10 High-Frequency Address Tokens:")
    for tok, count in val_addr_tokens_counter.most_common(10):
        pct = (count / total_val_s1) * 100
        print(f"   - '{tok}': {count:,} occurrences ({pct:.2f}% of validation entities)")
    print("---------------------------------------------------------\n")

    # -------------------------------------------------------------
    # Section 3: Ground Truth Alignment on Validation Entities
    # -------------------------------------------------------------
    print("---------- SECTION 3: GROUND TRUTH KEY ALIGNMENT --------")
    print("Analyzing ground-truth pair alignment across candidate sources...")

    # Load validation ground truth
    all_gt = loader.load_ground_truth()
    val_gt = splitter.filter_ground_truth_for_split(all_gt, val_s1_ids)

    # Sample S2 and S3 entities corresponding to validation matches to test key sharing
    # Collect all needed candidate IDs
    needed_s2_ids: Set[str] = set()
    needed_s3_ids: Set[str] = set()

    sample_val_pairs: List[Tuple[str, str]] = []
    # Take a sample of 25,000 validation S1 entities with their true matches for fast evaluation
    sample_size = 25000
    sampled_s1_keys = list(val_gt.keys())[:sample_size]

    for s1_id in sampled_s1_keys:
        matched_ids = val_gt[s1_id]
        for m_id in matched_ids:
            sample_val_pairs.append((s1_id, m_id))
            if m_id.startswith("S2-"):
                needed_s2_ids.add(m_id)
            elif m_id.startswith("S3-"):
                needed_s3_ids.add(m_id)

    print(f"   Evaluating key overlap on {len(sample_val_pairs):,} ground-truth pairs ({len(sampled_s1_keys):,} S1 entities)...")

    # Stream S2 and S3 to grab entity data for needed target IDs
    s2_lookup: Dict[str, Tuple[str, str, str]] = {}
    s3_lookup: Dict[str, Tuple[str, str, str]] = {}

    s2_train_file = loader.get_file_path("train", "source2")
    with open(s2_train_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        _ = next(reader, None)
        for row in reader:
            if not row: continue
            eid = row[0].strip()
            if eid in needed_s2_ids:
                s2_lookup[eid] = (row[1].strip(), row[2].strip() if len(row) > 2 else "", row[3].strip() if len(row) > 3 else "")
                if len(s2_lookup) == len(needed_s2_ids):
                    break

    s3_train_file = loader.get_file_path("train", "source3")
    with open(s3_train_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        _ = next(reader, None)
        for row in reader:
            if not row: continue
            eid = row[0].strip()
            if eid in needed_s3_ids:
                s3_lookup[eid] = (row[1].strip(), row[2].strip() if len(row) > 2 else "", row[3].strip() if len(row) > 3 else "")
                if len(s3_lookup) == len(needed_s3_ids):
                    break

    # Measure key alignment
    exact_raw_name_match = 0
    exact_norm_name_match = 0
    exact_norm_name_country_match = 0
    same_country_match = 0
    at_least_1_token_match = 0
    at_least_2_tokens_match = 0
    valid_pair_count = 0

    for s1_id, m_id in sample_val_pairs:
        s1_data = val_s1_entities.get(s1_id)
        if not s1_data: continue

        m_data = s2_lookup.get(m_id) if m_id.startswith("S2-") else s3_lookup.get(m_id)
        if not m_data: continue

        valid_pair_count += 1
        s1_name, s1_addr, s1_country = s1_data
        m_name, m_addr, m_country = m_data

        # 1. Exact Country
        if s1_country.strip().lower() == m_country.strip().lower():
            same_country_match += 1

        # 2. Exact Raw Name
        if s1_name.strip().lower() == m_name.strip().lower():
            exact_raw_name_match += 1

        # 3. Exact Normalized Name
        s1_norm = normalize_text(s1_name, remove_legal_suffix=True)
        m_norm = normalize_text(m_name, remove_legal_suffix=True)
        if s1_norm == m_norm and s1_norm != "":
            exact_norm_name_match += 1
            if s1_country.strip().lower() == m_country.strip().lower():
                exact_norm_name_country_match += 1

        # 4. Token Overlap (ignoring legal suffixes and len <= 2)
        s1_tokens = set(extract_tokens(s1_name)) - LEGAL_SUFFIXES
        m_tokens = set(extract_tokens(m_name)) - LEGAL_SUFFIXES
        common_tokens = s1_tokens.intersection(m_tokens)

        if len(common_tokens) >= 1:
            at_least_1_token_match += 1
        if len(common_tokens) >= 2 or (len(s1_tokens) == 1 and len(common_tokens) == 1):
            at_least_2_tokens_match += 1

    print(f"Evaluated {valid_pair_count:,} ground-truth pairs:")
    print(f"  - Same Country:                     {same_country_match:,} ({(same_country_match/valid_pair_count)*100:.2f}%)")
    print(f"  - Exact Raw Name Match:             {exact_raw_name_match:,} ({(exact_raw_name_match/valid_pair_count)*100:.2f}%)")
    print(f"  - Exact Normalized Name Match:      {exact_norm_name_match:,} ({(exact_norm_name_match/valid_pair_count)*100:.2f}%)")
    print(f"  - Exact Norm Name + Country Match:  {exact_norm_name_country_match:,} ({(exact_norm_name_country_match/valid_pair_count)*100:.2f}%)")
    print(f"  - Share >= 1 Significant Name Token:{at_least_1_token_match:,} ({(at_least_1_token_match/valid_pair_count)*100:.2f}%)")
    print(f"  - Share >= 2 Name Tokens / Full 1:  {at_least_2_tokens_match:,} ({(at_least_2_tokens_match/valid_pair_count)*100:.2f}%)")
    print("---------------------------------------------------------\n")


if __name__ == "__main__":
    main()
