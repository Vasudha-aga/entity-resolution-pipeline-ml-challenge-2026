"""
Experimental B5 Address-Only Blocking Pass for Amazon ML Challenge 2026.

Implements and evaluates:
- B5-A: Building/House Number + Distinctive Locality/Address Token + Country
- B5-B: Two Distinctive Non-Generic Address Tokens + Country

Evaluates recall progression, candidate volume growth, and block size statistics
on the frozen 500-S1 smoke validation dataset.
"""

import os
import sys
import csv
import time
import json
import array
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Any, Optional
import numpy as np

from src.data_loader import DataLoader, Entity
from src.validation import ValidationSplitter
from src.normalization import (
    normalize_business_name,
    normalize_address,
    tokenize_business_name,
    extract_address_tokens,
    SINGLE_WORD_LEGAL_SUFFIXES
)
from src.blocking import CandidateGenerator, BlockingConfig

# Strict generic address stopwords/markers to exclude from distinctive token pairs
GENERIC_ADDR_TOKENS = {
    "road", "rd", "street", "st", "drive", "dr", "avenue", "ave", "lane", "ln",
    "unit", "floor", "fl", "suite", "ste", "apt", "apartment", "null", "box",
    "po", "no", "near", "opp", "opposite", "behind", "phase", "plot", "sector",
    "flat", "block", "house", "hno", "kh", "door", "st", "nd", "rd", "th",
    "east", "west", "north", "south", "city", "nagar", "colony", "bhavan",
    "complex", "building", "tower", "plaza", "market", "bazaar", "center", "centre",
    "delhi", "mumbai", "india", "state", "dist", "district", "post", "extn", "extension"
}


def load_raw_s1_entities(loader: DataLoader, target_s1_ids: Set[str]) -> Dict[str, Entity]:
    s1_file = loader.get_file_path("train", "source1")
    s1_entities: Dict[str, Entity] = {}
    with open(s1_file, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            if not line: continue
            tab_pos = line.find("\t")
            if tab_pos == -1: continue
            eid = line[:tab_pos].strip()
            if eid in target_s1_ids:
                row = line.rstrip("\r\n").split("\t")
                s1_entities[eid] = Entity.from_row(row)
                if len(s1_entities) >= len(target_s1_ids):
                    break
    return s1_entities


def load_smoke_ground_truth(file_path: str, target_s1_ids: Set[str]) -> Dict[str, List[str]]:
    gt_mapping: Dict[str, List[str]] = {}
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            if not line: continue
            tab_pos = line.find("\t")
            if tab_pos == -1: continue
            s1_id = line[:tab_pos].strip()
            if s1_id in target_s1_ids:
                matched_str = line[tab_pos + 1:].strip()
                gt_mapping[s1_id] = [m.strip() for m in matched_str.split(",") if m.strip()]
                if len(gt_mapping) == len(target_s1_ids):
                    break
    return gt_mapping


def build_smoke_target_subset(
    loader: DataLoader,
    source: str,
    required_ids: Set[str],
    min_target_size: int
) -> List[Entity]:
    target_file = loader.get_file_path("train", source)
    target_size = max(min_target_size, len(required_ids))

    selected_required: List[Entity] = []
    found_required_ids: Set[str] = set()
    distractors: List[Entity] = []

    with open(target_file, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            if not line: continue
            tab_pos = line.find("\t")
            if tab_pos == -1: continue
            eid = line[:tab_pos].strip()

            is_req = eid in required_ids
            has_enough = (len(selected_required) + len(distractors)) >= target_size

            if is_req:
                row = line.rstrip("\r\n").split("\t")
                selected_required.append(Entity.from_row(row))
                found_required_ids.add(eid)
            elif not has_enough:
                row = line.rstrip("\r\n").split("\t")
                distractors.append(Entity.from_row(row))

            if len(found_required_ids) == len(required_ids) and (len(selected_required) + len(distractors)) >= target_size:
                break

    return selected_required + distractors


# =========================================================================
# B5 ADDRESS KEY BUILDERS
# =========================================================================

def build_b5a_keys(address: Optional[str], country: Optional[str], max_nums: int = 2, max_toks: int = 3) -> List[str]:
    """
    B5-A: Building/House Number + Distinctive Locality Token + Country.
    Example: 'b5_num_68_rtnagar___india'
    """
    if not address or not isinstance(address, str) or not address.strip():
        return []

    norm_country = country.strip().lower() if country and isinstance(country, str) else ""
    tokens = extract_address_tokens(address, min_token_len=2)
    if not tokens:
        return []

    # Identify numeric and alphanumeric building tokens (e.g. '68', '273', '196a', 'cc54')
    num_tokens = []
    distinctive_tokens = []

    for t in tokens:
        if t.isdigit() or (len(t) >= 2 and any(c.isdigit() for c in t) and any(c.isalpha() for c in t)):
            if t not in num_tokens:
                num_tokens.append(t)
        elif len(t) >= 3 and t not in GENERIC_ADDR_TOKENS:
            if t not in distinctive_tokens:
                distinctive_tokens.append(t)

    selected_nums = num_tokens[:max_nums]
    selected_toks = distinctive_tokens[:max_toks]

    keys: List[str] = []
    for num in selected_nums:
        for tok in selected_toks:
            keys.append(f"b5_num_{num}_{tok}___{norm_country}")
    return keys


def build_b5b_keys(address: Optional[str], country: Optional[str], max_toks: int = 4) -> List[str]:
    """
    B5-B: Two Distinctive Non-Generic Address Tokens + Country.
    Example: 'b5_pair_beturkar_pada___india'
    """
    if not address or not isinstance(address, str) or not address.strip():
        return []

    norm_country = country.strip().lower() if country and isinstance(country, str) else ""
    tokens = extract_address_tokens(address, min_token_len=3)
    if not tokens:
        return []

    distinctive_tokens = []
    for t in tokens:
        if t not in GENERIC_ADDR_TOKENS and not t.isdigit():
            if t not in distinctive_tokens:
                distinctive_tokens.append(t)

    selected = distinctive_tokens[:max_toks]
    if len(selected) < 2:
        return []

    keys: List[str] = []
    # Generate unique sorted pairs
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            t1, t2 = sorted([selected[i], selected[j]])
            keys.append(f"b5_pair_{t1}_{t2}___{norm_country}")
    return keys


# =========================================================================
# EXPERIMENTAL B5 INDEX
# =========================================================================

class TargetIndexB5:
    """Experimental address-only inverted index for B5-A and B5-B."""

    def __init__(self, source_name: str, max_postings_threshold: int = 500):
        self.source_name = source_name
        self.max_postings_threshold = max_postings_threshold
        self.entity_ids: List[str] = []
        self.b5a_index: Dict[str, array.array] = defaultdict(lambda: array.array('I'))
        self.b5b_index: Dict[str, array.array] = defaultdict(lambda: array.array('I'))

    def add_batch(self, entities: List[Entity]) -> None:
        base_idx = len(self.entity_ids)
        for i, entity in enumerate(entities):
            idx = base_idx + i
            self.entity_ids.append(entity.entity_id)

            if entity.business_address:
                # B5-A
                for k in build_b5a_keys(entity.business_address, entity.country):
                    self.b5a_index[k].append(idx)
                # B5-B
                for k in build_b5b_keys(entity.business_address, entity.country):
                    self.b5b_index[k].append(idx)

    def prune_high_frequency_keys(self) -> Tuple[int, int]:
        """Prune keys with postings > max_postings_threshold."""
        pruned_a = 0
        keys_to_del_a = [k for k, v in self.b5a_index.items() if len(v) > self.max_postings_threshold]
        for k in keys_to_del_a:
            del self.b5a_index[k]
            pruned_a += 1

        pruned_b = 0
        keys_to_del_b = [k for k, v in self.b5b_index.items() if len(v) > self.max_postings_threshold]
        for k in keys_to_del_b:
            del self.b5b_index[k]
            pruned_b += 1

        return pruned_a, pruned_b

    def query_b5a(self, s1_entity: Entity) -> List[str]:
        keys = build_b5a_keys(s1_entity.business_address, s1_entity.country)
        if not keys: return []
        seen: Set[int] = set()
        cands: List[str] = []
        for k in keys:
            if k in self.b5a_index:
                for idx in self.b5a_index[k]:
                    if idx not in seen:
                        seen.add(idx)
                        cands.append(self.entity_ids[idx])
        return cands

    def query_b5b(self, s1_entity: Entity) -> List[str]:
        keys = build_b5b_keys(s1_entity.business_address, s1_entity.country)
        if not keys: return []
        seen: Set[int] = set()
        cands: List[str] = []
        for k in keys:
            if k in self.b5b_index:
                for idx in self.b5b_index[k]:
                    if idx not in seen:
                        seen.add(idx)
                        cands.append(self.entity_ids[idx])
        return cands


def get_block_size_stats(index_dict: Dict[str, array.array]) -> Dict[str, Any]:
    if not index_dict:
        return {"total_keys": 0, "max_block": 0, "median_block": 0, "p99_block": 0, "p95_block": 0}
    sizes = sorted([len(v) for v in index_dict.values()])
    n = len(sizes)
    return {
        "total_keys": n,
        "max_block": max(sizes) if sizes else 0,
        "median_block": float(np.median(sizes)) if sizes else 0,
        "p95_block": sizes[int(n * 0.95)] if n > 0 else 0,
        "p99_block": sizes[int(n * 0.99)] if n > 0 else 0
    }


def evaluate_candidate_dict(
    val_s1_entities: Dict[str, Entity],
    cand_dict: Dict[str, Set[str]],
    val_gt: Dict[str, List[str]]
) -> Dict[str, Any]:
    total_candidates = sum(len(cands) for cands in cand_dict.values())
    total_gt_s2 = 0
    retrieved_gt_s2 = 0
    total_gt_s3 = 0
    retrieved_gt_s3 = 0
    total_gt_all = 0
    retrieved_gt_all = 0

    s1_complete_count = 0
    total_s1 = len(val_s1_entities)
    retrieved_links_set: Set[Tuple[str, str]] = set()

    for s1_id in val_s1_entities:
        gt_matches = val_gt.get(s1_id, [])
        gt_s2 = [m for m in gt_matches if m.startswith("S2-")]
        gt_s3 = [m for m in gt_matches if m.startswith("S3-")]

        total_gt_s2 += len(gt_s2)
        total_gt_s3 += len(gt_s3)
        total_gt_all += len(gt_matches)

        cands = cand_dict.get(s1_id, set())

        retrieved_s2 = 0
        for m in gt_s2:
            if m in cands:
                retrieved_s2 += 1
                retrieved_links_set.add((s1_id, m))

        retrieved_s3 = 0
        for m in gt_s3:
            if m in cands:
                retrieved_s3 += 1
                retrieved_links_set.add((s1_id, m))

        retrieved_all = retrieved_s2 + retrieved_s3
        retrieved_gt_s2 += retrieved_s2
        retrieved_gt_s3 += retrieved_s3
        retrieved_gt_all += retrieved_all

        if retrieved_all == len(gt_matches):
            s1_complete_count += 1

    rec_s2 = (retrieved_gt_s2 / total_gt_s2) if total_gt_s2 > 0 else 1.0
    rec_s3 = (retrieved_gt_s3 / total_gt_s3) if total_gt_s3 > 0 else 1.0
    rec_all = (retrieved_gt_all / total_gt_all) if total_gt_all > 0 else 1.0
    s1_cov_pct = (s1_complete_count / total_s1 * 100) if total_s1 > 0 else 100.0

    return {
        "num_candidates": total_candidates,
        "total_gt_links": total_gt_all,
        "retrieved_gt_links": retrieved_gt_all,
        "s2_recall": float(np.round(rec_s2, 4)),
        "s3_recall": float(np.round(rec_s3, 4)),
        "combined_recall": float(np.round(rec_all, 4)),
        "s1_complete_coverage_count": s1_complete_count,
        "s1_complete_coverage_pct": float(np.round(s1_cov_pct, 2)),
        "retrieved_links_set": retrieved_links_set
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=========================================================")
    print(" Amazon ML Challenge 2026: Experimental B5 Blocking Test ")
    print("=========================================================\n")

    loader = DataLoader(data_dir="data")
    splitter = ValidationSplitter(data_loader=loader, seed=42)

    splits_dir = "experiments/splits"
    train_split_path = os.path.join(splits_dir, "train_s1_ids.txt")
    val_split_path = os.path.join(splits_dir, "val_s1_ids.txt")

    all_train_s1 = splitter.load_split_ids(train_split_path)
    all_val_s1 = splitter.load_split_ids(val_split_path)

    sample_s1_train = 2000
    sample_s1_val = 500
    smoke_target_size = 20000

    train_s1_subset = set(sorted(list(all_train_s1))[:sample_s1_train])
    val_s1_subset = set(sorted(list(all_val_s1))[:sample_s1_val])
    all_selected_s1 = train_s1_subset.union(val_s1_subset)

    gt_file = loader.get_file_path("train", "ground_truth")
    gt_mapping = load_smoke_ground_truth(gt_file, all_selected_s1)
    val_gt = splitter.filter_ground_truth_for_split(gt_mapping, val_s1_subset)

    required_s2_ids: Set[str] = set()
    required_s3_ids: Set[str] = set()
    for s1_id in all_selected_s1:
        for m_id in gt_mapping.get(s1_id, []):
            if m_id.startswith("S2-"):
                required_s2_ids.add(m_id)
            elif m_id.startswith("S3-"):
                required_s3_ids.add(m_id)

    print("1. Loading target subsets for S2 and S3...")
    s2_smoke_entities = build_smoke_target_subset(loader, "source2", required_s2_ids, min_target_size=smoke_target_size)
    s3_smoke_entities = build_smoke_target_subset(loader, "source3", required_s3_ids, min_target_size=smoke_target_size)

    # 2. Build standard B1-B4 CandidateGenerator
    print("2. Building baseline B1-B4 indexes...")
    cand_gen = CandidateGenerator(data_loader=loader, config=BlockingConfig())
    cand_gen.build_indexes_from_entities(s2_smoke_entities, s3_smoke_entities)

    # 3. Build Experimental B5 Indexes
    print("3. Building Experimental B5-A and B5-B address indexes...")
    t_b5_0 = time.perf_counter()
    b5_s2_index = TargetIndexB5("source2", max_postings_threshold=500)
    b5_s3_index = TargetIndexB5("source3", max_postings_threshold=500)

    b5_s2_index.add_batch(s2_smoke_entities)
    b5_s3_index.add_batch(s3_smoke_entities)

    pruned_s2_a, pruned_s2_b = b5_s2_index.prune_high_frequency_keys()
    pruned_s3_a, pruned_s3_b = b5_s3_index.prune_high_frequency_keys()
    t_b5_elapsed = time.perf_counter() - t_b5_0
    print(f"   B5 indexes built in {t_b5_elapsed:.2f}s (Pruned high-freq keys > 500: S2: {pruned_s2_a+pruned_s2_b}, S3: {pruned_s3_a+pruned_s3_b})")

    # Block size statistics
    stats_b5a_s2 = get_block_size_stats(b5_s2_index.b5a_index)
    stats_b5b_s2 = get_block_size_stats(b5_s2_index.b5b_index)
    stats_b5a_s3 = get_block_size_stats(b5_s3_index.b5a_index)
    stats_b5b_s3 = get_block_size_stats(b5_s3_index.b5b_index)

    print(f"   B5-A Block Sizes (S2): Max={stats_b5a_s2['max_block']}, p95={stats_b5a_s2['p95_block']}, p99={stats_b5a_s2['p99_block']}, Keys={stats_b5a_s2['total_keys']:,}")
    print(f"   B5-B Block Sizes (S2): Max={stats_b5b_s2['max_block']}, p95={stats_b5b_s2['p95_block']}, p99={stats_b5b_s2['p99_block']}, Keys={stats_b5b_s2['total_keys']:,}")

    # 4. Generate candidates for 500 validation S1 entities
    print("\n4. Generating candidates for 500 validation S1 entities across passes...")
    val_s1_entities = load_raw_s1_entities(loader, val_s1_subset)

    cands_b1_b4: Dict[str, Set[str]] = defaultdict(set)
    cands_b5a_only: Dict[str, Set[str]] = defaultdict(set)
    cands_b5b_only: Dict[str, Set[str]] = defaultdict(set)

    cands_with_b5a: Dict[str, Set[str]] = defaultdict(set)
    cands_with_b5a_b5b: Dict[str, Set[str]] = defaultdict(set)

    for s1_id, s1_entity in val_s1_entities.items():
        # Baseline B1-B4
        s2_b14, s3_b14 = cand_gen.generate_candidates(s1_entity)
        b14_set = set(s2_b14 + s3_b14)
        cands_b1_b4[s1_id] = b14_set

        # B5-A
        s2_b5a = b5_s2_index.query_b5a(s1_entity)
        s3_b5a = b5_s3_index.query_b5a(s1_entity)
        b5a_set = set(s2_b5a + s3_b5a)
        cands_b5a_only[s1_id] = b5a_set

        # B5-B
        s2_b5b = b5_s2_index.query_b5b(s1_entity)
        s3_b5b = b5_s3_index.query_b5b(s1_entity)
        b5b_set = set(s2_b5b + s3_b5b)
        cands_b5b_only[s1_id] = b5b_set

        # Cumulative stages
        cands_with_b5a[s1_id] = b14_set | b5a_set
        cands_with_b5a_b5b[s1_id] = b14_set | b5a_set | b5b_set

    # 5. Evaluate recall and candidate counts
    res_b14 = evaluate_candidate_dict(val_s1_entities, cands_b1_b4, val_gt)
    res_with_b5a = evaluate_candidate_dict(val_s1_entities, cands_with_b5a, val_gt)
    res_with_b5ab = evaluate_candidate_dict(val_s1_entities, cands_with_b5a_b5b, val_gt)

    # Calculate additional true links
    links_b14 = res_b14["retrieved_links_set"]
    links_with_b5a = res_with_b5a["retrieved_links_set"]
    links_with_b5ab = res_with_b5ab["retrieved_links_set"]

    new_links_b5a = len(links_with_b5a - links_b14)
    new_links_b5b = len(links_with_b5ab - links_with_b5a)
    total_new_links_b5 = len(links_with_b5ab - links_b14)

    total_gt = res_b14["total_gt_links"]
    new_cands_b5a = res_with_b5a["num_candidates"] - res_b14["num_candidates"]
    new_cands_b5b = res_with_b5ab["num_candidates"] - res_with_b5a["num_candidates"]
    total_new_cands_b5 = res_with_b5ab["num_candidates"] - res_b14["num_candidates"]

    # Print Comparison Table
    print("\n------------------------- B5 EXPERIMENTAL PROGRESSION TABLE -------------------------")
    print(f"{'PASS':<24} | {'CANDIDATES':<12} | {'S2 RECALL':<10} | {'S3 RECALL':<10} | {'COMBINED':<10} | {'S1 COVERAGE':<14} | {'NEW TRUE LINKS'}")
    print("-" * 110)
    print(f"{'B1+B2+B3+B4 (Baseline)':<24} | {res_b14['num_candidates']:<12,} | {res_b14['s2_recall']*100:<9.2f}% | {res_b14['s3_recall']*100:<9.2f}% | {res_b14['combined_recall']*100:<9.2f}% | {res_b14['s1_complete_coverage_pct']:<5.2f}% ({res_b14['s1_complete_coverage_count']}/500) | baseline ({len(links_b14):,})")
    print(f"{'B1..B4 + B5-A':<24} | {res_with_b5a['num_candidates']:<12,} | {res_with_b5a['s2_recall']*100:<9.2f}% | {res_with_b5a['s3_recall']*100:<9.2f}% | {res_with_b5a['combined_recall']*100:<9.2f}% | {res_with_b5a['s1_complete_coverage_pct']:<5.2f}% ({res_with_b5a['s1_complete_coverage_count']}/500) | +{new_links_b5a} (+{new_links_b5a/total_gt*100:.2f}%)")
    print(f"{'B1..B4 + B5-A + B5-B':<24} | {res_with_b5ab['num_candidates']:<12,} | {res_with_b5ab['s2_recall']*100:<9.2f}% | {res_with_b5ab['s3_recall']*100:<9.2f}% | {res_with_b5ab['combined_recall']*100:<9.2f}% | {res_with_b5ab['s1_complete_coverage_pct']:<5.2f}% ({res_with_b5ab['s1_complete_coverage_count']}/500) | +{new_links_b5b} (+{new_links_b5b/total_gt*100:.2f}%)")
    print("--------------------------------------------------------------------------------------\n")

    print("---------------------------- DETAILED B5 RECOVERY METRICS ----------------------------")
    print(f"Total True Ground Truth Links in Val:           {total_gt:,}")
    print(f"Baseline B1-B4 Retrieved Links:                 {len(links_b14):,} ({len(links_b14)/total_gt*100:.2f}%)")
    print(f"B5-A Standalone Candidates Generated:           {sum(len(c) for c in cands_b5a_only.values()):,}")
    print(f"B5-B Standalone Candidates Generated:           {sum(len(c) for c in cands_b5b_only.values()):,}")
    print(f"Net New Candidates Added by B5-A:               +{new_cands_b5a:,} (+{(new_cands_b5a/res_b14['num_candidates'])*100:.2f}%)")
    print(f"Net New Candidates Added by B5-B:               +{new_cands_b5b:,} (+{(new_cands_b5b/res_with_b5a['num_candidates'])*100:.2f}%)")
    print(f"Total Net New Candidates Added by B5 (A+B):     +{total_new_cands_b5:,} (+{(total_new_cands_b5/res_b14['num_candidates'])*100:.2f}%)")
    print(f"New True Links Recovered by B5-A:               +{new_links_b5a} (+{new_links_b5a/total_gt*100:.2f}% recall)")
    print(f"New True Links Recovered by B5-B:               +{new_links_b5b} (+{new_links_b5b/total_gt*100:.2f}% recall)")
    print(f"Total New True Links Recovered by B5:           +{total_new_links_b5} (+{total_new_links_b5/total_gt*100:.2f}% recall)")
    print(f"Final Total True Links Recovered (B1..B5):      {len(links_with_b5ab):,} / {total_gt:,} ({len(links_with_b5ab)/total_gt*100:.2f}%)")
    print(f"Remaining Missed True Links:                    {total_gt - len(links_with_b5ab):,} / {total_gt:,} ({(total_gt - len(links_with_b5ab))/total_gt*100:.2f}%)")
    print(f"Final S1 Complete-Candidate Coverage:           {res_with_b5ab['s1_complete_coverage_pct']:.2f}% ({res_with_b5ab['s1_complete_coverage_count']} / 500 S1 entities)")
    print("--------------------------------------------------------------------------------------\n")

    # Save JSON Report
    report = {
        "baseline_b1_b4": {
            "candidates": res_b14["num_candidates"],
            "s2_recall": res_b14["s2_recall"],
            "s3_recall": res_b14["s3_recall"],
            "combined_recall": res_b14["combined_recall"],
            "s1_complete_coverage_pct": res_b14["s1_complete_coverage_pct"],
            "s1_complete_coverage_count": res_b14["s1_complete_coverage_count"],
            "retrieved_links_count": len(links_b14)
        },
        "with_b5a": {
            "candidates": res_with_b5a["num_candidates"],
            "net_new_candidates": new_cands_b5a,
            "s2_recall": res_with_b5a["s2_recall"],
            "s3_recall": res_with_b5a["s3_recall"],
            "combined_recall": res_with_b5a["combined_recall"],
            "s1_complete_coverage_pct": res_with_b5a["s1_complete_coverage_pct"],
            "s1_complete_coverage_count": res_with_b5a["s1_complete_coverage_count"],
            "new_true_links_recovered": new_links_b5a,
            "total_retrieved_links": len(links_with_b5a)
        },
        "with_b5a_b5b": {
            "candidates": res_with_b5ab["num_candidates"],
            "net_new_candidates": total_new_cands_b5,
            "s2_recall": res_with_b5ab["s2_recall"],
            "s3_recall": res_with_b5ab["s3_recall"],
            "combined_recall": res_with_b5ab["combined_recall"],
            "s1_complete_coverage_pct": res_with_b5ab["s1_complete_coverage_pct"],
            "s1_complete_coverage_count": res_with_b5ab["s1_complete_coverage_count"],
            "new_true_links_recovered": total_new_links_b5,
            "total_retrieved_links": len(links_with_b5ab),
            "remaining_missed_links": total_gt - len(links_with_b5ab)
        },
        "b5_block_size_statistics": {
            "b5a_s2": stats_b5a_s2,
            "b5b_s2": stats_b5b_s2,
            "b5a_s3": stats_b5a_s3,
            "b5b_s3": stats_b5b_s3
        }
    }

    report_path = "experiments/blocking_b5_experiment.json"
    os.makedirs("experiments", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved experiment results to: {report_path}\n")


if __name__ == "__main__":
    main()
