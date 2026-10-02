"""
Diagnostic script for B1-B4 Blocking Pass Analysis on Amazon ML Challenge 2026.

Evaluates individual and cumulative blocking passes (B1, B1+B2, B1+B2+B3, B1+B2+B3+B4)
on the exact smoke-test validation set (500 S1 entities, same seed/split, same target subset).
Computes:
- Link-level S2 recall, S3 recall, combined recall
- S1 complete-candidate coverage
- Number of candidates generated
- Additional true links contributed by each pass
- Categorical error analysis on missed true links
"""

import os
import sys
import csv
import time
import json
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
    TextNormalizer,
    SINGLE_WORD_LEGAL_SUFFIXES
)
from src.blocking import CandidateGenerator, BlockingConfig


def load_raw_s1_entities(loader: DataLoader, target_s1_ids: Set[str]) -> Dict[str, Entity]:
    """Fast load raw S1 Entity objects for requested target S1 IDs."""
    s1_file = loader.get_file_path("train", "source1")
    s1_entities: Dict[str, Entity] = {}
    with open(s1_file, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            if not line:
                continue
            tab_pos = line.find("\t")
            if tab_pos == -1:
                continue
            eid = line[:tab_pos].strip()
            if eid in target_s1_ids:
                row = line.rstrip("\r\n").split("\t")
                s1_entities[eid] = Entity.from_row(row)
                if len(s1_entities) >= len(target_s1_ids):
                    break
    return s1_entities


def load_smoke_ground_truth(file_path: str, target_s1_ids: Set[str]) -> Dict[str, List[str]]:
    """Fast load ground truth mappings for specified S1 entity IDs only."""
    gt_mapping: Dict[str, List[str]] = {}
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            if not line:
                continue
            tab_pos = line.find("\t")
            if tab_pos == -1:
                continue
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
    """
    Build a deterministic small target subset of S2 or S3 entities for smoke testing.
    """
    target_file = loader.get_file_path("train", source)
    target_size = max(min_target_size, len(required_ids))

    selected_required: List[Entity] = []
    found_required_ids: Set[str] = set()
    distractors: List[Entity] = []

    with open(target_file, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            if not line:
                continue
            tab_pos = line.find("\t")
            if tab_pos == -1:
                continue
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


def evaluate_stage_recall(
    val_s1_entities: Dict[str, Entity],
    candidate_dict: Dict[str, Set[str]],
    val_gt: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Evaluate candidate recall and S1 complete-coverage metrics for a specific candidate set.
    """
    total_candidates = sum(len(cands) for cands in candidate_dict.values())

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

        cands = candidate_dict.get(s1_id, set())

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
        "total_gt_s2_links": total_gt_s2,
        "retrieved_gt_s2_links": retrieved_gt_s2,
        "s2_recall": float(np.round(rec_s2, 4)),
        "total_gt_s3_links": total_gt_s3,
        "retrieved_gt_s3_links": retrieved_gt_s3,
        "s3_recall": float(np.round(rec_s3, 4)),
        "total_gt_links": total_gt_all,
        "retrieved_gt_links": retrieved_gt_all,
        "combined_recall": float(np.round(rec_all, 4)),
        "s1_complete_coverage_count": s1_complete_count,
        "s1_complete_coverage_pct": float(np.round(s1_cov_pct, 2)),
        "retrieved_links_set": retrieved_links_set
    }


def char_ngrams(s: str, n: int = 3) -> Set[str]:
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def categorize_missed_link(e1: Entity, e2: Entity) -> str:
    """
    Deterministically categorize why a true match pair was not retrieved by B1-B4.
    """
    c1 = e1.country.strip().lower()
    c2 = e2.country.strip().lower()
    if c1 != c2:
        return "country-related issue"

    norm1 = normalize_business_name(e1.business_name, remove_legal_suffix=False)
    norm2 = normalize_business_name(e2.business_name, remove_legal_suffix=False)

    norm_no_legal1 = normalize_business_name(e1.business_name, remove_legal_suffix=True)
    norm_no_legal2 = normalize_business_name(e2.business_name, remove_legal_suffix=True)

    # 1. Transliteration / Unicode differences
    if e1.business_name != e2.business_name and norm_no_legal1 == norm_no_legal2:
        return "transliteration"

    # 2. Legal suffix variation
    if norm1 != norm2 and norm_no_legal1 == norm_no_legal2:
        return "legal suffix variation"

    t1 = tokenize_business_name(e1.business_name, remove_stopwords=False, min_token_len=2)
    t2 = tokenize_business_name(e2.business_name, remove_stopwords=False, min_token_len=2)
    s1, s2 = set(t1), set(t2)

    # 3. Word-order variation
    if s1 and s2 and s1 == s2 and t1 != t2:
        return "word-order variation"

    # 4. Abbreviation
    if t1 and t2:
        p1, p2 = t1[0], t2[0]
        if (p1.startswith(p2) or p2.startswith(p1)) and p1 != p2:
            return "abbreviation"
        # Check acronym
        acr1 = "".join(w[0] for w in t1 if w)
        acr2 = "".join(w[0] for w in t2 if w)
        if acr1 in t2 or acr2 in t1:
            return "abbreviation"

    # 5. Typo / Near-duplicate spelling
    ng1 = char_ngrams(norm_no_legal1, 3)
    ng2 = char_ngrams(norm_no_legal2, 3)
    union = len(ng1 | ng2)
    ngram_sim = (len(ng1 & ng2) / union) if union > 0 else 0.0

    if ngram_sim >= 0.65:
        return "typo"

    # 6. Generic name
    common_generic = {"care", "health", "center", "group", "services", "partners", "associates", "clinic", "hospital"}
    if s1.intersection(common_generic) and (len(s1) <= 2 or len(s2) <= 2):
        return "generic name"

    # 7. Missing address vs Address variation
    addr1 = normalize_address(e1.business_address)
    addr2 = normalize_address(e2.business_address)
    if not addr1 or not addr2:
        return "missing address"

    a_toks1 = set(extract_address_tokens(e1.business_address, min_token_len=3))
    a_toks2 = set(extract_address_tokens(e2.business_address, min_token_len=3))
    a_union = len(a_toks1 | a_toks2)
    a_sim = (len(a_toks1 & a_toks2) / a_union) if a_union > 0 else 0.0

    if a_sim >= 0.40:
        return "address variation"

    # 8. Exact / Name normalization failure
    if len(s1 & s2) == 0:
        return "exact/name normalization failure"

    return "unknown"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=========================================================")
    print(" Amazon ML Challenge 2026: B1-B4 Blocking Pass Analysis  ")
    print("=========================================================\n")

    loader = DataLoader(data_dir="data")
    splitter = ValidationSplitter(data_loader=loader, seed=42)

    splits_dir = "experiments/splits"
    train_split_path = os.path.join(splits_dir, "train_s1_ids.txt")
    val_split_path = os.path.join(splits_dir, "val_s1_ids.txt")

    all_train_s1 = splitter.load_split_ids(train_split_path)
    all_val_s1 = splitter.load_split_ids(val_split_path)

    # Smoke test subset: 2,000 train S1, 500 val S1
    sample_s1_train = 2000
    sample_s1_val = 500
    smoke_target_size = 20000

    train_s1_subset = set(sorted(list(all_train_s1))[:sample_s1_train])
    val_s1_subset = set(sorted(list(all_val_s1))[:sample_s1_val])
    all_selected_s1 = train_s1_subset.union(val_s1_subset)

    # Load ground truth for selected S1 entities ONLY
    gt_file = loader.get_file_path("train", "ground_truth")
    gt_mapping = load_smoke_ground_truth(gt_file, all_selected_s1)
    val_gt = splitter.filter_ground_truth_for_split(gt_mapping, val_s1_subset)

    # Required target IDs
    required_s2_ids: Set[str] = set()
    required_s3_ids: Set[str] = set()
    for s1_id in all_selected_s1:
        for m_id in gt_mapping.get(s1_id, []):
            if m_id.startswith("S2-"):
                required_s2_ids.add(m_id)
            elif m_id.startswith("S3-"):
                required_s3_ids.add(m_id)

    print(f"Loading smoke test target subsets (S2 required: {len(required_s2_ids):,}, S3 required: {len(required_s3_ids):,})...")
    s2_smoke_entities = build_smoke_target_subset(loader, "source2", required_s2_ids, min_target_size=smoke_target_size)
    s3_smoke_entities = build_smoke_target_subset(loader, "source3", required_s3_ids, min_target_size=smoke_target_size)

    s2_lookup = {e.entity_id: e for e in s2_smoke_entities}
    s3_lookup = {e.entity_id: e for e in s3_smoke_entities}

    # Build inverted indexes using CandidateGenerator
    print("Building CandidateGenerator indexes...")
    cand_gen = CandidateGenerator(data_loader=loader, config=BlockingConfig())
    build_times = cand_gen.build_indexes_from_entities(s2_smoke_entities, s3_smoke_entities)
    print(f"Indexes built: S2 in {build_times['source2']:.2f}s | S3 in {build_times['source3']:.2f}s\n")

    # Load validation S1 entities
    val_s1_entities = load_raw_s1_entities(loader, val_s1_subset)

    # Containers for pass queries
    cands_b1: Dict[str, Set[str]] = defaultdict(set)
    cands_b2: Dict[str, Set[str]] = defaultdict(set)
    cands_b3: Dict[str, Set[str]] = defaultdict(set)
    cands_b4: Dict[str, Set[str]] = defaultdict(set)

    cands_cum_12: Dict[str, Set[str]] = defaultdict(set)
    cands_cum_123: Dict[str, Set[str]] = defaultdict(set)
    cands_cum_1234: Dict[str, Set[str]] = defaultdict(set)

    print("Querying B1, B2, B3, B4 passes for 500 validation S1 entities...")
    for s1_id, s1_entity in val_s1_entities.items():
        passes = cand_gen.generate_candidates_passes(s1_entity)
        s2_b1, s3_b1 = passes["b1"]
        s2_b2, s3_b2 = passes["b2"]
        s2_b3, s3_b3 = passes["b3"]
        s2_b4, s3_b4 = passes["b4"]

        s2_12, s3_12 = passes["cumulative_b1_b2"]
        s2_123, s3_123 = passes["cumulative_b1_b2_b3"]
        s2_1234, s3_1234 = passes["cumulative"]

        cands_b1[s1_id] = set(s2_b1 + s3_b1)
        cands_b2[s1_id] = set(s2_b2 + s3_b2)
        cands_b3[s1_id] = set(s2_b3 + s3_b3)
        cands_b4[s1_id] = set(s2_b4 + s3_b4)

        cands_cum_12[s1_id] = set(s2_12 + s3_12)
        cands_cum_123[s1_id] = set(s2_123 + s3_123)
        cands_cum_1234[s1_id] = set(s2_1234 + s3_1234)

    # Evaluate each stage
    res_b1 = evaluate_stage_recall(val_s1_entities, cands_b1, val_gt)
    res_cum_12 = evaluate_stage_recall(val_s1_entities, cands_cum_12, val_gt)
    res_cum_123 = evaluate_stage_recall(val_s1_entities, cands_cum_123, val_gt)
    res_cum_1234 = evaluate_stage_recall(val_s1_entities, cands_cum_1234, val_gt)

    # Calculate additional true links contributed by each pass
    set_b1 = res_b1["retrieved_links_set"]
    set_cum12 = res_cum_12["retrieved_links_set"]
    set_cum123 = res_cum_123["retrieved_links_set"]
    set_cum1234 = res_cum_1234["retrieved_links_set"]

    new_b1 = len(set_b1)
    new_b2 = len(set_cum12 - set_b1)
    new_b3 = len(set_cum123 - set_cum12)
    new_b4 = len(set_cum1234 - set_cum123)

    # Identify all true ground-truth links for validation S1 entities
    all_val_gt_links: Set[Tuple[str, str]] = set()
    for s1_id, m_list in val_gt.items():
        for m_id in m_list:
            all_val_gt_links.add((s1_id, m_id))

    missed_links = all_val_gt_links - set_cum1234

    # Identify incomplete S1 entities
    incomplete_s1_ids = []
    for s1_id in val_s1_entities:
        gt_matches = set(val_gt.get(s1_id, []))
        cands = cands_cum_1234.get(s1_id, set())
        if not gt_matches.issubset(cands):
            incomplete_s1_ids.append(s1_id)

    # Categorize missed true links
    missed_category_counts = Counter()
    missed_samples_by_category = defaultdict(list)

    for s1_id, m_id in missed_links:
        e1 = val_s1_entities.get(s1_id)
        e2 = s2_lookup.get(m_id) if m_id.startswith("S2-") else s3_lookup.get(m_id)
        if e1 and e2:
            cat = categorize_missed_link(e1, e2)
            missed_category_counts[cat] += 1
            if len(missed_samples_by_category[cat]) < 3:
                missed_samples_by_category[cat].append({
                    "s1_id": s1_id,
                    "target_id": m_id,
                    "s1_name": e1.business_name,
                    "target_name": e2.business_name,
                    "s1_addr": e1.business_address,
                    "target_addr": e2.business_address,
                    "country": e1.country
                })

    # Print Summary Table
    print("------------------------- BLOCKING PASS PROGRESSION -------------------------")
    print(f"{'PASS':<16} {'CANDIDATES':<12} {'S2 RECALL':<12} {'S3 RECALL':<12} {'COMBINED':<12} {'S1 COVERAGE':<14}")
    print("-" * 80)
    print(f"{'B1':<16} {res_b1['num_candidates']:<12,} {res_b1['s2_recall']*100:<11.2f}% {res_b1['s3_recall']*100:<11.2f}% {res_b1['combined_recall']*100:<11.2f}% {res_b1['s1_complete_coverage_pct']:<5.2f}% ({res_b1['s1_complete_coverage_count']}/{len(val_s1_entities)})")
    print(f"{'B1+B2':<16} {res_cum_12['num_candidates']:<12,} {res_cum_12['s2_recall']*100:<11.2f}% {res_cum_12['s3_recall']*100:<11.2f}% {res_cum_12['combined_recall']*100:<11.2f}% {res_cum_12['s1_complete_coverage_pct']:<5.2f}% ({res_cum_12['s1_complete_coverage_count']}/{len(val_s1_entities)})")
    print(f"{'B1+B2+B3':<16} {res_cum_123['num_candidates']:<12,} {res_cum_123['s2_recall']*100:<11.2f}% {res_cum_123['s3_recall']*100:<11.2f}% {res_cum_123['combined_recall']*100:<11.2f}% {res_cum_123['s1_complete_coverage_pct']:<5.2f}% ({res_cum_123['s1_complete_coverage_count']}/{len(val_s1_entities)})")
    print(f"{'B1+B2+B3+B4':<16} {res_cum_1234['num_candidates']:<12,} {res_cum_1234['s2_recall']*100:<11.2f}% {res_cum_1234['s3_recall']*100:<11.2f}% {res_cum_1234['combined_recall']*100:<11.2f}% {res_cum_1234['s1_complete_coverage_pct']:<5.2f}% ({res_cum_1234['s1_complete_coverage_count']}/{len(val_s1_entities)})")
    print("-----------------------------------------------------------------------------\n")

    print("---------------------- ADDITIONAL TRUE LINKS RECOVERED ----------------------")
    print(f"  - B1 (Exact Name + Country) adds:       {new_b1:>5,} new true links ({new_b1/len(all_val_gt_links)*100:.2f}% of total GT)")
    print(f"  - B2 (Distinctive Tokens) adds:         {new_b2:>5,} new true links ({new_b2/len(all_val_gt_links)*100:.2f}% of total GT)")
    print(f"  - B3 (Name Prefix 4-gram) adds:         {new_b3:>5,} new true links ({new_b3/len(all_val_gt_links)*100:.2f}% of total GT)")
    print(f"  - B4 (Name + Address Token) adds:       {new_b4:>5,} new true links ({new_b4/len(all_val_gt_links)*100:.2f}% of total GT)")
    print(f"  - Total Retrieved Links:                {len(set_cum1234):>5,} / {len(all_val_gt_links):,} ({len(set_cum1234)/len(all_val_gt_links)*100:.2f}%)")
    print(f"  - Total Missed Links remaining:         {len(missed_links):>5,} / {len(all_val_gt_links):,} ({len(missed_links)/len(all_val_gt_links)*100:.2f}%)")
    print(f"  - Incomplete Validation S1 entities:    {len(incomplete_s1_ids):>5,} / {len(val_s1_entities):,} ({len(incomplete_s1_ids)/len(val_s1_entities)*100:.2f}%)")
    print("-----------------------------------------------------------------------------\n")

    print("------------------- MISSED TRUE LINKS FAILURE CATEGORIES -------------------")
    for cat, count in missed_category_counts.most_common():
        pct = (count / len(missed_links)) * 100 if missed_links else 0.0
        print(f"  - {cat:<32}: {count:>4,} missed links ({pct:>5.2f}%)")
    print("-----------------------------------------------------------------------------\n")

    # Save JSON Report
    report = {
        "evaluation_scope": {
            "validation_s1_entities": len(val_s1_entities),
            "total_ground_truth_links": len(all_val_gt_links),
            "smoke_target_s2_size": len(s2_smoke_entities),
            "smoke_target_s3_size": len(s3_smoke_entities)
        },
        "pass_progression": {
            "B1": {
                "candidates": res_b1["num_candidates"],
                "s2_recall": res_b1["s2_recall"],
                "s3_recall": res_b1["s3_recall"],
                "combined_recall": res_b1["combined_recall"],
                "s1_complete_coverage_pct": res_b1["s1_complete_coverage_pct"],
                "s1_complete_coverage_count": res_b1["s1_complete_coverage_count"],
                "additional_links_recovered": new_b1
            },
            "B1_B2": {
                "candidates": res_cum_12["num_candidates"],
                "s2_recall": res_cum_12["s2_recall"],
                "s3_recall": res_cum_12["s3_recall"],
                "combined_recall": res_cum_12["combined_recall"],
                "s1_complete_coverage_pct": res_cum_12["s1_complete_coverage_pct"],
                "s1_complete_coverage_count": res_cum_12["s1_complete_coverage_count"],
                "additional_links_recovered": new_b2
            },
            "B1_B2_B3": {
                "candidates": res_cum_123["num_candidates"],
                "s2_recall": res_cum_123["s2_recall"],
                "s3_recall": res_cum_123["s3_recall"],
                "combined_recall": res_cum_123["combined_recall"],
                "s1_complete_coverage_pct": res_cum_123["s1_complete_coverage_pct"],
                "s1_complete_coverage_count": res_cum_123["s1_complete_coverage_count"],
                "additional_links_recovered": new_b3
            },
            "B1_B2_B3_B4": {
                "candidates": res_cum_1234["num_candidates"],
                "s2_recall": res_cum_1234["s2_recall"],
                "s3_recall": res_cum_1234["s3_recall"],
                "combined_recall": res_cum_1234["combined_recall"],
                "s1_complete_coverage_pct": res_cum_1234["s1_complete_coverage_pct"],
                "s1_complete_coverage_count": res_cum_1234["s1_complete_coverage_count"],
                "additional_links_recovered": new_b4
            }
        },
        "missed_links_summary": {
            "total_missed_links": len(missed_links),
            "incomplete_s1_entities": len(incomplete_s1_ids),
            "failure_category_counts": dict(missed_category_counts),
            "sample_missed_links_by_category": dict(missed_samples_by_category)
        }
    }

    report_path = "experiments/blocking_pass_analysis.json"
    os.makedirs("experiments", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved diagnostic report to: {report_path}\n")


if __name__ == "__main__":
    main()
