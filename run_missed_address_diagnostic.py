"""
Diagnostic analysis of address overlap signals on missed ground-truth links.

Inspects the 200 missed links (and specifically the 142 address/script-divergence links)
from B1-B4 blocking to determine exact address token overlap, numeric overlap,
postal code overlap, and locality sharing.
"""

import os
import sys
import csv
import json
from collections import Counter, defaultdict
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

# Common generic address stopwords/markers
GENERIC_ADDR_TOKENS = {
    "road", "rd", "street", "st", "drive", "dr", "avenue", "ave", "lane", "ln",
    "unit", "floor", "fl", "suite", "ste", "apt", "apartment", "null", "box",
    "po", "no", "near", "opp", "opposite", "behind", "phase", "plot", "sector",
    "flat", "block", "house", "hno", "kh", "door", "st", "nd", "rd", "th",
    "east", "west", "north", "south", "city", "nagar", "colony", "bhavan",
    "complex", "building", "tower", "plaza", "market", "bazaar", "center", "centre"
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


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=========================================================")
    print(" Amazon ML Challenge 2026: Missed Address Link Diagnostic")
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

    print("Loading target subsets and building blocking indexes...")
    s2_smoke_entities = build_smoke_target_subset(loader, "source2", required_s2_ids, min_target_size=smoke_target_size)
    s3_smoke_entities = build_smoke_target_subset(loader, "source3", required_s3_ids, min_target_size=smoke_target_size)

    s2_lookup = {e.entity_id: e for e in s2_smoke_entities}
    s3_lookup = {e.entity_id: e for e in s3_smoke_entities}

    # Compute global document frequencies of address tokens across the target pools
    target_addr_token_df = Counter()
    for e in s2_smoke_entities + s3_smoke_entities:
        if e.business_address:
            toks = set(extract_address_tokens(e.business_address, min_token_len=2))
            for t in toks:
                target_addr_token_df[t] += 1

    cand_gen = CandidateGenerator(data_loader=loader, config=BlockingConfig())
    cand_gen.build_indexes_from_entities(s2_smoke_entities, s3_smoke_entities)

    val_s1_entities = load_raw_s1_entities(loader, val_s1_subset)

    # Generate B1-B4 cumulative candidates
    cands_b1_b4: Dict[str, Set[str]] = defaultdict(set)
    for s1_id, s1_entity in val_s1_entities.items():
        s2_cands, s3_cands = cand_gen.generate_candidates(s1_entity)
        cands_b1_b4[s1_id] = set(s2_cands + s3_cands)

    # Collect all ground truth links and find missed ones
    all_val_gt_links: List[Tuple[str, str]] = []
    missed_links: List[Tuple[str, str]] = []

    for s1_id, m_list in val_gt.items():
        for m_id in m_list:
            all_val_gt_links.append((s1_id, m_id))
            if m_id not in cands_b1_b4[s1_id]:
                missed_links.append((s1_id, m_id))

    print(f"Total True Validation Links: {len(all_val_gt_links):,}")
    print(f"Retrieved Links:             {len(all_val_gt_links) - len(missed_links):,} ({((len(all_val_gt_links)-len(missed_links))/len(all_val_gt_links))*100:.2f}%)")
    print(f"Missed Links to Analyze:     {len(missed_links):,} ({len(missed_links)/len(all_val_gt_links)*100:.2f}%)\n")

    # Detailed inspection of each missed link
    detailed_missed_records = []

    shared_postal_count = 0
    shared_building_num_count = 0
    shared_rare_token_count = 0
    shared_only_common_count = 0
    no_usable_overlap_count = 0

    useful_token_freq = Counter()

    for s1_id, m_id in missed_links:
        e1 = val_s1_entities[s1_id]
        e2 = s2_lookup.get(m_id) if m_id.startswith("S2-") else s3_lookup.get(m_id)

        addr1 = e1.business_address or ""
        addr2 = (e2.business_address or "") if e2 else ""

        norm_addr1 = normalize_address(addr1)
        norm_addr2 = normalize_address(addr2)

        toks1 = extract_address_tokens(addr1, min_token_len=2)
        toks2 = extract_address_tokens(addr2, min_token_len=2)

        set1 = set(toks1)
        set2 = set(toks2)

        shared_tokens = sorted(list(set1 & set2))

        # Numeric tokens (postal codes vs building/street numbers)
        shared_nums = [t for t in shared_tokens if t.isdigit()]
        shared_postal = [t for t in shared_nums if len(t) in (5, 6)]
        shared_bldg = [t for t in shared_nums if len(t) < 5 or (len(t) >= 1 and not len(t) in (5, 6))]

        # Rare address tokens: target doc freq <= 100 in target subset and len >= 3 and not generic
        shared_rare = [
            t for t in shared_tokens
            if t not in GENERIC_ADDR_TOKENS and len(t) >= 3 and (target_addr_token_df[t] <= 100)
        ]

        shared_distinctive = [
            t for t in shared_tokens
            if t not in GENERIC_ADDR_TOKENS and len(t) >= 3
        ]

        for t in shared_distinctive:
            useful_token_freq[t] += 1
        for p in shared_postal:
            useful_token_freq[f"POSTAL:{p}"] += 1
        for b in shared_bldg:
            useful_token_freq[f"NUM:{b}"] += 1

        # Classify overlap nature
        has_postal = len(shared_postal) > 0
        has_bldg = len(shared_bldg) > 0
        has_rare = len(shared_rare) > 0
        has_any_overlap = len(shared_tokens) > 0

        if has_postal:
            shared_postal_count += 1
        if has_bldg:
            shared_building_num_count += 1
        if has_rare:
            shared_rare_token_count += 1

        if not has_any_overlap or (not norm_addr1 or not norm_addr2):
            no_usable_overlap_count += 1
        elif not has_postal and not has_bldg and not has_rare:
            shared_only_common_count += 1

        record = {
            "s1_id": s1_id,
            "target_id": m_id,
            "s1_name": e1.business_name,
            "target_name": e2.business_name if e2 else "",
            "s1_address": addr1,
            "target_address": addr2,
            "country": e1.country,
            "s1_addr_tokens": toks1,
            "target_addr_tokens": toks2,
            "shared_address_tokens": shared_tokens,
            "shared_numeric_tokens": shared_nums,
            "shared_postal_codes": shared_postal,
            "shared_building_numbers": shared_bldg,
            "shared_distinctive_tokens": shared_distinctive,
            "shared_rare_tokens": shared_rare,
            "has_shared_postal": has_postal,
            "has_shared_building_number": has_bldg,
            "has_shared_rare_token": has_rare
        }
        detailed_missed_records.append(record)

    total_missed = len(missed_links)

    print("-------------------- AGGREGATE ADDRESS OVERLAP FINDINGS --------------------")
    print(f"Total Missed True Links Analyzed:                {total_missed:>4,}")
    print(f"  1. Share an exact postal code (5/6 digits):    {shared_postal_count:>4,} ({shared_postal_count/total_missed*100:>5.2f}%)")
    print(f"  2. Share a building/house/street number:       {shared_building_num_count:>4,} ({shared_building_num_count/total_missed*100:>5.2f}%)")
    print(f"  3. Share a rare/distinctive address token:     {shared_rare_token_count:>4,} ({shared_rare_token_count/total_missed*100:>5.2f}%)")
    print(f"  4. Share a postal code OR building number:     {sum(1 for r in detailed_missed_records if r['has_shared_postal'] or r['has_shared_building_number']):>4,} ({sum(1 for r in detailed_missed_records if r['has_shared_postal'] or r['has_shared_building_number'])/total_missed*100:>5.2f}%)")
    print(f"  5. Share postal code OR building OR rare token:{sum(1 for r in detailed_missed_records if r['has_shared_postal'] or r['has_shared_building_number'] or r['has_shared_rare_token']):>4,} ({sum(1 for r in detailed_missed_records if r['has_shared_postal'] or r['has_shared_building_number'] or r['has_shared_rare_token'])/total_missed*100:>5.2f}%)")
    print(f"  6. Share ONLY common/generic address tokens:   {shared_only_common_count:>4,} ({shared_only_common_count/total_missed*100:>5.2f}%)")
    print(f"  7. No usable address overlap (blank/disjoint): {no_usable_overlap_count:>4,} ({no_usable_overlap_count/total_missed*100:>5.2f}%)")
    print("----------------------------------------------------------------------------\n")

    print("Top 20 Most Frequent Distinctive Shared Address Signals Among Missed Links:")
    for sig, count in useful_token_freq.most_common(20):
        print(f"   - {sig:<24}: {count:>3} occurrences")
    print("----------------------------------------------------------------------------\n")

    print("=================== 20 REPRESENTATIVE MISSED EXAMPLES ===================")
    for idx, r in enumerate(detailed_missed_records[:20], 1):
        print(f"\n[{idx:02d}] S1 ID: {r['s1_id']}  -->  Target ID: {r['target_id']} (Country: {r['country']})")
        print(f"     S1 Name:     {r['s1_name']}")
        print(f"     Target Name: {r['target_name']}")
        print(f"     S1 Address:     {r['s1_address']}")
        print(f"     Target Address: {r['target_address']}")
        print(f"     >>> Shared Postal:     {r['shared_postal_codes']}")
        print(f"     >>> Shared Numbers:    {r['shared_building_numbers']}")
        print(f"     >>> Shared Distinctive:{r['shared_distinctive_tokens']}")
        print(f"     >>> Shared Rare Tokens:{r['shared_rare_tokens']}")
    print("\n=========================================================================\n")

    # Save JSON Diagnostic Report
    report = {
        "summary": {
            "total_missed_links": total_missed,
            "shared_postal_code_count": shared_postal_count,
            "shared_postal_code_pct": round(shared_postal_count / total_missed * 100, 2),
            "shared_building_number_count": shared_building_num_count,
            "shared_building_number_pct": round(shared_building_num_count / total_missed * 100, 2),
            "shared_rare_address_token_count": shared_rare_token_count,
            "shared_rare_address_token_pct": round(shared_rare_token_count / total_missed * 100, 2),
            "shared_postal_or_building_count": sum(1 for r in detailed_missed_records if r['has_shared_postal'] or r['has_shared_building_number']),
            "shared_postal_or_building_pct": round(sum(1 for r in detailed_missed_records if r['has_shared_postal'] or r['has_shared_building_number']) / total_missed * 100, 2),
            "shared_postal_or_building_or_rare_count": sum(1 for r in detailed_missed_records if r['has_shared_postal'] or r['has_shared_building_number'] or r['has_shared_rare_token']),
            "shared_postal_or_building_or_rare_pct": round(sum(1 for r in detailed_missed_records if r['has_shared_postal'] or r['has_shared_building_number'] or r['has_shared_rare_token']) / total_missed * 100, 2),
            "shared_only_common_tokens_count": shared_only_common_count,
            "shared_only_common_tokens_pct": round(shared_only_common_count / total_missed * 100, 2),
            "no_usable_address_overlap_count": no_usable_overlap_count,
            "no_usable_address_overlap_pct": round(no_usable_overlap_count / total_missed * 100, 2)
        },
        "top_useful_shared_signals": dict(useful_token_freq.most_common(50)),
        "detailed_missed_records": detailed_missed_records
    }

    report_path = "experiments/missed_address_link_analysis.json"
    os.makedirs("experiments", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved complete diagnostic report to: {report_path}\n")


if __name__ == "__main__":
    main()
