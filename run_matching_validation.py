"""
Execution runner for Pairwise Matching Model & Validation Pipeline.

Pipeline Steps:
1. Loads S1 train/validation split (experiments/splits/train_s1_ids.txt & val_s1_ids.txt).
2. Loads ground truth mappings.
3. Pre-normalizes entity records in memory using src.normalization.
4. Generates candidate pairs for training and validation.
5. Evaluates candidate blocking recall and complete-candidate coverage.
6. Extracts 25-dimensional pairwise feature vectors in memory-safe batches.
7. Evaluates RuleBasedBaseline benchmark.
8. Trains EntityMatchingModel on train split entities only.
9. Evaluates predictions on validation split entities.
10. Runs decision threshold experiments (0.30 to 0.98) selecting best validation S1-level F0.5.
11. Evaluates S1 cardinality performance (zero-match, single-match, multi-match).
12. Performs Error Analysis with deterministic categorical tagging.
13. Saves JSON report to experiments/matching_validation_report.json and prints summary.

Supports --smoke-test flag for fast, memory-safe end-to-end integration testing.
"""

import os
import sys
import csv
import time
import json
import random
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Any, Optional
import numpy as np

from src.data_loader import DataLoader, Entity
from src.validation import ValidationSplitter
from src.normalization import TextNormalizer
from src.features import PairwiseFeatureExtractor, NormalizedEntity
from src.matching_model import EntityMatchingModel, RuleBasedBaseline
from src.blocking import CandidateGenerator, BlockingConfig


def load_raw_s1_entities(loader: DataLoader, target_s1_ids: Set[str]) -> Dict[str, Entity]:
    """Fast load raw S1 Entity objects for requested target S1 IDs."""
    s1_file = loader.get_file_path("train", "source1")
    s1_entities: Dict[str, Entity] = {}
    with open(s1_file, "r", encoding="utf-8", errors="replace") as f:
        f.readline()  # Skip header
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
    Ensures ALL required ground truth target IDs are included, plus deterministic
    non-ground-truth distractors up to at least min_target_size (or more if required_ids is larger).
    Does NOT modify any raw dataset files.
    """
    target_file = loader.get_file_path("train", source)
    target_size = max(min_target_size, len(required_ids))

    selected_required: List[Entity] = []
    found_required_ids: Set[str] = set()
    distractors: List[Entity] = []

    with open(target_file, "r", encoding="utf-8", errors="replace") as f:
        f.readline()  # Skip header
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

            # Stop early only if all required IDs have been found AND target size reached
            if len(found_required_ids) == len(required_ids) and (len(selected_required) + len(distractors)) >= target_size:
                break

    return selected_required + distractors


def generate_blocking_candidate_pairs(
    s1_entities: Dict[str, Entity],
    candidate_generator: CandidateGenerator
) -> List[Tuple[str, str]]:
    """
    Generate candidate pairs using Ashwika's CandidateGenerator.
    GROUND TRUTH IS NEVER CONSULTED DURING CANDIDATE GENERATION.
    Returns candidate pairs [(s1_id, candidate_id), ...].
    """
    pairs: List[Tuple[str, str]] = []
    for s1_id, s1_entity in s1_entities.items():
        s2_cands, s3_cands = candidate_generator.generate_candidates(s1_entity)
        for cand_id in s2_cands:
            pairs.append((s1_id, cand_id))
        for cand_id in s3_cands:
            pairs.append((s1_id, cand_id))
    return pairs


def evaluate_candidate_blocking_recall(
    val_s1_entities: Dict[str, Entity],
    candidate_pairs: List[Tuple[str, str]],
    val_gt: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Evaluate candidate recall and S1 complete-coverage metrics on validation set.
    """
    cand_dict: Dict[str, Set[str]] = defaultdict(set)
    for s1_id, cand_id in candidate_pairs:
        cand_dict[s1_id].add(cand_id)

    total_gt_s2 = 0
    retrieved_gt_s2 = 0
    total_gt_s3 = 0
    retrieved_gt_s3 = 0
    total_gt_all = 0
    retrieved_gt_all = 0

    s1_complete_count = 0
    total_s1 = len(val_s1_entities)

    for s1_id in val_s1_entities:
        gt_matches = val_gt.get(s1_id, [])
        gt_s2 = [m for m in gt_matches if m.startswith("S2-")]
        gt_s3 = [m for m in gt_matches if m.startswith("S3-")]

        total_gt_s2 += len(gt_s2)
        total_gt_s3 += len(gt_s3)
        total_gt_all += len(gt_matches)

        retrieved_s2 = sum(1 for m in gt_s2 if m in cand_dict[s1_id])
        retrieved_s3 = sum(1 for m in gt_s3 if m in cand_dict[s1_id])
        retrieved_all = sum(1 for m in gt_matches if m in cand_dict[s1_id])

        retrieved_gt_s2 += retrieved_s2
        retrieved_gt_s3 += retrieved_s3
        retrieved_gt_all += retrieved_all

        # S1 complete-candidate coverage: all GT matches present in candidate set
        if retrieved_all == len(gt_matches):
            s1_complete_count += 1

    rec_s2 = (retrieved_gt_s2 / total_gt_s2) if total_gt_s2 > 0 else 1.0
    rec_s3 = (retrieved_gt_s3 / total_gt_s3) if total_gt_s3 > 0 else 1.0
    rec_all = (retrieved_gt_all / total_gt_all) if total_gt_all > 0 else 1.0
    s1_cov_pct = (s1_complete_count / total_s1 * 100) if total_s1 > 0 else 100.0

    return {
        "total_val_s1_entities": total_s1,
        "total_gt_s2_links": total_gt_s2,
        "retrieved_gt_s2_links": retrieved_gt_s2,
        "link_level_s2_recall": float(np.round(rec_s2, 4)),
        "total_gt_s3_links": total_gt_s3,
        "retrieved_gt_s3_links": retrieved_gt_s3,
        "link_level_s3_recall": float(np.round(rec_s3, 4)),
        "total_gt_links": total_gt_all,
        "retrieved_gt_links": retrieved_gt_all,
        "combined_link_level_recall": float(np.round(rec_all, 4)),
        "s1_complete_coverage_count": s1_complete_count,
        "s1_complete_coverage_pct": float(np.round(s1_cov_pct, 2))
    }


def assign_ground_truth_labels(
    pairs: List[Tuple[str, str]],
    gt_mapping: Dict[str, List[str]]
) -> np.ndarray:
    """
    Assign binary ground truth labels to candidate pairs POST-BLOCKING.
    label = 1 if candidate_id is in gt_mapping[s1_id] else 0.
    """
    labels = [1 if cand_id in gt_mapping.get(s1_id, []) else 0 for s1_id, cand_id in pairs]
    return np.array(labels, dtype=np.int32)


def load_normalized_entity_store(
    loader: DataLoader,
    normalizer: TextNormalizer,
    target_s1_ids: Optional[Set[str]] = None,
    target_s2_ids: Optional[Set[str]] = None,
    target_s3_ids: Optional[Set[str]] = None,
    preloaded_entities: Optional[Dict[str, Entity]] = None
) -> Dict[str, NormalizedEntity]:
    """
    Load and pre-normalize entity records for requested target IDs across S1, S2, and S3.
    Uses preloaded_entities if provided, otherwise streams from raw TSVs.
    """
    entity_store: Dict[str, NormalizedEntity] = {}

    all_target_ids: Set[str] = set()
    if target_s1_ids:
        all_target_ids.update(target_s1_ids)
    if target_s2_ids:
        all_target_ids.update(target_s2_ids)
    if target_s3_ids:
        all_target_ids.update(target_s3_ids)

    # Use in-memory preloaded entities if available
    if preloaded_entities:
        for eid in all_target_ids:
            if eid in preloaded_entities:
                entity_store[eid] = NormalizedEntity.from_entity(preloaded_entities[eid], normalizer)

        remaining_s1 = (target_s1_ids or set()) - set(entity_store.keys())
        remaining_s2 = (target_s2_ids or set()) - set(entity_store.keys())
        remaining_s3 = (target_s3_ids or set()) - set(entity_store.keys())
        if not remaining_s1 and not remaining_s2 and not remaining_s3:
            return entity_store
    else:
        remaining_s1 = target_s1_ids
        remaining_s2 = target_s2_ids
        remaining_s3 = target_s3_ids

    # Stream Source 1 if needed
    if remaining_s1 is None or len(remaining_s1) > 0:
        s1_file = loader.get_file_path("train", "source1")
        with open(s1_file, "r", encoding="utf-8", errors="replace") as f:
            f.readline()
            for line in f:
                if not line:
                    continue
                tab_pos = line.find("\t")
                if tab_pos == -1:
                    continue
                eid = line[:tab_pos].strip()
                if remaining_s1 is None or eid in remaining_s1:
                    row = line.rstrip("\r\n").split("\t")
                    e = Entity.from_row(row)
                    entity_store[eid] = NormalizedEntity.from_entity(e, normalizer)

    # Stream Source 2 if needed
    if remaining_s2 is None or len(remaining_s2) > 0:
        s2_file = loader.get_file_path("train", "source2")
        with open(s2_file, "r", encoding="utf-8", errors="replace") as f:
            f.readline()
            for line in f:
                if not line:
                    continue
                tab_pos = line.find("\t")
                if tab_pos == -1:
                    continue
                eid = line[:tab_pos].strip()
                if remaining_s2 is None or eid in remaining_s2:
                    row = line.rstrip("\r\n").split("\t")
                    e = Entity.from_row(row)
                    entity_store[eid] = NormalizedEntity.from_entity(e, normalizer)

    # Stream Source 3 if needed
    if remaining_s3 is None or len(remaining_s3) > 0:
        s3_file = loader.get_file_path("train", "source3")
        with open(s3_file, "r", encoding="utf-8", errors="replace") as f:
            f.readline()
            for line in f:
                if not line:
                    continue
                tab_pos = line.find("\t")
                if tab_pos == -1:
                    continue
                eid = line[:tab_pos].strip()
                if remaining_s3 is None or eid in remaining_s3:
                    row = line.rstrip("\r\n").split("\t")
                    e = Entity.from_row(row)
                    entity_store[eid] = NormalizedEntity.from_entity(e, normalizer)

    return entity_store


def evaluate_s1_cardinality_breakdown(
    val_pairs: List[Tuple[str, str]],
    probs: np.ndarray,
    threshold: float,
    val_gt: Dict[str, List[str]],
    val_s1_ids: Set[str]
) -> Dict[str, Any]:
    """
    Evaluate validation performance partitioned by S1 entity ground-truth match cardinality:
    - zero-match: Ground truth matches == 0
    - single-match: Ground truth matches == 1
    - multi-match: Ground truth matches > 1
    """
    predicted_matches: Dict[str, List[str]] = {}
    for (s1_id, cand_id), p in zip(val_pairs, probs):
        if p >= threshold:
            if s1_id not in predicted_matches:
                predicted_matches[s1_id] = []
            predicted_matches[s1_id].append(cand_id)

    zero_match_s1 = {s for s in val_s1_ids if len(val_gt.get(s, [])) == 0}
    single_match_s1 = {s for s in val_s1_ids if len(val_gt.get(s, [])) == 1}
    multi_match_s1 = {s for s in val_s1_ids if len(val_gt.get(s, [])) > 1}

    def calc_group_exact(group_ids: Set[str]) -> Tuple[int, float]:
        if not group_ids:
            return 0, 0.0
        exact = sum(1 for s in group_ids if set(val_gt.get(s, [])) == set(predicted_matches.get(s, [])))
        pct = (exact / len(group_ids)) * 100
        return exact, pct

    z_exact, z_pct = calc_group_exact(zero_match_s1)
    s_exact, s_pct = calc_group_exact(single_match_s1)
    m_exact, m_pct = calc_group_exact(multi_match_s1)

    return {
        "zero-match": {
            "total_entities": len(zero_match_s1),
            "exact_matches": z_exact,
            "exact_match_rate_pct": float(np.round(z_pct, 2))
        },
        "single-match": {
            "total_entities": len(single_match_s1),
            "exact_matches": s_exact,
            "exact_match_rate_pct": float(np.round(s_pct, 2))
        },
        "multi-match": {
            "total_entities": len(multi_match_s1),
            "exact_matches": m_exact,
            "exact_match_rate_pct": float(np.round(m_pct, 2))
        }
    }


def categorize_error(
    e1: NormalizedEntity,
    e2: NormalizedEntity,
    is_fp: bool,
    prob: float,
    threshold: float
) -> str:
    """
    Categorize error sample deterministically based ONLY on dataset fields and features.
    """
    if not e1.norm_address or not e2.norm_address:
        return "missing_address"

    raw_diff = (e1.raw_name != e2.raw_name)
    norm_same = (e1.norm_name == e2.norm_name)

    if raw_diff and norm_same:
        return "transliteration"

    t1, t2 = set(e1.name_tokens), set(e2.name_tokens)
    if t1 and t2:
        if t1.issubset(t2) or t2.issubset(t1):
            return "legal_suffix"

        p1, p2 = (e1.name_tokens[0] if e1.name_tokens else ""), (e2.name_tokens[0] if e2.name_tokens else "")
        if p1 and p2 and (p1.startswith(p2) or p2.startswith(p1)) and p1 != p2:
            return "abbreviation"

    # Name similarity vs Address conflict
    n_ngrams1, n_ngrams2 = e1.name_char_3grams, e2.name_char_3grams
    n_inter = len(n_ngrams1 & n_ngrams2)
    n_union = len(n_ngrams1 | n_ngrams2)
    n_char_sim = (n_inter / n_union) if n_union > 0 else 0.0

    a_t1, a_t2 = set(e1.addr_tokens), set(e2.addr_tokens)
    a_inter = len(a_t1 & a_t2)
    a_union = len(a_t1 | a_t2)
    a_jaccard = (a_inter / a_union) if a_union > 0 else 0.0

    if n_char_sim >= 0.70 and a_jaccard < 0.20:
        return "address_conflict"

    if n_char_sim >= 0.75:
        return "spelling_typo"

    if len(e1.norm_name) < 12 or len(e2.norm_name) < 12:
        return "generic_name"

    return "uncategorized"


def run_error_analysis(
    val_pairs: List[Tuple[str, str]],
    val_y_true: np.ndarray,
    val_probs: np.ndarray,
    best_threshold: float,
    entity_store: Dict[str, NormalizedEntity]
) -> Dict[str, Any]:
    """
    Conduct detailed error analysis with deterministic error categorization.
    """
    y_pred = (val_probs >= best_threshold).astype(int)

    false_positives = []
    false_negatives = []

    category_counts: Counter = Counter()

    for idx, (pair, y_t, y_p, prob) in enumerate(zip(val_pairs, val_y_true, y_pred, val_probs)):
        s1_id, cand_id = pair
        e1 = entity_store.get(s1_id)
        e2 = entity_store.get(cand_id)

        if e1 and e2:
            is_fp = (y_t == 0 and y_p == 1)
            is_fn = (y_t == 1 and y_p == 0)

            if is_fp or is_fn:
                cat = categorize_error(e1, e2, is_fp, prob, best_threshold)
                category_counts[cat] += 1

                info = {
                    "s1_id": s1_id,
                    "cand_id": cand_id,
                    "s1_name": e1.raw_name,
                    "cand_name": e2.raw_name,
                    "s1_addr": e1.raw_address,
                    "cand_addr": e2.raw_address,
                    "s1_country": e1.country,
                    "cand_country": e2.country,
                    "predicted_prob": float(np.round(prob, 4)),
                    "error_category": cat,
                    "threshold": best_threshold
                }

                if is_fp:
                    false_positives.append(info)
                elif is_fn:
                    false_negatives.append(info)

    false_positives.sort(key=lambda x: x["predicted_prob"], reverse=True)
    false_negatives.sort(key=lambda x: x["predicted_prob"])

    return {
        "summary": {
            "total_false_positives": len(false_positives),
            "total_false_negatives": len(false_negatives),
            "error_category_counts": dict(category_counts)
        },
        "top_false_positives_sample": false_positives[:10],
        "top_false_negatives_sample": false_negatives[:10]
    }


def stream_train_pipeline(
    train_s1_ids: List[str],
    loader: DataLoader,
    cand_gen: CandidateGenerator,
    normalizer: TextNormalizer,
    extractor: PairwiseFeatureExtractor,
    gt_mapping: Dict[str, List[str]],
    preloaded_entities: Optional[Dict[str, Entity]] = None,
    chunk_s1_size: int = 10000,
    max_negative_train_pairs: int = 5000000,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Memory-safe chunked candidate generation and feature extraction for training.
    Retains all positive candidate pairs and samples negative candidate pairs up to max_negative_train_pairs.
    Does NOT materialize the complete candidate pair list in RAM.
    """
    total_train_s1 = len(train_s1_ids)
    num_chunks = (total_train_s1 + chunk_s1_size - 1) // chunk_s1_size

    train_X_chunks: List[np.ndarray] = []
    train_y_chunks: List[np.ndarray] = []

    total_candidates_generated = 0
    total_positives_found = 0
    total_negatives_found = 0
    total_negatives_sampled = 0

    t_start = time.perf_counter()

    for chunk_idx in range(num_chunks):
        t_chunk0 = time.perf_counter()
        start_i = chunk_idx * chunk_s1_size
        end_i = min(start_i + chunk_s1_size, total_train_s1)
        chunk_s1_ids = train_s1_ids[start_i:end_i]
        chunk_s1_set = set(chunk_s1_ids)

        # 1. Load S1 entities for current chunk
        if preloaded_entities:
            chunk_s1_entities = {eid: preloaded_entities[eid] for eid in chunk_s1_ids if eid in preloaded_entities}
        else:
            chunk_s1_entities = load_raw_s1_entities(loader, chunk_s1_set)

        # 2. Generate Candidate Pairs for chunk
        chunk_pairs: List[Tuple[str, str]] = []
        for s1_id, s1_entity in chunk_s1_entities.items():
            s2_cands, s3_cands = cand_gen.generate_candidates(s1_entity)
            for cand_id in s2_cands:
                chunk_pairs.append((s1_id, cand_id))
            for cand_id in s3_cands:
                chunk_pairs.append((s1_id, cand_id))

        if not chunk_pairs:
            continue

        total_candidates_generated += len(chunk_pairs)

        # 3. Assign chunk ground truth labels
        chunk_y = np.array([1 if cand_id in gt_mapping.get(s1_id, []) else 0 for s1_id, cand_id in chunk_pairs], dtype=np.int32)
        pos_mask = (chunk_y == 1)
        neg_mask = (chunk_y == 0)

        chunk_pos_count = int(np.sum(pos_mask))
        chunk_neg_count = int(np.sum(neg_mask))
        total_positives_found += chunk_pos_count
        total_negatives_found += chunk_neg_count

        # 4. Load normalized entities for current chunk only
        needed_s2 = {pair[1] for pair in chunk_pairs if pair[1].startswith("S2-")}
        needed_s3 = {pair[1] for pair in chunk_pairs if pair[1].startswith("S3-")}

        chunk_entity_store: Dict[str, NormalizedEntity] = {}
        for s1_id, s1_ent in chunk_s1_entities.items():
            chunk_entity_store[s1_id] = NormalizedEntity.from_entity(s1_ent, normalizer)

        for target_id in needed_s2.union(needed_s3):
            if preloaded_entities and target_id in preloaded_entities:
                chunk_entity_store[target_id] = NormalizedEntity.from_entity(preloaded_entities[target_id], normalizer)
            else:
                raw_target = cand_gen.get_raw_entity(target_id)
                if raw_target:
                    chunk_entity_store[target_id] = NormalizedEntity.from_entity(raw_target, normalizer)

        # 5. Extract features for chunk pairs
        X_chunk, valid_pairs = extractor.extract_batch_features(chunk_pairs, chunk_entity_store)

        # 6. Sample negatives deterministically for current chunk
        pos_indices = np.where(pos_mask)[0]
        neg_indices = np.where(neg_mask)[0]

        chunk_frac = len(chunk_s1_ids) / total_train_s1
        chunk_target_neg = int(max_negative_train_pairs * chunk_frac)

        if len(neg_indices) > chunk_target_neg and max_negative_train_pairs < 500000000:
            chunk_rng = np.random.RandomState(random_state + chunk_idx)
            sampled_neg_indices = chunk_rng.choice(neg_indices, size=chunk_target_neg, replace=False)
        else:
            sampled_neg_indices = neg_indices

        total_negatives_sampled += len(sampled_neg_indices)
        selected_indices = np.concatenate([pos_indices, sampled_neg_indices])
        selected_indices.sort()

        train_X_chunks.append(X_chunk[selected_indices])
        train_y_chunks.append(chunk_y[selected_indices])

        chunk_elapsed = time.perf_counter() - t_chunk0
        total_elapsed = time.perf_counter() - t_start
        print(f"   [TRAIN STREAMING] Chunk {chunk_idx + 1:3d}/{num_chunks:3d} ({len(chunk_s1_ids):,} S1) -> {len(chunk_pairs):,} pairs (Pos: {chunk_pos_count:,}, Neg: {chunk_neg_count:,}, Sampled Neg: {len(sampled_neg_indices):,}) in {chunk_elapsed:.2f}s | Elapsed: {total_elapsed:.1f}s")

    X_train = np.vstack(train_X_chunks) if train_X_chunks else np.empty((0, 25), dtype=np.float32)
    y_train = np.concatenate(train_y_chunks) if train_y_chunks else np.empty((0,), dtype=np.int32)

    stats = {
        "total_train_s1": total_train_s1,
        "total_candidates_generated": total_candidates_generated,
        "total_positives": total_positives_found,
        "total_negatives_observed": total_negatives_found,
        "total_negatives_sampled": total_negatives_sampled,
        "total_training_pairs": len(y_train),
        "positive_negative_ratio": float(np.round(total_positives_found / total_negatives_sampled, 6)) if total_negatives_sampled > 0 else 0.0,
        "random_state": random_state,
        "total_elapsed_seconds": time.perf_counter() - t_start
    }

    return X_train, y_train, stats


def stream_validation_pipeline(
    val_s1_ids: List[str],
    loader: DataLoader,
    cand_gen: CandidateGenerator,
    normalizer: TextNormalizer,
    extractor: PairwiseFeatureExtractor,
    gt_mapping: Dict[str, List[str]],
    model: EntityMatchingModel,
    rule_baseline: RuleBasedBaseline,
    preloaded_entities: Optional[Dict[str, Entity]] = None,
    chunk_s1_size: int = 10000,
    thresholds: Optional[List[float]] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Memory-safe chunked candidate generation, feature extraction, and evaluation for validation.
    Maintains running S1-level metrics across all decision thresholds.
    Does NOT materialize the complete candidate pair list or predictions in RAM.
    """
    if thresholds is None:
        thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]

    total_val_s1 = len(val_s1_ids)
    num_chunks = (total_val_s1 + chunk_s1_size - 1) // chunk_s1_size

    # Incremental candidate blocking recall stats
    total_gt_s2 = 0
    retrieved_gt_s2 = 0
    total_gt_s3 = 0
    retrieved_gt_s3 = 0
    total_gt_all = 0
    retrieved_gt_all = 0
    s1_complete_count = 0
    total_val_pairs_generated = 0

    # RuleBasedBaseline running metrics
    rule_tp = 0
    rule_fp = 0
    rule_fn = 0
    rule_tn = 0

    # Threshold evaluation accumulators
    threshold_stats = {
        t: {
            "s1_tp": 0, "s1_fp": 0, "s1_fn": 0, "exact_matches": 0,
            "zero_exact": 0, "single_exact": 0, "multi_exact": 0,
            "pairwise_tp": 0, "pairwise_fp": 0, "pairwise_fn": 0, "pairwise_tn": 0
        } for t in thresholds
    }

    # Cardinality group entity counts
    zero_match_count = 0
    single_match_count = 0
    multi_match_count = 0

    # Bounded error analysis samples
    false_positives_sample: List[Dict[str, Any]] = []
    false_negatives_sample: List[Dict[str, Any]] = []
    error_category_counts: Counter = Counter()

    t_start = time.perf_counter()

    for chunk_idx in range(num_chunks):
        t_chunk0 = time.perf_counter()
        start_i = chunk_idx * chunk_s1_size
        end_i = min(start_i + chunk_s1_size, total_val_s1)
        chunk_s1_ids = val_s1_ids[start_i:end_i]
        chunk_s1_set = set(chunk_s1_ids)

        # 1. Load S1 entities for current chunk
        if preloaded_entities:
            chunk_s1_entities = {eid: preloaded_entities[eid] for eid in chunk_s1_ids if eid in preloaded_entities}
        else:
            chunk_s1_entities = load_raw_s1_entities(loader, chunk_s1_set)

        # 2. Generate Candidate Pairs for chunk
        chunk_pairs: List[Tuple[str, str]] = []
        chunk_cand_map: Dict[str, Set[str]] = defaultdict(set)
        for s1_id, s1_entity in chunk_s1_entities.items():
            s2_cands, s3_cands = cand_gen.generate_candidates(s1_entity)
            for cand_id in s2_cands:
                chunk_pairs.append((s1_id, cand_id))
                chunk_cand_map[s1_id].add(cand_id)
            for cand_id in s3_cands:
                chunk_pairs.append((s1_id, cand_id))
                chunk_cand_map[s1_id].add(cand_id)

        total_val_pairs_generated += len(chunk_pairs)

        # 3. Update candidate blocking recall incrementally for this chunk
        for s1_id in chunk_s1_ids:
            gt_matches = gt_mapping.get(s1_id, [])
            gt_s2 = [m for m in gt_matches if m.startswith("S2-")]
            gt_s3 = [m for m in gt_matches if m.startswith("S3-")]

            total_gt_s2 += len(gt_s2)
            total_gt_s3 += len(gt_s3)
            total_gt_all += len(gt_matches)

            cands_for_s1 = chunk_cand_map.get(s1_id, set())
            ret_s2 = sum(1 for m in gt_s2 if m in cands_for_s1)
            ret_s3 = sum(1 for m in gt_s3 if m in cands_for_s1)
            ret_all = sum(1 for m in gt_matches if m in cands_for_s1)

            retrieved_gt_s2 += ret_s2
            retrieved_gt_s3 += ret_s3
            retrieved_gt_all += ret_all

            if ret_all == len(gt_matches):
                s1_complete_count += 1

            if len(gt_matches) == 0:
                zero_match_count += 1
            elif len(gt_matches) == 1:
                single_match_count += 1
            else:
                multi_match_count += 1

        if not chunk_pairs:
            continue

        # 4. Assign chunk labels
        chunk_y = np.array([1 if cand_id in gt_mapping.get(s1_id, []) else 0 for s1_id, cand_id in chunk_pairs], dtype=np.int32)

        # 5. Build chunk-scoped normalized entity store
        needed_s2 = {pair[1] for pair in chunk_pairs if pair[1].startswith("S2-")}
        needed_s3 = {pair[1] for pair in chunk_pairs if pair[1].startswith("S3-")}

        chunk_entity_store: Dict[str, NormalizedEntity] = {}
        for s1_id, s1_ent in chunk_s1_entities.items():
            chunk_entity_store[s1_id] = NormalizedEntity.from_entity(s1_ent, normalizer)

        for target_id in needed_s2.union(needed_s3):
            if preloaded_entities and target_id in preloaded_entities:
                chunk_entity_store[target_id] = NormalizedEntity.from_entity(preloaded_entities[target_id], normalizer)
            else:
                raw_target = cand_gen.get_raw_entity(target_id)
                if raw_target:
                    chunk_entity_store[target_id] = NormalizedEntity.from_entity(raw_target, normalizer)

        # 6. Extract features and predict probabilities
        X_chunk, valid_pairs = extractor.extract_batch_features(chunk_pairs, chunk_entity_store)
        val_probs = model.predict_proba(X_chunk)
        rule_val_probs = rule_baseline.predict_proba(X_chunk)

        # 7. Update RuleBasedBaseline benchmark metrics
        rule_pred = (rule_val_probs >= 0.50).astype(int)
        rule_tp += int(np.sum((chunk_y == 1) & (rule_pred == 1)))
        rule_fp += int(np.sum((chunk_y == 0) & (rule_pred == 1)))
        rule_fn += int(np.sum((chunk_y == 1) & (rule_pred == 0)))
        rule_tn += int(np.sum((chunk_y == 0) & (rule_pred == 0)))

        # 8. Group predictions per S1 entity
        s1_preds_map: Dict[str, List[Tuple[str, int, float]]] = defaultdict(list)
        for (s1_id, cand_id), y_t, prob in zip(valid_pairs, chunk_y, val_probs):
            s1_preds_map[s1_id].append((cand_id, int(y_t), float(prob)))

        # 9. Evaluate all thresholds for this chunk's S1 entities
        for s1_id in chunk_s1_ids:
            gt_set = set(gt_mapping.get(s1_id, []))
            preds_list = s1_preds_map.get(s1_id, [])

            for t in thresholds:
                pred_set = {cand_id for cand_id, _, p in preds_list if p >= t}
                tp_s1 = len(pred_set & gt_set)
                fp_s1 = len(pred_set - gt_set)
                fn_s1 = len(gt_set - pred_set)

                threshold_stats[t]["s1_tp"] += tp_s1
                threshold_stats[t]["s1_fp"] += fp_s1
                threshold_stats[t]["s1_fn"] += fn_s1

                is_exact = (pred_set == gt_set)
                if is_exact:
                    threshold_stats[t]["exact_matches"] += 1
                    if len(gt_set) == 0:
                        threshold_stats[t]["zero_exact"] += 1
                    elif len(gt_set) == 1:
                        threshold_stats[t]["single_exact"] += 1
                    else:
                        threshold_stats[t]["multi_exact"] += 1

                pw_tp = sum(1 for _, y_t, p in preds_list if y_t == 1 and p >= t)
                pw_fp = sum(1 for _, y_t, p in preds_list if y_t == 0 and p >= t)
                pw_fn = sum(1 for _, y_t, p in preds_list if y_t == 1 and p < t)
                pw_tn = sum(1 for _, y_t, p in preds_list if y_t == 0 and p < t)

                threshold_stats[t]["pairwise_tp"] += pw_tp
                threshold_stats[t]["pairwise_fp"] += pw_fp
                threshold_stats[t]["pairwise_fn"] += pw_fn
                threshold_stats[t]["pairwise_tn"] += pw_tn

        # 10. Sample bounded error analysis instances
        for (s1_id, cand_id), y_t, prob in zip(valid_pairs, chunk_y, val_probs):
            e1 = chunk_entity_store.get(s1_id)
            e2 = chunk_entity_store.get(cand_id)
            if not e1 or not e2:
                continue

            # Candidate errors at baseline threshold 0.90
            is_fp = (y_t == 0 and prob >= 0.90)
            is_fn = (y_t == 1 and prob < 0.90)

            if is_fp or is_fn:
                cat = categorize_error(e1, e2, is_fp, float(prob), 0.90)
                error_category_counts[cat] += 1

                err_info = {
                    "s1_id": s1_id,
                    "cand_id": cand_id,
                    "s1_name": e1.raw_name,
                    "cand_name": e2.raw_name,
                    "s1_addr": e1.raw_address,
                    "cand_addr": e2.raw_address,
                    "s1_country": e1.country,
                    "cand_country": e2.country,
                    "predicted_prob": float(np.round(prob, 4)),
                    "error_category": cat,
                    "threshold": 0.90
                }

                if is_fp and len(false_positives_sample) < 20:
                    false_positives_sample.append(err_info)
                elif is_fn and len(false_negatives_sample) < 20:
                    false_negatives_sample.append(err_info)

        chunk_elapsed = time.perf_counter() - t_chunk0
        total_elapsed = time.perf_counter() - t_start
        print(f"   [VAL STREAMING]   Chunk {chunk_idx + 1:3d}/{num_chunks:3d} ({len(chunk_s1_ids):,} S1) -> {len(chunk_pairs):,} pairs in {chunk_elapsed:.2f}s | Elapsed: {total_elapsed:.1f}s")

    # Finalize Threshold Experiment Results
    threshold_results = []
    beta_sq = 0.5 ** 2

    for t in thresholds:
        st = threshold_stats[t]
        s1_tp = st["s1_tp"]
        s1_fp = st["s1_fp"]
        s1_fn = st["s1_fn"]

        prec = (s1_tp / (s1_tp + s1_fp)) if (s1_tp + s1_fp) > 0 else 0.0
        rec = (s1_tp / (s1_tp + s1_fn)) if (s1_tp + s1_fn) > 0 else 0.0
        f05 = ((1 + beta_sq) * prec * rec) / ((beta_sq * prec) + rec) if ((beta_sq * prec) + rec) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        exact_pct = (st["exact_matches"] / total_val_s1 * 100) if total_val_s1 > 0 else 0.0

        threshold_results.append({
            "threshold": t,
            "s1_precision": float(np.round(prec, 4)),
            "s1_recall": float(np.round(rec, 4)),
            "s1_f05": float(np.round(f05, 4)),
            "s1_f1": float(np.round(f1, 4)),
            "exact_matched_s1_entities": st["exact_matches"],
            "s1_exact_match_rate_pct": float(np.round(exact_pct, 2))
        })

    best_res = max(threshold_results, key=lambda item: item["s1_f05"])
    best_t = best_res["threshold"]
    best_st = threshold_stats[best_t]

    # RuleBasedBaseline metrics
    rule_prec = (rule_tp / (rule_tp + rule_fp)) if (rule_tp + rule_fp) > 0 else 0.0
    rule_rec = (rule_tp / (rule_tp + rule_fn)) if (rule_tp + rule_fn) > 0 else 0.0
    rule_f05 = ((1 + beta_sq) * rule_prec * rule_rec) / ((beta_sq * rule_prec) + rule_rec) if ((beta_sq * rule_prec) + rule_rec) > 0 else 0.0
    rule_f1 = (2 * rule_prec * rule_rec) / (rule_prec + rule_rec) if (rule_prec + rule_rec) > 0 else 0.0

    # Blocking recall metrics
    cand_recall_metrics = {
        "total_val_s1_entities": total_val_s1,
        "total_gt_s2_links": total_gt_s2,
        "retrieved_gt_s2_links": retrieved_gt_s2,
        "link_level_s2_recall": float(np.round(retrieved_gt_s2 / total_gt_s2, 4)) if total_gt_s2 > 0 else 1.0,
        "total_gt_s3_links": total_gt_s3,
        "retrieved_gt_s3_links": retrieved_gt_s3,
        "link_level_s3_recall": float(np.round(retrieved_gt_s3 / total_gt_s3, 4)) if total_gt_s3 > 0 else 1.0,
        "total_gt_links": total_gt_all,
        "retrieved_gt_links": retrieved_gt_all,
        "combined_link_level_recall": float(np.round(retrieved_gt_all / total_gt_all, 4)) if total_gt_all > 0 else 1.0,
        "s1_complete_coverage_count": s1_complete_count,
        "s1_complete_coverage_pct": float(np.round(s1_complete_count / total_val_s1 * 100, 2)) if total_val_s1 > 0 else 100.0
    }

    # Pairwise metrics at best threshold
    pw_tp = best_st["pairwise_tp"]
    pw_fp = best_st["pairwise_fp"]
    pw_fn = best_st["pairwise_fn"]
    pw_tn = best_st["pairwise_tn"]
    pw_prec = (pw_tp / (pw_tp + pw_fp)) if (pw_tp + pw_fp) > 0 else 0.0
    pw_rec = (pw_tp / (pw_tp + pw_fn)) if (pw_tp + pw_fn) > 0 else 0.0
    pw_f05 = ((1 + beta_sq) * pw_prec * pw_rec) / ((beta_sq * pw_prec) + pw_rec) if ((beta_sq * pw_prec) + pw_rec) > 0 else 0.0
    pw_f1 = (2 * pw_prec * pw_rec) / (pw_prec + pw_rec) if (pw_prec + pw_rec) > 0 else 0.0

    cardinality_breakdown = {
        "zero-match": {
            "total_entities": zero_match_count,
            "exact_matches": best_st["zero_exact"],
            "exact_match_rate_pct": float(np.round(best_st["zero_exact"] / zero_match_count * 100, 2)) if zero_match_count > 0 else 0.0
        },
        "single-match": {
            "total_entities": single_match_count,
            "exact_matches": best_st["single_exact"],
            "exact_match_rate_pct": float(np.round(best_st["single_exact"] / single_match_count * 100, 2)) if single_match_count > 0 else 0.0
        },
        "multi-match": {
            "total_entities": multi_match_count,
            "exact_matches": best_st["multi_exact"],
            "exact_match_rate_pct": float(np.round(best_st["multi_exact"] / multi_match_count * 100, 2)) if multi_match_count > 0 else 0.0
        }
    }

    error_analysis = {
        "summary": {
            "total_false_positives": best_st["s1_fp"],
            "total_false_negatives": best_st["s1_fn"],
            "error_category_counts": dict(error_category_counts)
        },
        "top_false_positives_sample": false_positives_sample[:10],
        "top_false_negatives_sample": false_negatives_sample[:10]
    }

    return {
        "candidate_blocking_metrics": cand_recall_metrics,
        "total_val_pairs": total_val_pairs_generated,
        "rule_based_baseline_metrics": {
            "precision": float(np.round(rule_prec, 4)),
            "recall": float(np.round(rule_rec, 4)),
            "f05": float(np.round(rule_f05, 4)),
            "f1": float(np.round(rule_f1, 4))
        },
        "threshold_experiment": {
            "results": threshold_results,
            "best_threshold": best_t,
            "best_validation_s1_f05": best_res["s1_f05"],
            "threshold_selection_criterion": "highest validation S1-level F0.5 score"
        },
        "best_threshold": best_t,
        "s1_level_metrics": {
            "s1_precision": best_res["s1_precision"],
            "s1_recall": best_res["s1_recall"],
            "s1_f05": best_res["s1_f05"],
            "s1_f1": best_res["s1_f1"],
            "s1_tp": best_st["s1_tp"],
            "s1_fp": best_st["s1_fp"],
            "s1_fn": best_st["s1_fn"],
            "s1_exact_match_rate_pct": best_res["s1_exact_match_rate_pct"],
            "exact_matched_s1_entities": best_st["exact_matches"],
            "total_eval_s1_entities": total_val_s1
        },
        "pairwise_diagnostic_metrics": {
            "pairwise_precision": float(np.round(pw_prec, 4)),
            "pairwise_recall": float(np.round(pw_rec, 4)),
            "pairwise_f05": float(np.round(pw_f05, 4)),
            "pairwise_f1": float(np.round(pw_f1, 4)),
            "confusion_matrix": [[pw_tn, pw_fp], [pw_fn, pw_tp]]
        },
        "s1_cardinality_breakdown": cardinality_breakdown,
        "error_analysis": error_analysis
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Pairwise Matching Model & Validation Runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run small smoke test on subset of data (2,000 train / 500 val S1 entities)")
    parser.add_argument("--full", action="store_true", help="Run full validation on complete training datasets (default behavior)")
    parser.add_argument("--streaming", action="store_true", help="Explicitly enable chunked streaming processing (default: True for full validation)")
    parser.add_argument("--chunk-s1-size", "--chunk-size", dest="chunk_s1_size", type=int, default=10000, help="Number of S1 entities per chunk (default: 10,000)")
    parser.add_argument("--max-negative-train-pairs", type=int, default=5000000, help="Max negative candidate pairs sampled for training (default: 5,000,000)")
    parser.add_argument("--smoke-s1-train-size", "--sample-s1-train", dest="smoke_s1_train_size", type=int, default=2000, help="Number of S1 train entities for smoke test")
    parser.add_argument("--smoke-s1-val-size", "--sample-s1-val", dest="smoke_s1_val_size", type=int, default=500, help="Number of S1 val entities for smoke test")
    parser.add_argument("--smoke-target-size", type=int, default=20000, help="Target subset size for S2 and S3 in smoke test")
    args = parser.parse_args()

    # Mode determination
    is_smoke_test = args.smoke_test
    # Streaming is used for full validation by default, or if explicitly requested with --streaming
    use_streaming = args.streaming if args.streaming else (not is_smoke_test)

    print("=========================================================")
    print(f" Amazon ML Challenge 2026: Pairwise Matching Validation ")
    print(f" Mode: {'SMOKE TEST' if is_smoke_test else 'FULL VALIDATION'} | Pipeline: {'STREAMING (CHUNKED)' if use_streaming else 'MONOLITHIC IN-MEMORY'} ")
    print("=========================================================\n")

    loader = DataLoader(data_dir="data")
    splitter = ValidationSplitter(data_loader=loader, seed=42)
    normalizer = TextNormalizer()
    extractor = PairwiseFeatureExtractor(normalizer=normalizer)

    splits_dir = "experiments/splits"
    train_split_path = os.path.join(splits_dir, "train_s1_ids.txt")
    val_split_path = os.path.join(splits_dir, "val_s1_ids.txt")

    if not os.path.exists(train_split_path) or not os.path.exists(val_split_path):
        print("Split files not found. Generating deterministic splits...")
        splitter.generate_and_save_splits(output_dir=splits_dir)

    print("1. Loading split entity IDs...")
    all_train_s1 = splitter.load_split_ids(train_split_path)
    all_val_s1 = splitter.load_split_ids(val_split_path)
    print(f"   Train S1 IDs: {len(all_train_s1):,} | Val S1 IDs: {len(all_val_s1):,}")

    gt_file = loader.get_file_path("train", "ground_truth")
    preloaded_entities: Dict[str, Entity] = {}

    if is_smoke_test:
        train_s1_subset = list(sorted(list(all_train_s1))[:args.smoke_s1_train_size])
        val_s1_subset = list(sorted(list(all_val_s1))[:args.smoke_s1_val_size])
        all_selected_s1 = set(train_s1_subset).union(set(val_s1_subset))

        # Load ground truth for selected S1 entities ONLY
        t_gt0 = time.perf_counter()
        gt_mapping = load_smoke_ground_truth(gt_file, all_selected_s1)
        t_gt_elapsed = time.perf_counter() - t_gt0

        # Identify required GT target IDs for selected S1 entities ONLY for subset construction
        required_s2_ids: Set[str] = set()
        required_s3_ids: Set[str] = set()
        for s1_id in all_selected_s1:
            for m_id in gt_mapping.get(s1_id, []):
                if m_id.startswith("S2-"):
                    required_s2_ids.add(m_id)
                elif m_id.startswith("S3-"):
                    required_s3_ids.add(m_id)

        print("\n[SMOKE TEST CONFIGURATION]")
        print(f"   [SMOKE TEST] Train S1: {len(train_s1_subset):,}")
        print(f"   [SMOKE TEST] Val S1: {len(val_s1_subset):,}")
        print(f"   [SMOKE TEST] Required GT S2 IDs: {len(required_s2_ids):,}")
        print(f"   [SMOKE TEST] Required GT S3 IDs: {len(required_s3_ids):,}")

        # Build small deterministic S2 and S3 target subsets
        t_sub0 = time.perf_counter()
        s2_smoke_entities = build_smoke_target_subset(
            loader, "source2", required_s2_ids, min_target_size=args.smoke_target_size
        )
        s3_smoke_entities = build_smoke_target_subset(
            loader, "source3", required_s3_ids, min_target_size=args.smoke_target_size
        )
        t_sub_elapsed = time.perf_counter() - t_sub0

        s2_subset_id_set = {e.entity_id for e in s2_smoke_entities}
        s3_subset_id_set = {e.entity_id for e in s3_smoke_entities}

        print(f"   [SMOKE TEST] S2 target subset: {len(s2_smoke_entities):,}")
        print(f"   [SMOKE TEST] S3 target subset: {len(s3_smoke_entities):,}")
        print(f"   [SMOKE TEST] Required GT S2 IDs included: {len(required_s2_ids.intersection(s2_subset_id_set)):,} / {len(required_s2_ids):,}")
        print(f"   [SMOKE TEST] Required GT S3 IDs included: {len(required_s3_ids.intersection(s3_subset_id_set)):,} / {len(required_s3_ids):,}")
        print(f"   [SMOKE TEST] Target subsets prepared in {t_sub_elapsed:.2f}s.")

        # INTEGRITY CHECKS BEFORE RUNNING THE MODEL
        print("\n[RUNNING INTEGRITY CHECKS]")
        val_gt_s2 = {m for s in val_s1_subset for m in gt_mapping.get(s, []) if m.startswith("S2-")}
        val_gt_s3 = {m for s in val_s1_subset for m in gt_mapping.get(s, []) if m.startswith("S3-")}
        assert val_gt_s2.issubset(s2_subset_id_set), "Integrity Error: Missing GT S2 IDs in validation target subset!"
        assert val_gt_s3.issubset(s3_subset_id_set), "Integrity Error: Missing GT S3 IDs in validation target subset!"
        assert len(s2_smoke_entities) == len(s2_subset_id_set), "Integrity Error: Duplicate S2 entity IDs!"
        assert len(s3_smoke_entities) == len(s3_subset_id_set), "Integrity Error: Duplicate S3 entity IDs!"
        print("   [*] All integrity checks passed successfully!")

        # Preload into in-memory dictionary for ultra-fast normalization store creation
        for e in s2_smoke_entities:
            preloaded_entities[e.entity_id] = e
        for e in s3_smoke_entities:
            preloaded_entities[e.entity_id] = e

        train_s1_entities = load_raw_s1_entities(loader, set(train_s1_subset))
        val_s1_entities = load_raw_s1_entities(loader, set(val_s1_subset))
        for e in train_s1_entities.values():
            preloaded_entities[e.entity_id] = e
        for e in val_s1_entities.values():
            preloaded_entities[e.entity_id] = e

    else:
        train_s1_subset = list(all_train_s1)
        val_s1_subset = list(all_val_s1)
        gt_mapping = loader.load_ground_truth()
        s2_smoke_entities = None
        s3_smoke_entities = None

    print("\n2. Building Inverted Index Candidate Generator...")
    cand_gen = CandidateGenerator(data_loader=loader, config=BlockingConfig())

    t_build0 = time.perf_counter()
    if is_smoke_test and s2_smoke_entities is not None and s3_smoke_entities is not None:
        build_times = cand_gen.build_indexes_from_entities(s2_smoke_entities, s3_smoke_entities)
    else:
        build_times = cand_gen.build_indexes(split="train")
    total_build_time = time.perf_counter() - t_build0

    print(f"   Inverted Index Built: Source 2 in {build_times['source2']:.2f}s | Source 3 in {build_times['source3']:.2f}s (Total: {total_build_time:.2f}s)")

    if use_streaming:
        # =========================================================================
        # CHUNKED STREAMING PIPELINE (MEMORY-SAFE)
        # =========================================================================
        chunk_s1_size = args.chunk_s1_size if not is_smoke_test else min(args.chunk_s1_size, 500)
        max_negatives = args.max_negative_train_pairs

        print(f"\n3. Running Memory-Safe Streaming Training (Chunk Size: {chunk_s1_size:,} S1, Max Negatives: {max_negatives:,})...")
        t_train_stream0 = time.perf_counter()
        X_train, train_y, train_stats = stream_train_pipeline(
            train_s1_ids=train_s1_subset,
            loader=loader,
            cand_gen=cand_gen,
            normalizer=normalizer,
            extractor=extractor,
            gt_mapping=gt_mapping,
            preloaded_entities=preloaded_entities if is_smoke_test else None,
            chunk_s1_size=chunk_s1_size,
            max_negative_train_pairs=max_negatives,
            random_state=42
        )
        cand_gen_time = train_stats["total_elapsed_seconds"]

        print(f"\n4. Training EntityMatchingModel (LogisticRegression + StandardScaler on {len(train_y):,} pairs)...")
        t_model0 = time.perf_counter()
        model = EntityMatchingModel(C=1.0, class_weight="balanced", random_state=42)
        model.fit(X_train, train_y)
        model_train_time = time.perf_counter() - t_model0
        print(f"   Model training complete in {model_train_time:.2f}s.")

        print(f"\n5. Running Memory-Safe Streaming Validation (Chunk Size: {chunk_s1_size:,} S1)...")
        rule_baseline = RuleBasedBaseline()
        val_stream_results = stream_validation_pipeline(
            val_s1_ids=val_s1_subset,
            loader=loader,
            cand_gen=cand_gen,
            normalizer=normalizer,
            extractor=extractor,
            gt_mapping=gt_mapping,
            model=model,
            rule_baseline=rule_baseline,
            preloaded_entities=preloaded_entities if is_smoke_test else None,
            chunk_s1_size=chunk_s1_size,
            thresholds=[0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98],
            random_state=42
        )

        cand_recall_metrics = val_stream_results["candidate_blocking_metrics"]
        rule_metrics = val_stream_results["rule_based_baseline_metrics"]
        exp_results = val_stream_results["threshold_experiment"]
        best_t = val_stream_results["best_threshold"]
        val_s1_metrics = val_stream_results["s1_level_metrics"]
        val_pairwise = val_stream_results["pairwise_diagnostic_metrics"]
        cardinality_breakdown = val_stream_results["s1_cardinality_breakdown"]
        error_analysis = val_stream_results["error_analysis"]
        valid_train_pairs_count = len(train_y)
        valid_val_pairs_count = val_stream_results["total_val_pairs"]

    else:
        # =========================================================================
        # IN-MEMORY MONOLITHIC PIPELINE (FOR SMOKE-TEST BACKWARD COMPATIBILITY)
        # =========================================================================
        print("\n3. Generating Candidate Pairs via Blocking (NO Ground Truth Access)...")
        train_s1_entities = load_raw_s1_entities(loader, set(train_s1_subset))
        val_s1_entities = load_raw_s1_entities(loader, set(val_s1_subset))

        for e in train_s1_entities.values():
            preloaded_entities[e.entity_id] = e
        for e in val_s1_entities.values():
            preloaded_entities[e.entity_id] = e

        t_cand0 = time.perf_counter()
        train_pairs = generate_blocking_candidate_pairs(train_s1_entities, cand_gen)
        val_pairs = generate_blocking_candidate_pairs(val_s1_entities, cand_gen)
        cand_gen_time = time.perf_counter() - t_cand0
        print(f"   Candidate Generation Complete in {cand_gen_time:.2f}s.")

        print("\n4. Assigning Ground Truth Labels (POST-BLOCKING ONLY)...")
        train_gt = splitter.filter_ground_truth_for_split(gt_mapping, set(train_s1_subset))
        val_gt = splitter.filter_ground_truth_for_split(gt_mapping, set(val_s1_subset))

        train_y = assign_ground_truth_labels(train_pairs, gt_mapping)
        val_y = assign_ground_truth_labels(val_pairs, gt_mapping)

        print(f"   Generated Train pairs: {len(train_pairs):,} (Positives: {int(np.sum(train_y)):,}, Negatives: {len(train_pairs) - int(np.sum(train_y)):,})")
        print(f"   Generated Val pairs:   {len(val_pairs):,} (Positives: {int(np.sum(val_y)):,}, Negatives: {len(val_pairs) - int(np.sum(val_y)):,})")

        # Evaluate Candidate Blocking Recall
        cand_recall_metrics = evaluate_candidate_blocking_recall(val_s1_entities, val_pairs, val_gt)
        print("\n[CANDIDATE BLOCKING RECALL & COVERAGE]")
        print(f"   - Link-level S2 Recall:           {cand_recall_metrics['link_level_s2_recall'] * 100:.2f}% ({cand_recall_metrics['retrieved_gt_s2_links']:,} / {cand_recall_metrics['total_gt_s2_links']:,})")
        print(f"   - Link-level S3 Recall:           {cand_recall_metrics['link_level_s3_recall'] * 100:.2f}% ({cand_recall_metrics['retrieved_gt_s3_links']:,} / {cand_recall_metrics['total_gt_s3_links']:,})")
        print(f"   - Combined Link-level Recall:     {cand_recall_metrics['combined_link_level_recall'] * 100:.2f}% ({cand_recall_metrics['retrieved_gt_links']:,} / {cand_recall_metrics['total_gt_links']:,})")
        print(f"   - S1 Complete-Candidate Coverage: {cand_recall_metrics['s1_complete_coverage_pct']:.2f}% ({cand_recall_metrics['s1_complete_coverage_count']:,} / {cand_recall_metrics['total_val_s1_entities']:,})")

        # Load normalized entities for extracted pairs
        needed_s1 = set(train_s1_subset).union(set(val_s1_subset))
        needed_s2 = {pair[1] for pair in train_pairs + val_pairs if pair[1].startswith("S2-")}
        needed_s3 = {pair[1] for pair in train_pairs + val_pairs if pair[1].startswith("S3-")}

        print("\n5. Loading & Pre-normalizing Entities into Memory Store...")
        entity_store = load_normalized_entity_store(
            loader,
            normalizer,
            target_s1_ids=needed_s1,
            target_s2_ids=needed_s2,
            target_s3_ids=needed_s3,
            preloaded_entities=preloaded_entities if is_smoke_test else None
        )
        print(f"   Normalized entity store built: {len(entity_store):,} entities.")

        # Extract Features
        print("\n6. Extracting Pairwise Features (25 dimensions)...")
        X_train, valid_train_pairs = extractor.extract_batch_features(train_pairs, entity_store)
        X_val, valid_val_pairs = extractor.extract_batch_features(val_pairs, entity_store)

        print(f"   Feature Matrix X_train: shape={X_train.shape}, dtype={X_train.dtype}")
        print(f"   Feature Matrix X_val:   shape={X_val.shape}, dtype={X_val.dtype}")

        # Rule-Based Baseline Benchmark
        print("\n7. Evaluating RuleBasedBaseline Benchmark...")
        rule_baseline = RuleBasedBaseline()
        rule_val_probs = rule_baseline.predict_proba(X_val)
        rule_val_pred = (rule_val_probs >= 0.50).astype(int)
        eval_model_helper = EntityMatchingModel(C=1.0, class_weight="balanced", random_state=42)
        rule_metrics = eval_model_helper.evaluate_pairwise(val_y, rule_val_pred)
        print(f"   RuleBasedBaseline Benchmark: Precision={rule_metrics['precision']:.4f}, Recall={rule_metrics['recall']:.4f}, F0.5={rule_metrics['f05']:.4f}")

        # Train Model
        print("\n8. Training EntityMatchingModel (LogisticRegression + StandardScaler)...")
        t_train0 = time.perf_counter()
        model = EntityMatchingModel(C=1.0, class_weight="balanced", random_state=42)
        model.fit(X_train, train_y)
        model_train_time = time.perf_counter() - t_train0
        print(f"   Model training complete in {model_train_time:.2f}s.")

        # Predict Validation Probabilities
        print("\n9. Predicting Validation Probabilities...")
        val_probs = model.predict_proba(X_val)

        # Threshold Experiments
        print("\n10. Running Threshold Experiments (Selecting by Validation S1-level F0.5)...")
        exp_results = model.run_threshold_experiment(
            val_pairs=valid_val_pairs,
            val_probs=val_probs,
            val_y_true=val_y,
            val_gt=val_gt,
            val_s1_ids=set(val_s1_subset),
            thresholds=[0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]
        )

        best_t = exp_results["best_threshold"]
        best_s1_f05 = exp_results["best_validation_s1_f05"]
        print(f"   Optimal Threshold: {best_t:.2f} | Best Validation S1-level F0.5: {best_s1_f05:.4f}")

        # Final Validation Evaluation at Best Threshold
        val_y_pred = (val_probs >= best_t).astype(int)
        val_pairwise = model.evaluate_pairwise(val_y, val_y_pred)
        val_s1_metrics = model.evaluate_s1_metrics(valid_val_pairs, val_probs, best_t, val_gt, set(val_s1_subset))
        cardinality_breakdown = evaluate_s1_cardinality_breakdown(valid_val_pairs, val_probs, best_t, val_gt, set(val_s1_subset))

        # Error Analysis
        print("\n11. Performing Error Analysis with Categorical Tagging...")
        error_analysis = run_error_analysis(valid_val_pairs, val_y, val_probs, best_t, entity_store)
        valid_train_pairs_count = len(valid_train_pairs)
        valid_val_pairs_count = len(valid_val_pairs)

    # Compile Final Report
    report = {
        "mode": "SMOKE TEST" if is_smoke_test else "FULL VALIDATION",
        "pipeline": "STREAMING (CHUNKED)" if use_streaming else "MONOLITHIC IN-MEMORY",
        "sample_counts": {
            "train_s1_entities": len(train_s1_subset),
            "val_s1_entities": len(val_s1_subset),
            "train_pairs": valid_train_pairs_count,
            "val_pairs": valid_val_pairs_count
        },
        "timing_seconds": {
            "index_build_time_s2": build_times.get("source2", 0.0),
            "index_build_time_s3": build_times.get("source3", 0.0),
            "total_index_build_time": total_build_time,
            "candidate_generation_time": cand_gen_time,
            "model_training_time": model_train_time
        },
        "candidate_blocking_metrics": cand_recall_metrics,
        "feature_list": extractor.get_feature_names(),
        "num_features": len(extractor.get_feature_names()),
        "rule_based_baseline_metrics": rule_metrics,
        "threshold_experiment": exp_results,
        "best_threshold": best_t,
        "s1_level_metrics": {
            "s1_precision": val_s1_metrics["s1_precision"],
            "s1_recall": val_s1_metrics["s1_recall"],
            "s1_f05": val_s1_metrics["s1_f05"],
            "s1_f1": val_s1_metrics["s1_f1"],
            "s1_tp": val_s1_metrics["s1_tp"],
            "s1_fp": val_s1_metrics["s1_fp"],
            "s1_fn": val_s1_metrics["s1_fn"],
            "s1_exact_match_rate_pct": val_s1_metrics["s1_exact_match_rate_pct"],
            "exact_matched_s1_entities": val_s1_metrics["exact_matched_s1_entities"],
            "total_eval_s1_entities": val_s1_metrics["total_s1_entities"]
        },
        "pairwise_diagnostic_metrics": {
            "pairwise_precision": val_pairwise["precision"],
            "pairwise_recall": val_pairwise["recall"],
            "pairwise_f05": val_pairwise["f05"],
            "pairwise_f1": val_pairwise["f1"],
            "confusion_matrix": val_pairwise["confusion_matrix"]
        },
        "s1_cardinality_breakdown": cardinality_breakdown,
        "error_analysis": error_analysis
    }

    report_path = "experiments/matching_validation_report.json"
    os.makedirs("experiments", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved structured report to: {report_path}")

    # Print Report Summary
    print("\n=================== VALIDATION RESULTS SUMMARY ===================")
    print(f"Mode:                                 {'SMOKE TEST' if is_smoke_test else 'FULL VALIDATION'} ({'STREAMING' if use_streaming else 'MONOLITHIC'})")
    print(f"Index Build Time:                     {total_build_time:.2f}s (S2: {build_times['source2']:.2f}s, S3: {build_times['source3']:.2f}s)")
    print(f"Candidate Generation Time:            {cand_gen_time:.2f}s")
    print(f"Model Training Time:                  {model_train_time:.2f}s")
    print(f"Candidate Pairs Processed:            Train: {valid_train_pairs_count:,} | Val: {valid_val_pairs_count:,}")
    print(f"Candidate Combined Recall:            {cand_recall_metrics['combined_link_level_recall'] * 100:.2f}% (S2: {cand_recall_metrics['link_level_s2_recall'] * 100:.2f}%, S3: {cand_recall_metrics['link_level_s3_recall'] * 100:.2f}%)")
    print(f"S1 Complete-Candidate Coverage:       {cand_recall_metrics['s1_complete_coverage_pct']:.2f}% ({cand_recall_metrics['s1_complete_coverage_count']:,} / {cand_recall_metrics['total_val_s1_entities']:,})")
    print(f"RuleBasedBaseline Benchmark F0.5:     {rule_metrics['f05']:.4f}")
    print(f"Optimal Threshold (S1-level F0.5):    {best_t:.2f}")
    print(f"S1-level F0.5 Score (OFFICIAL TARGET): {val_s1_metrics['s1_f05']:.4f}")
    print(f"S1-level F1 Score:                    {val_s1_metrics['s1_f1']:.4f}")
    print(f"S1-level Precision:                   {val_s1_metrics['s1_precision']:.4f}")
    print(f"S1-level Recall:                      {val_s1_metrics['s1_recall']:.4f}")
    print(f"S1 Exact Match Rate:                 {val_s1_metrics['s1_exact_match_rate_pct']:.2f}% ({val_s1_metrics['exact_matched_s1_entities']:,} / {val_s1_metrics['total_s1_entities']:,})")
    print(f"Pairwise Precision (Diagnostic):     {val_pairwise['precision']:.4f}")
    print(f"Pairwise Recall (Diagnostic):        {val_pairwise['recall']:.4f}")
    print(f"Pairwise F0.5 (Diagnostic):          {val_pairwise['f05']:.4f}")
    print(f"Pairwise F1 (Diagnostic):            {val_pairwise['f1']:.4f}")
    print(f"Cardinality Breakdown:")
    print(f"  - zero-match S1 entities:          {cardinality_breakdown['zero-match']['exact_match_rate_pct']:.2f}% ({cardinality_breakdown['zero-match']['exact_matches']}/{cardinality_breakdown['zero-match']['total_entities']})")
    print(f"  - single-match S1 entities:        {cardinality_breakdown['single-match']['exact_match_rate_pct']:.2f}% ({cardinality_breakdown['single-match']['exact_matches']}/{cardinality_breakdown['single-match']['total_entities']})")
    print(f"  - multi-match S1 entities:         {cardinality_breakdown['multi-match']['exact_match_rate_pct']:.2f}% ({cardinality_breakdown['multi-match']['exact_matches']}/{cardinality_breakdown['multi-match']['total_entities']})")
    print(f"Error Categories Breakdown:           {error_analysis['summary']['error_category_counts']}")
    print("==================================================================\n")


if __name__ == "__main__":
    main()


