"""
Execution runner for Pairwise Matching Model & Validation Pipeline.

Pipeline Steps:
1. Loads S1 train/validation split (experiments/splits/train_s1_ids.txt & val_s1_ids.txt).
2. Loads ground truth mappings.
3. Pre-normalizes entity records in memory using src.normalization.
4. Generates candidate pairs for training and validation.
5. Extracts 25-dimensional pairwise feature vectors in memory-safe batches.
6. Evaluates RuleBasedBaseline benchmark.
7. Trains EntityMatchingModel on train split entities only.
8. Evaluates predictions on validation split entities.
9. Runs decision threshold experiments (0.30 to 0.90) selecting best F0.5 threshold.
10. Evaluates S1 cardinality performance (zero-match, single-match, multi-match).
11. Performs Error Analysis with deterministic categorical tagging.
12. Saves JSON report to experiments/matching_validation_report.json and prints summary.

Supports --smoke-test flag for fast, memory-safe validation verification.
"""

import os
import sys
import csv
import json
import random
import argparse
from collections import Counter
from typing import Dict, List, Tuple, Set, Any, Optional
import numpy as np

from src.data_loader import DataLoader, Entity
from src.validation import ValidationSplitter
from src.normalization import TextNormalizer
from src.features import PairwiseFeatureExtractor, NormalizedEntity
from src.matching_model import EntityMatchingModel, RuleBasedBaseline, fbeta_score


def load_normalized_entity_store(
    loader: DataLoader,
    normalizer: TextNormalizer,
    target_s1_ids: Optional[Set[str]] = None,
    target_s2_ids: Optional[Set[str]] = None,
    target_s3_ids: Optional[Set[str]] = None
) -> Dict[str, NormalizedEntity]:
    """
    Load and pre-normalize entity records for requested target IDs across S1, S2, and S3.
    """
    entity_store: Dict[str, NormalizedEntity] = {}

    # Stream Source 1
    s1_file = loader.get_file_path("train", "source1")
    with open(s1_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        _ = next(reader, None)
        for row in reader:
            if not row:
                continue
            eid = row[0].strip()
            if target_s1_ids is None or eid in target_s1_ids:
                e = Entity.from_row(row)
                entity_store[eid] = NormalizedEntity.from_entity(e, normalizer)

    # Stream Source 2
    if target_s2_ids is None or len(target_s2_ids) > 0:
        s2_file = loader.get_file_path("train", "source2")
        with open(s2_file, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            _ = next(reader, None)
            for row in reader:
                if not row:
                    continue
                eid = row[0].strip()
                if target_s2_ids is None or eid in target_s2_ids:
                    e = Entity.from_row(row)
                    entity_store[eid] = NormalizedEntity.from_entity(e, normalizer)
                    if target_s2_ids is not None and len(entity_store) >= (len(target_s1_ids or []) + len(target_s2_ids)):
                        break

    # Stream Source 3
    if target_s3_ids is None or len(target_s3_ids) > 0:
        s3_file = loader.get_file_path("train", "source3")
        with open(s3_file, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            _ = next(reader, None)
            for row in reader:
                if not row:
                    continue
                eid = row[0].strip()
                if target_s3_ids is None or eid in target_s3_ids:
                    e = Entity.from_row(row)
                    entity_store[eid] = NormalizedEntity.from_entity(e, normalizer)

    return entity_store


def generate_candidate_pairs(
    s1_ids: Set[str],
    gt_mapping: Dict[str, List[str]],
    all_target_cand_ids: List[str],
    neg_ratio: int = 3,
    seed: int = 42
) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    """
    Generate candidate pairs for S1 entities.
    """
    rng = random.Random(seed)
    pairs: List[Tuple[str, str]] = []
    labels: List[int] = []

    for s1_id in s1_ids:
        pos_matches = gt_mapping.get(s1_id, [])
        for pos_id in pos_matches:
            pairs.append((s1_id, pos_id))
            labels.append(1)

        # Sample negatives
        n_neg = max(1, len(pos_matches) * neg_ratio)
        sampled_negs = rng.sample(all_target_cand_ids, min(n_neg, len(all_target_cand_ids)))
        for neg_id in sampled_negs:
            if neg_id not in pos_matches:
                pairs.append((s1_id, neg_id))
                labels.append(0)

    return pairs, np.array(labels, dtype=np.int32)


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
    Categorize error sample deterministically based ONLY on dataset fields and features:
    - missing_address: Address missing in S1 or candidate.
    - legal_suffix: Names differ only by legal corporate entity designator tokens.
    - spelling_typo: Name tokens share high character 3-gram similarity (>= 0.75) but differ slightly.
    - abbreviation: Token length difference where one token is a prefix/abbreviation of another.
    - generic_name: Name consists of short or generic words (e.g., 'Coffee Shop', 'Pharmacy').
    - address_conflict: Names match strongly (>= 0.70) but non-empty addresses conflict (< 0.20 Jaccard).
    - transliteration: Raw names differ in non-ASCII representation but normalized base ASCII matches.
    - uncategorized: Fallback when no specific deterministic category rule triggers.
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


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Pairwise Matching Model & Validation Runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run small smoke test on subset of data")
    parser.add_argument("--sample-s1-train", type=int, default=2000, help="Number of S1 train entities for smoke test")
    parser.add_argument("--sample-s1-val", type=int, default=500, help="Number of S1 val entities for smoke test")
    args = parser.parse_args()

    is_smoke_test = args.smoke_test or ("--full" not in sys.argv)

    print("=========================================================")
    print(f" Amazon ML Challenge 2026: Pairwise Matching Validation ")
    print(f" Mode: {'SMOKE TEST' if is_smoke_test else 'FULL VALIDATION'} ")
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

    if is_smoke_test:
        train_s1_subset = set(sorted(list(all_train_s1))[:args.sample_s1_train])
        val_s1_subset = set(sorted(list(all_val_s1))[:args.sample_s1_val])
        print(f"   [SMOKE TEST] Subsetting to {len(train_s1_subset):,} Train S1 and {len(val_s1_subset):,} Val S1 entities.")
    else:
        train_s1_subset = all_train_s1
        val_s1_subset = all_val_s1

    print("\n2. Loading Ground Truth mapping...")
    gt_mapping = loader.load_ground_truth()
    train_gt = splitter.filter_ground_truth_for_split(gt_mapping, train_s1_subset)
    val_gt = splitter.filter_ground_truth_for_split(gt_mapping, val_s1_subset)
    print(f"   Loaded GT for {len(train_gt):,} Train S1 and {len(val_gt):,} Val S1 entities.")

    # Collect needed candidate IDs
    train_needed_cand_ids: Set[str] = set()
    for matches in train_gt.values():
        train_needed_cand_ids.update(matches)

    val_needed_cand_ids: Set[str] = set()
    for matches in val_gt.values():
        val_needed_cand_ids.update(matches)

    print(f"   Needed Target Candidates: Train={len(train_needed_cand_ids):,} | Val={len(val_needed_cand_ids):,}")

    # Build Candidate Pairs
    print("\n3. Generating Candidate Pairs (Positives + Sampled Negatives)...")
    train_cand_pool = list(train_needed_cand_ids)
    val_cand_pool = list(val_needed_cand_ids)

    train_pairs, train_y = generate_candidate_pairs(train_s1_subset, train_gt, train_cand_pool, neg_ratio=3, seed=42)
    val_pairs, val_y = generate_candidate_pairs(val_s1_subset, val_gt, val_cand_pool, neg_ratio=3, seed=42)

    print(f"   Generated Train pairs: {len(train_pairs):,} (Positives: {int(np.sum(train_y)):,})")
    print(f"   Generated Val pairs:   {len(val_pairs):,} (Positives: {int(np.sum(val_y)):,})")

    # Load normalized entities
    needed_s1 = train_s1_subset.union(val_s1_subset)
    needed_s2 = {c for c in train_needed_cand_ids.union(val_needed_cand_ids) if c.startswith("S2-")}
    needed_s3 = {c for c in train_needed_cand_ids.union(val_needed_cand_ids) if c.startswith("S3-")}

    print("\n4. Loading & Pre-normalizing Entities into Memory Store...")
    entity_store = load_normalized_entity_store(
        loader,
        normalizer,
        target_s1_ids=needed_s1,
        target_s2_ids=needed_s2,
        target_s3_ids=needed_s3
    )
    print(f"   Normalized entity store built: {len(entity_store):,} entities.")

    # Extract Features
    print("\n5. Extracting Pairwise Features (25 dimensions)...")
    X_train, valid_train_pairs = extractor.extract_batch_features(train_pairs, entity_store)
    X_val, valid_val_pairs = extractor.extract_batch_features(val_pairs, entity_store)

    print(f"   Feature Matrix X_train: shape={X_train.shape}, dtype={X_train.dtype}")
    print(f"   Feature Matrix X_val:   shape={X_val.shape}, dtype={X_val.dtype}")

    # Rule-Based Baseline Benchmark
    print("\n6. Evaluating RuleBasedBaseline Benchmark...")
    rule_baseline = RuleBasedBaseline()
    rule_val_probs = rule_baseline.predict_proba(X_val)
    rule_val_pred = (rule_val_probs >= 0.50).astype(int)
    rule_pairwise = model = EntityMatchingModel(C=1.0, class_weight="balanced", random_state=42)
    rule_metrics = model.evaluate_pairwise(val_y, rule_val_pred)
    print(f"   RuleBasedBaseline Benchmark: Precision={rule_metrics['precision']:.4f}, Recall={rule_metrics['recall']:.4f}, F0.5={rule_metrics['f05']:.4f}")

    # Train Model
    print("\n7. Training EntityMatchingModel (LogisticRegression + StandardScaler)...")
    model = EntityMatchingModel(C=1.0, class_weight="balanced", random_state=42)
    model.fit(X_train, train_y)
    print("   Model training complete.")

    # Predict Validation Probabilities
    print("\n8. Predicting Validation Probabilities...")
    val_probs = model.predict_proba(X_val)

    # Threshold Experiments
    print("\n9. Running Threshold Experiments (0.30 to 0.90)...")
    exp_results = model.run_threshold_experiment(
        val_pairs=valid_val_pairs,
        val_probs=val_probs,
        val_y_true=val_y,
        val_gt=val_gt,
        val_s1_ids=val_s1_subset,
        thresholds=[0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    )

    best_t = exp_results["best_threshold"]
    best_f05 = exp_results["best_validation_f05"]
    print(f"   Optimal Threshold: {best_t:.2f} | Best Validation Pairwise F0.5: {best_f05:.4f}")

    # Final Validation Evaluation at Best Threshold
    val_y_pred = (val_probs >= best_t).astype(int)
    val_pairwise = model.evaluate_pairwise(val_y, val_y_pred)
    val_s1_exact = model.evaluate_s1_exact_match(valid_val_pairs, val_probs, best_t, val_gt, val_s1_subset)
    cardinality_breakdown = evaluate_s1_cardinality_breakdown(valid_val_pairs, val_probs, best_t, val_gt, val_s1_subset)

    # Error Analysis
    print("\n10. Performing Error Analysis with Categorical Tagging...")
    error_analysis = run_error_analysis(valid_val_pairs, val_y, val_probs, best_t, entity_store)

    # Compile Final Report
    report = {
        "mode": "SMOKE TEST" if is_smoke_test else "FULL VALIDATION",
        "sample_counts": {
            "train_s1_entities": len(train_s1_subset),
            "val_s1_entities": len(val_s1_subset),
            "train_pairs": len(valid_train_pairs),
            "val_pairs": len(valid_val_pairs)
        },
        "feature_list": extractor.get_feature_names(),
        "num_features": len(extractor.get_feature_names()),
        "rule_based_baseline_metrics": rule_metrics,
        "threshold_experiment": exp_results,
        "best_threshold": best_t,
        "validation_metrics": {
            "pairwise_precision": val_pairwise["precision"],
            "pairwise_recall": val_pairwise["recall"],
            "pairwise_f05": val_pairwise["f05"],
            "pairwise_f1": val_pairwise["f1"],
            "confusion_matrix": val_pairwise["confusion_matrix"],
            "s1_exact_match_rate_pct": val_s1_exact["s1_exact_match_rate_pct"],
            "exact_matched_s1_entities": val_s1_exact["exact_matched_s1_entities"],
            "total_eval_s1_entities": val_s1_exact["total_s1_entities"]
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
    print(f"RuleBasedBaseline Benchmark F0.5:  {rule_metrics['f05']:.4f}")
    print(f"Optimal Logistic Regression Threshold: {best_t:.2f}")
    print(f"Pairwise Precision:               {val_pairwise['precision']:.4f}")
    print(f"Pairwise Recall:                  {val_pairwise['recall']:.4f}")
    print(f"Pairwise F0.5-Score (Official):   {val_pairwise['f05']:.4f}")
    print(f"Pairwise F1-Score:                {val_pairwise['f1']:.4f}")
    print(f"S1 Exact Match Rate:              {val_s1_exact['s1_exact_match_rate_pct']:.2f}% ({val_s1_exact['exact_matched_s1_entities']:,} / {val_s1_exact['total_s1_entities']:,})")
    print(f"Cardinality Breakdown:")
    print(f"  - zero-match S1 entities:       {cardinality_breakdown['zero-match']['exact_match_rate_pct']:.2f}% ({cardinality_breakdown['zero-match']['exact_matches']}/{cardinality_breakdown['zero-match']['total_entities']})")
    print(f"  - single-match S1 entities:     {cardinality_breakdown['single-match']['exact_match_rate_pct']:.2f}% ({cardinality_breakdown['single-match']['exact_matches']}/{cardinality_breakdown['single-match']['total_entities']})")
    print(f"  - multi-match S1 entities:      {cardinality_breakdown['multi-match']['exact_match_rate_pct']:.2f}% ({cardinality_breakdown['multi-match']['exact_matches']}/{cardinality_breakdown['multi-match']['total_entities']})")
    print(f"Error Categories Breakdown:        {error_analysis['summary']['error_category_counts']}")
    print("==================================================================\n")


if __name__ == "__main__":
    main()
