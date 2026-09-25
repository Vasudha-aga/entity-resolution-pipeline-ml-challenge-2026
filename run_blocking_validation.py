"""
Validation Runner for Candidate Generation and Blocking (B1 + B2 + B3 + B4).

Evaluates all four passes:
- B1: Exact Normalized Business Name + Country
- B2: Distinctive Business Name Token + Country (with max_token_doc_frequency=5000 pruning)
- B3: Character Prefix (N=4) + Country
- B4: Name Token + Address/Locality (Additive)
- Cumulative B1 + B2
- Cumulative B1 + B2 + B3
- Cumulative B1 + B2 + B3 + B4

Reports recall, candidate volume distributions, index statistics, memory RSS telemetry,
missing address statistics, and target achievement (>= 98.0% Recall).
"""

import argparse
import os
import sys
import time
import statistics
import ctypes
from ctypes import wintypes
from typing import Dict, List, Set, Tuple, Any

from src.data_loader import DataLoader, Entity
from src.validation import ValidationSplitter
from src.blocking import BlockingConfig, CandidateGenerator


# Windows OS process-level memory query
class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def get_process_rss_mb() -> Tuple[float, float]:
    """Return (current_rss_mb, peak_rss_mb) for current process on Windows."""
    try:
        psapi = ctypes.windll.kernel32
        func = getattr(psapi, "K32GetProcessMemoryInfo", ctypes.windll.psapi.GetProcessMemoryInfo)
        func.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        func.restype = wintypes.BOOL

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        if func(psapi.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return (
                round(counters.WorkingSetSize / (1024 * 1024), 2),
                round(counters.PeakWorkingSetSize / (1024 * 1024), 2)
            )
    except Exception:
        pass
    return 0.0, 0.0


def compute_distribution_stats(counts: List[int]) -> Dict[str, Any]:
    """Compute distribution metrics for candidate counts."""
    if not counts:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0,
            "max": 0,
            "p95": 0,
            "p99": 0,
            "zero_pct": 0.0,
            "gt_100_pct": 0.0,
        }

    n = len(counts)
    sorted_counts = sorted(counts)
    return {
        "mean": round(statistics.mean(counts), 2),
        "median": round(statistics.median(counts), 2),
        "min": min(counts),
        "max": max(counts),
        "p95": sorted_counts[int(0.95 * n)] if n > 0 else 0,
        "p99": sorted_counts[int(0.99 * n)] if n > 0 else 0,
        "zero_pct": round((sum(1 for c in counts if c == 0) / n) * 100, 2),
        "gt_100_pct": round((sum(1 for c in counts if c > 100) / n) * 100, 2),
    }


def load_validation_s1_sample(
    loader: DataLoader,
    val_s1_ids: Set[str],
    sample_size: int = 5000
) -> List[Entity]:
    """Stream and collect a deterministic sample of validation S1 entities."""
    val_entities: List[Entity] = []
    s1_train_path = loader.get_file_path("train", "source1")

    for chunk in loader.stream_entities(s1_train_path, chunk_size=100000):
        for entity in chunk:
            if entity.entity_id in val_s1_ids:
                val_entities.append(entity)
                if sample_size > 0 and len(val_entities) >= sample_size:
                    return val_entities
    return val_entities


class BlockingEvaluator:
    """Evaluates multi-pass candidate generation performance against ground truth."""

    def __init__(
        self,
        generator: CandidateGenerator,
        val_ground_truth: Dict[str, List[str]]
    ):
        self.generator = generator
        self.val_gt = val_ground_truth

    def evaluate(self, val_entities: List[Entity]) -> Dict[str, Any]:
        """Run evaluation on validation S1 entities across all passes."""
        total_eval_entities = len(val_entities)

        # Ground truth match tracking
        total_true_s2 = 0
        total_true_s3 = 0
        total_true_comb = 0

        pass_keys = [
            "b1",
            "b2",
            "b3",
            "b4",
            "cumulative_b1_b2",
            "cumulative_b1_b2_b3",
            "cumulative"
        ]

        # Candidate hits tracking
        hits = {p: {"s2": 0, "s3": 0, "comb": 0} for p in pass_keys}

        # Candidate count lists for distribution tracking
        counts = {p: {"s2": [], "s3": [], "total": []} for p in pass_keys}

        entities_with_zero_gt = 0
        query_start = time.perf_counter()

        for entity in val_entities:
            s1_id = entity.entity_id
            true_matches = self.val_gt.get(s1_id, [])

            true_s2 = set(m for m in true_matches if m.startswith("S2-"))
            true_s3 = set(m for m in true_matches if m.startswith("S3-"))
            true_comb = true_s2.union(true_s3)

            len_s2 = len(true_s2)
            len_s3 = len(true_s3)
            len_comb = len(true_comb)

            if len_comb == 0:
                entities_with_zero_gt += 1
            else:
                total_true_s2 += len_s2
                total_true_s3 += len_s3
                total_true_comb += len_comb

            # Generate candidates partitioned by pass
            passes = self.generator.generate_candidates_passes(entity)

            for pass_name in pass_keys:
                s2_cands, s3_cands = passes[pass_name]
                s2_set = set(s2_cands)
                s3_set = set(s3_cands)

                # Record candidate counts
                c2 = len(s2_set)
                c3 = len(s3_set)
                counts[pass_name]["s2"].append(c2)
                counts[pass_name]["s3"].append(c3)
                counts[pass_name]["total"].append(c2 + c3)

                # Compute hits
                if len_s2 > 0:
                    hits[pass_name]["s2"] += len(s2_set.intersection(true_s2))
                if len_s3 > 0:
                    hits[pass_name]["s3"] += len(s3_set.intersection(true_s3))
                if len_comb > 0:
                    hits[pass_name]["comb"] += (
                        len(s2_set.intersection(true_s2)) + len(s3_set.intersection(true_s3))
                    )

        total_query_time = time.perf_counter() - query_start

        # Compute recall percentages
        recall_results = {}
        for p in pass_keys:
            recall_results[p] = {
                "s2_recall": round((hits[p]["s2"] / total_true_s2 * 100), 2) if total_true_s2 > 0 else 0.0,
                "s2_hits": hits[p]["s2"],
                "s2_true": total_true_s2,
                "s3_recall": round((hits[p]["s3"] / total_true_s3 * 100), 2) if total_true_s3 > 0 else 0.0,
                "s3_hits": hits[p]["s3"],
                "s3_true": total_true_s3,
                "comb_recall": round((hits[p]["comb"] / total_true_comb * 100), 2) if total_true_comb > 0 else 0.0,
                "comb_hits": hits[p]["comb"],
                "comb_true": total_true_comb,
            }

        # Compute candidate distributions
        dist_stats = {}
        for p in pass_keys:
            dist_stats[p] = {
                "s2": compute_distribution_stats(counts[p]["s2"]),
                "s3": compute_distribution_stats(counts[p]["s3"]),
                "total": compute_distribution_stats(counts[p]["total"]),
            }

        latency_ms = (total_query_time / total_eval_entities * 1000) if total_eval_entities > 0 else 0.0
        throughput = (total_eval_entities / total_query_time) if total_query_time > 0 else 0.0

        return {
            "evaluated_entities": total_eval_entities,
            "zero_gt_entities": entities_with_zero_gt,
            "query_time_seconds": round(total_query_time, 3),
            "latency_ms": round(latency_ms, 3),
            "throughput": round(throughput, 1),
            "recall": recall_results,
            "candidate_stats": dist_stats,
        }


def print_report(
    results: Dict[str, Any],
    build_times: Dict[str, float],
    mem_stats: Dict[str, Any],
    s2_stats: Dict[str, Any],
    s3_stats: Dict[str, Any],
    s2_top_b4: List[Tuple[str, int]],
    s3_top_b4: List[Tuple[str, int]]
) -> None:
    """Print clean formatted validation report."""
    rec = results["recall"]
    cand = results["candidate_stats"]

    print("\n" + "=" * 98)
    print(" Amazon ML Challenge 2026: Complete Multi-Pass Blocking Validation (B1 + B2 + B3 + B4)")
    print(" Pass 1 (B1): Exact Normalized Name + Country")
    print(" Pass 2 (B2): Distinctive Token + Country (Max Doc Frequency: 5000)")
    print(" Pass 3 (B3): Character Prefix (N=4) + Country")
    print(" Pass 4 (B4): Name Token + Address/Locality (Additive)")
    print(" Target: >= 98.0% Cumulative Recall | Mode: Uncapped Raw Candidates")
    print("=" * 98)

    # 1. Individual Pass Recall
    print("\n--- A. INDIVIDUAL PASS RECALL ---")
    print(f"{'Pass Configuration':<32} | {'S1 -> S2 Recall':<20} | {'S1 -> S3 Recall':<20} | {'Combined Recall':<20}")
    print("-" * 98)
    r1 = rec["b1"]
    print(f"{'B1: Exact Name + Country':<32} | {r1['s2_recall']}% ({r1['s2_hits']:,}/{r1['s2_true']:,}){'':<2} | {r1['s3_recall']}% ({r1['s3_hits']:,}/{r1['s3_true']:,}){'':<2} | {r1['comb_recall']}% ({r1['comb_hits']:,}/{r1['comb_true']:,})")
    r2 = rec["b2"]
    print(f"{'B2: Distinctive Token + Country':<32} | {r2['s2_recall']}% ({r2['s2_hits']:,}/{r2['s2_true']:,}){'':<2} | {r2['s3_recall']}% ({r2['s3_hits']:,}/{r2['s3_true']:,}){'':<2} | {r2['comb_recall']}% ({r2['comb_hits']:,}/{r2['comb_true']:,})")
    r3 = rec["b3"]
    print(f"{'B3: Name Prefix (N=4) + Country':<32} | {r3['s2_recall']}% ({r3['s2_hits']:,}/{r3['s2_true']:,}){'':<2} | {r3['s3_recall']}% ({r3['s3_hits']:,}/{r3['s3_true']:,}){'':<2} | {r3['comb_recall']}% ({r3['comb_hits']:,}/{r3['comb_true']:,})")
    r4 = rec["b4"]
    print(f"{'B4: Name Token + Address':<32} | {r4['s2_recall']}% ({r4['s2_hits']:,}/{r4['s2_true']:,}){'':<2} | {r4['s3_recall']}% ({r4['s3_hits']:,}/{r4['s3_true']:,}){'':<2} | {r4['comb_recall']}% ({r4['comb_hits']:,}/{r4['comb_true']:,})")
    print("-" * 98)

    # 2. Cumulative Recall
    print("\n--- B. CUMULATIVE RECALL PROGRESSION ---")
    print(f"{'Cumulative Passes':<32} | {'S1 -> S2 Recall':<20} | {'S1 -> S3 Recall':<20} | {'Combined Recall':<20}")
    print("-" * 98)
    r12 = rec["cumulative_b1_b2"]
    print(f"{'B1 + B2':<32} | {r12['s2_recall']}% ({r12['s2_hits']:,}/{r12['s2_true']:,}){'':<2} | {r12['s3_recall']}% ({r12['s3_hits']:,}/{r12['s3_true']:,}){'':<2} | {r12['comb_recall']}% ({r12['comb_hits']:,}/{r12['comb_true']:,})")
    r123 = rec["cumulative_b1_b2_b3"]
    print(f"{'B1 + B2 + B3':<32} | {r123['s2_recall']}% ({r123['s2_hits']:,}/{r123['s2_true']:,}){'':<2} | {r123['s3_recall']}% ({r123['s3_hits']:,}/{r123['s3_true']:,}){'':<2} | {r123['comb_recall']}% ({r123['comb_hits']:,}/{r123['comb_true']:,})")
    r_all = rec["cumulative"]
    target_met = r_all['comb_recall'] >= 98.0
    status_tag = ">= 98.0% [TARGET ACHIEVED]" if target_met else "Intermediate Milestone"
    print(f"{'B1 + B2 + B3 + B4 (FINAL)':<32} | {r_all['s2_recall']}% ({r_all['s2_hits']:,}/{r_all['s2_true']:,}){'':<2} | {r_all['s3_recall']}% ({r_all['s3_hits']:,}/{r_all['s3_true']:,}){'':<2} | {r_all['comb_recall']}% ({r_all['comb_hits']:,}/{r_all['comb_true']:,}) [{status_tag}]")
    print("-" * 98)

    # 3. Candidate Volume Distributions
    print("\n--- C. CANDIDATE VOLUME DISTRIBUTION (PER S1 ENTITY) ---")
    print(f"{'Pass':<24} | {'Source':<6} | {'Mean':<8} | {'Median':<6} | {'Min':<4} | {'Max':<8} | {'p95':<6} | {'p99':<6} | {'Zero %':<8} | {'>100 %':<8}")
    print("-" * 98)
    for p_label, p_key in [("B4 Alone", "b4"), ("Cumulative B1..B4", "cumulative")]:
        for src in ["s2", "s3", "total"]:
            s = cand[p_key][src]
            print(f"{p_label:<24} | {src.upper():<6} | {s['mean']:<8} | {s['median']:<6} | {s['min']:<4} | {s['max']:<8} | {s['p95']:<6} | {s['p99']:<6} | {s['zero_pct']:<7}% | {s['gt_100_pct']:<7}%")
        if p_key != "cumulative":
            print("-" * 98)
    print("-" * 98)

    # 4. Index Statistics & Process Memory
    print("\n--- D. INDEX STATISTICS & PROCESS MEMORY ---")
    print(f"  - Source 2 Total Entities:         {s2_stats['total_indexed']:,}")
    print(f"  - Source 2 Unique B1 Keys:         {s2_stats['b1_unique_keys']:,}")
    print(f"  - Source 2 Unique B2 Keys:         {s2_stats['b2_unique_keys']:,} (Pruned: {s2_stats['b2_pruned_keys']:,})")
    print(f"  - Source 2 Unique B3 Prefix Keys:  {s2_stats['b3_unique_keys']:,}")
    print(f"  - Source 2 Unique B4 Addr Keys:    {s2_stats['b4_unique_keys']:,} (Postings: {s2_stats['b4_total_postings']:,})")
    print(f"  - Source 3 Total Entities:         {s3_stats['total_indexed']:,}")
    print(f"  - Source 3 Unique B1 Keys:         {s3_stats['b1_unique_keys']:,}")
    print(f"  - Source 3 Unique B2 Keys:         {s3_stats['b2_unique_keys']:,} (Pruned: {s3_stats['b2_pruned_keys']:,})")
    print(f"  - Source 3 Unique B3 Prefix Keys:  {s3_stats['b3_unique_keys']:,}")
    print(f"  - Source 3 Unique B4 Addr Keys:    {s3_stats['b4_unique_keys']:,} (Postings: {s3_stats['b4_total_postings']:,})")
    print(f"  - Total B4 Postings Across Both:   {s2_stats['b4_total_postings'] + s3_stats['b4_total_postings']:,}")
    print(f"  - Source 2 Build Time:             {build_times.get('source2', 0.0):.2f}s")
    print(f"  - Source 3 Build Time:             {build_times.get('source3', 0.0):.2f}s")
    print(f"  - Total Build Time:                {sum(build_times.values()):.2f}s")
    if mem_stats.get("total_rss"):
        print(f"  - Baseline Process RSS:            {mem_stats.get('baseline_rss', 0):.2f} MB")
        print(f"  - Final (S2+S3) Process RSS:       {mem_stats.get('total_rss', 0):.2f} MB (Peak: {mem_stats.get('total_peak', 0):.2f} MB)")

    # 5. Address Diagnostics
    print("\n--- E. ADDRESS DIAGNOSTICS & MISSINGNESS ---")
    print(f"  - S2 Missing Addresses:            {s2_stats['missing_addresses']:,} ({s2_stats['missing_address_pct']}%)")
    print(f"  - S3 Missing Addresses:            {s3_stats['missing_addresses']:,} ({s3_stats['missing_address_pct']}%)")
    print(f"  Top 10 Most Frequent B4 Name+Address Keys in Source 2:")
    for idx, (k, cnt) in enumerate(s2_top_b4[:10], 1):
        print(f"    {idx:2d}. {k:<38} ({cnt:,} entities)")

    print(f"\n  Top 10 Most Frequent B4 Name+Address Keys in Source 3:")
    for idx, (k, cnt) in enumerate(s3_top_b4[:10], 1):
        print(f"    {idx:2d}. {k:<38} ({cnt:,} entities)")

    # 6. Query Performance
    print("\n--- F. QUERY LATENCY & THROUGHPUT ---")
    print(f"  - Evaluated Entities:              {results['evaluated_entities']:,} validation S1 records")
    print(f"  - Zero GT Entities:                {results['zero_gt_entities']:,} (singleton entities)")
    print(f"  - Total Query Time:                {results['query_time_seconds']} seconds")
    print(f"  - Query Latency:                   {results['latency_ms']} ms / entity")
    print(f"  - Query Throughput:                {results['throughput']:,} entities / sec")

    # 7. Final Target Summary
    print("\n--- G. RECALL TARGET VERIFICATION ---")
    print(f"  - Final Raw Cumulative Recall (B1+B2+B3+B4): {r_all['comb_recall']}%")
    print(f"  - Target Threshold:                          >= 98.0%")
    print(f"  - Target Status:                             {'MET / EXCEEDED' if target_met else 'NOT YET MET'}")
    print("=" * 98 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate complete blocking pipeline (B1 + B2 + B3 + B4).")
    parser.add_argument("--sample-size", type=int, default=5000,
                        help="Number of validation S1 entities to evaluate (default: 5,000).")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Root data directory.")
    parser.add_argument("--splits-dir", type=str, default="experiments/splits",
                        help="Splits directory.")
    args = parser.parse_args()

    print("=========================================================")
    print(" Amazon ML Challenge 2026: Complete Blocking Validation  ")
    print("=========================================================\n")

    loader = DataLoader(data_dir=args.data_dir)
    splitter = ValidationSplitter(data_loader=loader)

    # 1. Baseline Memory
    base_rss, base_peak = get_process_rss_mb()
    mem_stats: Dict[str, Any] = {"baseline_rss": base_rss}
    print(f"[0/4] Baseline Python Process RSS: {base_rss:.2f} MB")

    # 2. Load Validation IDs and Ground Truth
    val_split_path = os.path.join(args.splits_dir, "val_s1_ids.txt")
    val_s1_ids = splitter.load_split_ids(val_split_path)
    print(f"[1/4] Loaded {len(val_s1_ids):,} frozen validation S1 IDs.")

    all_gt = loader.load_ground_truth()
    val_gt = splitter.filter_ground_truth_for_split(all_gt, val_s1_ids)
    print(f"[2/4] Filtered {len(val_gt):,} validation ground-truth entries.")

    # 3. Build Complete Indexes (B1 + B2 + B3 + B4)
    print("[3/4] Streaming S2 and S3 to build B1 + B2 + B3 + B4 inverted indexes...")
    config = BlockingConfig(
        enabled_passes=[
            "exact_name_country",
            "distinctive_token_country",
            "name_prefix_country",
            "name_token_address"
        ],
        max_candidates_per_source=None,
        max_token_doc_frequency=5000,
        min_token_len=3,
        prefix_len=4,
        b4_name_min_len=2,
        b4_addr_min_len=4,
        b4_max_name_tokens=2,
        b4_max_addr_tokens=3,
        chunk_size=100000
    )
    generator = CandidateGenerator(data_loader=loader, config=config)

    # Stream S2
    s2_path = loader.get_file_path("train", "source2")
    t0 = time.perf_counter()
    for batch in loader.stream_entities(s2_path, chunk_size=config.chunk_size):
        generator.s2_index.add_batch(batch)
    generator.s2_index.prune_b2_high_frequency_keys()
    t_s2 = time.perf_counter() - t0

    # Stream S3
    s3_path = loader.get_file_path("train", "source3")
    t0 = time.perf_counter()
    for batch in loader.stream_entities(s3_path, chunk_size=config.chunk_size):
        generator.s3_index.add_batch(batch)
    generator.s3_index.prune_b2_high_frequency_keys()
    t_s3 = time.perf_counter() - t0

    tot_rss, tot_peak = get_process_rss_mb()
    mem_stats["total_rss"] = tot_rss
    mem_stats["total_peak"] = tot_peak

    build_times = {"source2": t_s2, "source3": t_s3}
    s2_stats = generator.s2_index.get_stats()
    s3_stats = generator.s3_index.get_stats()

    s2_top_b4 = generator.s2_index.get_top_b4_keys(top_n=10)
    s3_top_b4 = generator.s3_index.get_top_b4_keys(top_n=10)

    # 4. Stream Validation Sample
    sample_size = args.sample_size
    print(f"[4/4] Streaming {sample_size:,} validation S1 entities for evaluation...")
    val_entities = load_validation_s1_sample(loader, val_s1_ids, sample_size=sample_size)
    print(f"      Collected {len(val_entities):,} validation S1 entities.")

    # 5. Evaluate
    print("\nRunning multi-pass candidate evaluation (B1, B2, B3, B4)...")
    evaluator = BlockingEvaluator(generator=generator, val_ground_truth=val_gt)
    results = evaluator.evaluate(val_entities)

    # 6. Report
    print_report(
        results,
        build_times,
        mem_stats,
        s2_stats,
        s3_stats,
        s2_top_b4,
        s3_top_b4
    )


if __name__ == "__main__":
    main()
