"""
Verification script: Compare Streaming vs Monolithic implementation on Smoke Test.
"""

import os
import sys
import time
import json
from collections import defaultdict, Counter
import numpy as np

from src.data_loader import DataLoader, Entity
from src.validation import ValidationSplitter
from src.normalization import TextNormalizer
from src.features import PairwiseFeatureExtractor, NormalizedEntity
from src.matching_model import EntityMatchingModel, RuleBasedBaseline
from src.blocking import CandidateGenerator, BlockingConfig

def run_test():
    loader = DataLoader(data_dir="data")
    splitter = ValidationSplitter(data_loader=loader, seed=42)
    normalizer = TextNormalizer()
    extractor = PairwiseFeatureExtractor(normalizer=normalizer)

    splits_dir = "experiments/splits"
    train_split_path = os.path.join(splits_dir, "train_s1_ids.txt")
    val_split_path = os.path.join(splits_dir, "val_s1_ids.txt")

    all_train_s1 = splitter.load_split_ids(train_split_path)
    all_val_s1 = splitter.load_split_ids(val_split_path)

    sample_s1_train = 2000
    sample_s1_val = 500
    smoke_target_size = 20000

    train_s1_subset = list(sorted(list(all_train_s1))[:sample_s1_train])
    val_s1_subset = list(sorted(list(all_val_s1))[:sample_s1_val])
    all_selected_s1 = set(train_s1_subset).union(set(val_s1_subset))

    def load_smoke_ground_truth(file_path, target_s1_ids):
        gt = {}
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            f.readline()
            for line in f:
                if not line: continue
                tab = line.find("\t")
                if tab == -1: continue
                sid = line[:tab].strip()
                if sid in target_s1_ids:
                    m = line[tab+1:].strip()
                    gt[sid] = [x.strip() for x in m.split(",") if x.strip()]
        return gt

    gt_file = loader.get_file_path("train", "ground_truth")
    gt_mapping = load_smoke_ground_truth(gt_file, all_selected_s1)

    required_s2_ids = set()
    required_s3_ids = set()
    for sid in all_selected_s1:
        for mid in gt_mapping.get(sid, []):
            if mid.startswith("S2-"): required_s2_ids.add(mid)
            elif mid.startswith("S3-"): required_s3_ids.add(mid)

    def build_subset(source, req_ids):
        tf = loader.get_file_path("train", source)
        ts = max(smoke_target_size, len(req_ids))
        req = []
        found = set()
        dist = []
        with open(tf, "r", encoding="utf-8", errors="replace") as f:
            f.readline()
            for line in f:
                if not line: continue
                tab = line.find("\t")
                if tab == -1: continue
                eid = line[:tab].strip()
                is_req = eid in req_ids
                has_enough = (len(req) + len(dist)) >= ts
                if is_req:
                    req.append(Entity.from_row(line.rstrip("\r\n").split("\t")))
                    found.add(eid)
                elif not has_enough:
                    dist.append(Entity.from_row(line.rstrip("\r\n").split("\t")))
                if len(found) == len(req_ids) and (len(req) + len(dist)) >= ts:
                    break
        return req + dist

    s2_ents = build_subset("source2", required_s2_ids)
    s3_ents = build_subset("source3", required_s3_ids)

    preloaded_entities = {}
    for e in s2_ents + s3_ents:
        preloaded_entities[e.entity_id] = e

    def load_s1(target_ids):
        s1f = loader.get_file_path("train", "source1")
        res = {}
        target_set = set(target_ids)
        with open(s1f, "r", encoding="utf-8", errors="replace") as f:
            f.readline()
            for line in f:
                if not line: continue
                tab = line.find("\t")
                if tab == -1: continue
                eid = line[:tab].strip()
                if eid in target_set:
                    res[eid] = Entity.from_row(line.rstrip("\r\n").split("\t"))
                    if len(res) == len(target_set): break
        return res

    all_s1_dict = load_s1(all_selected_s1)
    for e in all_s1_dict.values():
        preloaded_entities[e.entity_id] = e

    cand_gen = CandidateGenerator(data_loader=loader, config=BlockingConfig())
    cand_gen.build_indexes_from_entities(s2_ents, s3_ents)

    # 1. Monolithic Baseline Run
    print("\n--- Running Monolithic Smoke Test ---")
    t0 = time.perf_counter()
    train_s1_entities = {s: all_s1_dict[s] for s in train_s1_subset}
    val_s1_entities = {s: all_s1_dict[s] for s in val_s1_subset}

    mono_train_pairs = []
    for sid, ent in train_s1_entities.items():
        s2_c, s3_c = cand_gen.generate_candidates(ent)
        for c in s2_c: mono_train_pairs.append((sid, c))
        for c in s3_c: mono_train_pairs.append((sid, c))

    mono_val_pairs = []
    for sid, ent in val_s1_entities.items():
        s2_c, s3_c = cand_gen.generate_candidates(ent)
        for c in s2_c: mono_val_pairs.append((sid, c))
        for c in s3_c: mono_val_pairs.append((sid, c))

    mono_train_y = np.array([1 if c in gt_mapping.get(s, []) else 0 for s, c in mono_train_pairs], dtype=np.int32)
    mono_val_y = np.array([1 if c in gt_mapping.get(s, []) else 0 for s, c in mono_val_pairs], dtype=np.int32)

    # Entity store
    entity_store = {}
    for eid, e in preloaded_entities.items():
        entity_store[eid] = NormalizedEntity.from_entity(e, normalizer)

    mono_X_train, _ = extractor.extract_batch_features(mono_train_pairs, entity_store)
    mono_X_val, _ = extractor.extract_batch_features(mono_val_pairs, entity_store)

    mono_model = EntityMatchingModel(C=1.0, class_weight="balanced", random_state=42)
    mono_model.fit(mono_X_train, mono_train_y)

    mono_val_probs = mono_model.predict_proba(mono_X_val)
    mono_exp = mono_model.run_threshold_experiment(
        val_pairs=mono_val_pairs,
        val_probs=mono_val_probs,
        val_y_true=mono_val_y,
        val_gt=gt_mapping,
        val_s1_ids=set(val_s1_subset),
        thresholds=[0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]
    )
    t_mono = time.perf_counter() - t0
    print(f"Monolithic Complete in {t_mono:.2f}s | Best Threshold: {mono_exp['best_threshold']:.2f} | Best F0.5: {mono_exp['best_validation_s1_f05']:.4f}")

    # 2. Streaming Run (Chunk size 500 S1 entities)
    print("\n--- Running Streaming Smoke Test (Chunk size 500) ---")
    t0_stream = time.perf_counter()
    chunk_size = 500
    
    # Train streaming
    train_X_chunks = []
    train_y_chunks = []
    total_pos = 0
    total_neg = 0
    
    for i in range(0, len(train_s1_subset), chunk_size):
        chunk_ids = train_s1_subset[i : i + chunk_size]
        chunk_pairs = []
        for sid in chunk_ids:
            ent = all_s1_dict[sid]
            s2_c, s3_c = cand_gen.generate_candidates(ent)
            for c in s2_c: chunk_pairs.append((sid, c))
            for c in s3_c: chunk_pairs.append((sid, c))
        
        chunk_y = np.array([1 if c in gt_mapping.get(s, []) else 0 for s, c in chunk_pairs], dtype=np.int32)
        X_chunk, _ = extractor.extract_batch_features(chunk_pairs, entity_store)
        
        pos_cnt = np.sum(chunk_y == 1)
        neg_cnt = np.sum(chunk_y == 0)
        total_pos += pos_cnt
        total_neg += neg_cnt
        
        train_X_chunks.append(X_chunk)
        train_y_chunks.append(chunk_y)
        print(f"   [TRAIN CHUNK {i//chunk_size + 1}] {len(chunk_ids)} S1 -> {len(chunk_pairs):,} pairs (Pos: {pos_cnt:,}, Neg: {neg_cnt:,})")

    stream_X_train = np.vstack(train_X_chunks)
    stream_y_train = np.concatenate(train_y_chunks)
    
    stream_model = EntityMatchingModel(C=1.0, class_weight="balanced", random_state=42)
    stream_model.fit(stream_X_train, stream_y_train)

    # Validation streaming
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]
    thresh_stats = {t: {"tp": 0, "fp": 0, "fn": 0, "exact": 0} for t in thresholds}
    
    val_cand_count = 0
    val_ret_gt_all = 0
    val_tot_gt_all = 0
    val_s1_complete = 0
    
    for i in range(0, len(val_s1_subset), chunk_size):
        chunk_ids = val_s1_subset[i : i + chunk_size]
        chunk_pairs = []
        chunk_cand_map = defaultdict(set)
        for sid in chunk_ids:
            ent = all_s1_dict[sid]
            s2_c, s3_c = cand_gen.generate_candidates(ent)
            for c in s2_c:
                chunk_pairs.append((sid, c))
                chunk_cand_map[sid].add(c)
            for c in s3_c:
                chunk_pairs.append((sid, c))
                chunk_cand_map[sid].add(c)
        
        val_cand_count += len(chunk_pairs)
        
        # Blocking recall for chunk
        for sid in chunk_ids:
            gt_m = gt_mapping.get(sid, [])
            val_tot_gt_all += len(gt_m)
            ret_cnt = sum(1 for m in gt_m if m in chunk_cand_map[sid])
            val_ret_gt_all += ret_cnt
            if ret_cnt == len(gt_m):
                val_s1_complete += 1

        chunk_y = np.array([1 if c in gt_mapping.get(s, []) else 0 for s, c in chunk_pairs], dtype=np.int32)
        X_chunk, _ = extractor.extract_batch_features(chunk_pairs, entity_store)
        chunk_probs = stream_model.predict_proba(X_chunk)
        
        s1_preds_map = defaultdict(list)
        for (sid, cid), prob in zip(chunk_pairs, chunk_probs):
            s1_preds_map[sid].append((cid, prob))
        
        for sid in chunk_ids:
            gt_set = set(gt_mapping.get(sid, []))
            preds_list = s1_preds_map.get(sid, [])
            for t in thresholds:
                pred_set = {cid for cid, p in preds_list if p >= t}
                thresh_stats[t]["tp"] += len(pred_set & gt_set)
                thresh_stats[t]["fp"] += len(pred_set - gt_set)
                thresh_stats[t]["fn"] += len(gt_set - pred_set)
                if pred_set == gt_set:
                    thresh_stats[t]["exact"] += 1
                    
        print(f"   [VAL CHUNK {i//chunk_size + 1}] {len(chunk_ids)} S1 -> {len(chunk_pairs):,} pairs processed.")

    # Calculate threshold results
    stream_results = []
    for t in thresholds:
        tp = thresh_stats[t]["tp"]
        fp = thresh_stats[t]["fp"]
        fn = thresh_stats[t]["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        beta_sq = 0.5 ** 2
        f05 = ((1 + beta_sq) * prec * rec) / ((beta_sq * prec) + rec) if ((beta_sq * prec) + rec) > 0 else 0.0
        exact_pct = (thresh_stats[t]["exact"] / len(val_s1_subset)) * 100
        stream_results.append({
            "threshold": t,
            "precision": float(np.round(prec, 4)),
            "recall": float(np.round(rec, 4)),
            "f05": float(np.round(f05, 4)),
            "exact_match_rate_pct": float(np.round(exact_pct, 2))
        })
    
    best_stream_res = max(stream_results, key=lambda x: x["f05"])
    t_stream = time.perf_counter() - t0_stream
    print(f"\nStreaming Complete in {t_stream:.2f}s | Best Threshold: {best_stream_res['threshold']:.2f} | Best F0.5: {best_stream_res['f05']:.4f}")

    print("\n--- COMPARISON RESULTS ---")
    print(f"Candidate Pairs: Monolithic Train={len(mono_train_pairs):,}, Stream Train={len(stream_X_train):,} (Diff={len(mono_train_pairs)-len(stream_X_train)})")
    print(f"Candidate Pairs: Monolithic Val={len(mono_val_pairs):,}, Stream Val={val_cand_count:,} (Diff={len(mono_val_pairs)-val_cand_count})")
    print(f"Candidate Recall: Stream={val_ret_gt_all/val_tot_gt_all*100:.2f}%, S1 Coverage={val_s1_complete/len(val_s1_subset)*100:.2f}%")
    print(f"Monolithic Best F0.5: {mono_exp['best_validation_s1_f05']:.4f} at Threshold {mono_exp['best_threshold']:.2f}")
    print(f"Streaming  Best F0.5: {best_stream_res['f05']:.4f} at Threshold {best_stream_res['threshold']:.2f}")
    
    assert len(mono_train_pairs) == len(stream_X_train)
    assert len(mono_val_pairs) == val_cand_count
    assert abs(mono_exp['best_validation_s1_f05'] - best_stream_res['f05']) < 1e-4
    print("\n[SUCCESS] Exact mathematical equivalence between Monolithic and Streaming implementations confirmed!")

if __name__ == "__main__":
    run_test()
