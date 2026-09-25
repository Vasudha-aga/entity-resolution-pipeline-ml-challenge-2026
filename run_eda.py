import csv
import os
import sys
from collections import Counter, defaultdict
import statistics
import json

def analyze_entity_file(filepath):
    print(f"\n==========================================")
    print(f"Analyzing {filepath} ...")
    row_count = 0
    missing_counts = defaultdict(int)
    entity_ids = set()
    dup_ids = 0
    name_lengths = []
    addr_lengths = []
    name_counter = Counter()
    countries = Counter()
    
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        cols = header
        for row in reader:
            row_count += 1
            if len(row) < len(header):
                row.extend([''] * (len(header) - len(row)))
            
            eid, name, addr, country = row[0], row[1], row[2], row[3]
            
            if eid in entity_ids:
                dup_ids += 1
            else:
                entity_ids.add(eid)
                
            if not eid.strip(): missing_counts['entity_id'] += 1
            if not name.strip(): missing_counts['business_name'] += 1
            if not addr.strip(): missing_counts['business_address'] += 1
            if not country.strip(): missing_counts['country'] += 1
            
            name_len = len(name.strip())
            addr_len = len(addr.strip())
            name_lengths.append(name_len)
            addr_lengths.append(addr_len)
            
            name_counter[name.strip().lower()] += 1
            countries[country.strip()] += 1
            
            if row_count % 500000 == 0:
                print(f"  Processed {row_count:,} rows...")
                
    name_stats = {
        'min': min(name_lengths) if name_lengths else 0,
        'max': max(name_lengths) if name_lengths else 0,
        'mean': round(statistics.mean(name_lengths), 2) if name_lengths else 0,
        'median': round(statistics.median(name_lengths), 2) if name_lengths else 0,
    }
    
    addr_stats = {
        'min': min(addr_lengths) if addr_lengths else 0,
        'max': max(addr_lengths) if addr_lengths else 0,
        'mean': round(statistics.mean(addr_lengths), 2) if addr_lengths else 0,
        'median': round(statistics.median(addr_lengths), 2) if addr_lengths else 0,
    }
    
    # Exact name duplicates count
    dup_names_count = sum(cnt - 1 for cnt in name_counter.values() if cnt > 1)
    unique_names_count = len(name_counter)
    
    res = {
        'file': filepath,
        'rows': row_count,
        'cols': cols,
        'missing': dict(missing_counts),
        'unique_eids': len(entity_ids),
        'duplicate_eids': dup_ids,
        'unique_names': unique_names_count,
        'duplicate_names_count': dup_names_count,
        'top_names': name_counter.most_common(5),
        'name_stats': name_stats,
        'addr_stats': addr_stats,
        'country_dist': countries.most_common(10),
        'total_countries': len(countries)
    }
    print(f"Done: {row_count:,} rows.")
    return res

def analyze_ground_truth(filepath):
    print(f"\n==========================================")
    print(f"Analyzing Ground Truth {filepath} ...")
    row_count = 0
    s1_entities = set()
    dup_s1 = 0
    match_counts = []
    s2_match_total = 0
    s3_match_total = 0
    match_dist = Counter()
    missing_counts = defaultdict(int)
    
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for row in reader:
            row_count += 1
            if len(row) < 2:
                s1_id = row[0] if len(row) > 0 else ''
                matched_str = ''
            else:
                s1_id, matched_str = row[0], row[1]
                
            if s1_id in s1_entities:
                dup_s1 += 1
            else:
                s1_entities.add(s1_id)
                
            if not s1_id.strip(): missing_counts['source1_entity_id'] += 1
            
            matched_list = [x.strip() for x in matched_str.split(',') if x.strip()]
            num_matches = len(matched_list)
            match_counts.append(num_matches)
            match_dist[num_matches] += 1
            
            for m in matched_list:
                if m.startswith('S2-'):
                    s2_match_total += 1
                elif m.startswith('S3-'):
                    s3_match_total += 1
                    
            if row_count % 500000 == 0:
                print(f"  Processed {row_count:,} rows...")
                
    zero_matches = match_dist[0]
    single_match = match_dist[1]
    multi_matches = sum(count for k, count in match_dist.items() if k > 1)
    
    gt_res = {
        'file': filepath,
        'rows': row_count,
        'cols': header,
        'missing': dict(missing_counts),
        'unique_s1_entities': len(s1_entities),
        'duplicate_s1_eids': dup_s1,
        'total_matches': sum(match_counts),
        's2_matches': s2_match_total,
        's3_matches': s3_match_total,
        'zero_matches_count': zero_matches,
        'single_match_count': single_match,
        'multi_matches_count': multi_matches,
        'pct_multiple_matches': round((multi_matches / row_count) * 100, 2) if row_count else 0,
        'pct_single_match': round((single_match / row_count) * 100, 2) if row_count else 0,
        'pct_zero_matches': round((zero_matches / row_count) * 100, 2) if row_count else 0,
        'min_matches': min(match_counts) if match_counts else 0,
        'max_matches': max(match_counts) if match_counts else 0,
        'mean_matches': round(statistics.mean(match_counts), 2) if match_counts else 0,
        'median_matches': round(statistics.median(match_counts), 2) if match_counts else 0,
        'match_distribution': sorted(match_dist.items())
    }
    print(f"Done GT: {row_count:,} rows.")
    return gt_res

if __name__ == '__main__':
    results = {}
    files_to_analyze = [
        ('train_s1', 'data/train/train_source1.tsv'),
        ('train_s2', 'data/train/train_source2.tsv'),
        ('train_s3', 'data/train/train_source3.tsv'),
        ('test_s1', 'data/test/test_source1.tsv'),
        ('test_s2', 'data/test/test_source2.tsv'),
        ('test_s3', 'data/test/test_source3.tsv'),
    ]
    
    for key, path in files_to_analyze:
        if os.path.exists(path):
            results[key] = analyze_entity_file(path)
            
    gt_path = 'data/train/train_ground_truth.tsv'
    if os.path.exists(gt_path):
        results['ground_truth'] = analyze_ground_truth(gt_path)
        
    with open('eda_summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    print("\nSaved full summary to eda_summary.json!")
