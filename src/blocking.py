"""
Candidate Generation and Blocking Module for Amazon ML Challenge 2026.

Supports:
- Pass 1 (B1): Exact Normalized Business Name + Country
- Pass 2 (B2): Distinctive Business Name Token + Country (with frequency pruning)
- Pass 3 (B3): Character Prefix (N-gram) + Country
- Pass 4 (B4): Name Token + Address/Locality (Additive pass)

Uses compact array.array('I') posting lists to minimize resident memory.
"""

import array
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set

from src.data_loader import DataLoader, Entity
from src.normalization import (
    normalize_business_name,
    tokenize_business_name,
    extract_address_tokens
)


@dataclass
class BlockingConfig:
    """Configuration parameters for blocking passes."""
    enabled_passes: List[str] = field(
        default_factory=lambda: [
            "exact_name_country",
            "distinctive_token_country",
            "name_prefix_country",
            "name_token_address"
        ]
    )
    chunk_size: int = 100000
    max_candidates_per_source: Optional[int] = None  # None = Uncapped / raw candidate sets
    max_token_doc_frequency: int = 5000  # Pruning threshold for high-frequency generic tokens in B2
    min_token_len: int = 3  # Minimum length for distinctive business name tokens in B2
    prefix_len: int = 4  # Character prefix length for Pass 3 (B3)
    b4_name_min_len: int = 2  # Minimum name token length for Pass 4 (B4)
    b4_addr_min_len: int = 4  # Minimum address token length for Pass 4 (B4)
    b4_max_name_tokens: int = 2  # Up to first N name tokens in B4
    b4_max_addr_tokens: int = 3  # Up to first N address tokens in B4


def build_exact_name_country_key(name: Optional[str], country: Optional[str]) -> str:
    """
    Construct exact normalized business name + country blocking key for Pass 1 (B1).
    
    Reuses normalize_business_name() from src.normalization.
    Country is normalized using .strip().lower() for open-set support.
    Format: '<normalized_name>___<country_lower>'
    
    Returns empty string if normalized name is empty.
    """
    norm_name = normalize_business_name(name)
    if not norm_name:
        return ""
    norm_country = country.strip().lower() if country and isinstance(country, str) else ""
    return f"{norm_name}___{norm_country}"


def build_distinctive_token_country_keys(
    name: Optional[str],
    country: Optional[str],
    min_token_len: int = 3
) -> List[str]:
    """
    Construct distinctive token + country blocking keys for Pass 2 (B2).
    
    Reuses tokenize_business_name() from src.normalization with
    remove_stopwords=True and min_token_len >= 3.
    Format: 'tok_<token>___<country_lower>'
    
    Returns list of unique keys for the entity.
    """
    if not name or not isinstance(name, str):
        return []
    tokens = tokenize_business_name(
        name,
        remove_stopwords=True,
        min_token_len=min_token_len
    )
    if not tokens:
        return []
    norm_country = country.strip().lower() if country and isinstance(country, str) else ""
    unique_tokens = set(tokens)
    return [f"tok_{t}___{norm_country}" for t in unique_tokens]


def build_name_prefix_country_key(
    name: Optional[str],
    country: Optional[str],
    prefix_len: int = 4
) -> str:
    """
    Construct character prefix + country blocking key for Pass 3 (B3).
    
    Reuses normalize_business_name() from src.normalization.
    If the normalized name length is at least prefix_len:
      uses the first prefix_len characters as the prefix.
    Format: 'pref_<prefix>___<country_lower>'
    
    Returns empty string if normalized name is shorter than prefix_len.
    """
    norm_name = normalize_business_name(name)
    if not norm_name or len(norm_name) < prefix_len:
        return ""
    prefix = norm_name[:prefix_len]
    norm_country = country.strip().lower() if country and isinstance(country, str) else ""
    return f"pref_{prefix}___{norm_country}"


def build_name_address_keys(
    name: Optional[str],
    address: Optional[str],
    name_min_len: int = 2,
    addr_min_len: int = 4,
    max_name_tokens: int = 2,
    max_addr_tokens: int = 3
) -> List[str]:
    """
    Construct additive Name Token + Address/Locality Token blocking keys for Pass 4 (B4).
    
    Reuses tokenize_business_name() and extract_address_tokens() from src.normalization.
    Uses up to first `max_name_tokens` name tokens and up to first `max_addr_tokens` address tokens.
    Format: 'na_<name_token>___<addr_token>'
    
    If address is missing or empty, safely returns [] (purely additive, never drops entities).
    """
    if not name or not address or not isinstance(name, str) or not isinstance(address, str):
        return []

    addr_tokens = extract_address_tokens(address, min_token_len=addr_min_len)
    if not addr_tokens:
        return []

    name_tokens = tokenize_business_name(
        name,
        remove_stopwords=True,
        min_token_len=name_min_len
    )
    if not name_tokens:
        return []

    selected_name = name_tokens[:max_name_tokens]
    selected_addr = addr_tokens[:max_addr_tokens]

    keys: List[str] = []
    seen: Set[str] = set()
    for nt in selected_name:
        for at in selected_addr:
            k = f"na_{nt}___{at}"
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


class TargetIndex:
    """
    Inverted index for a single target candidate source (Source 2 or Source 3).
    
    Stores entity IDs in an integer-indexed master list and maps
    blocking keys across passes B1, B2, B3, and B4 to compact array.array('I') posting lists.
    """

    def __init__(self, source_name: str, config: Optional[BlockingConfig] = None):
        self.source_name = source_name
        self.config = config or BlockingConfig()
        self.entity_ids: List[str] = []
        
        # Compact posting lists using unsigned 32-bit integer arrays
        self.b1_index: Dict[str, array.array] = defaultdict(lambda: array.array('I'))
        self.b2_index: Dict[str, array.array] = defaultdict(lambda: array.array('I'))
        self.b3_index: Dict[str, array.array] = defaultdict(lambda: array.array('I'))
        self.b4_index: Dict[str, array.array] = defaultdict(lambda: array.array('I'))
        self.index = self.b1_index  # Backward-compatibility alias for B1-only access
        
        # Track pruned high-frequency tokens in B2
        self.pruned_b2_keys: Dict[str, int] = {}
        
        # Track missing address statistics
        self.missing_address_count: int = 0

    def add_batch(self, entities: List[Entity]) -> None:
        """
        Streamingly ingest a batch of entities from DataLoader.
        
        Assigns sequential integer indices and updates the inverted indexes.
        Full Entity objects are NOT retained in memory.
        """
        base_idx = len(self.entity_ids)
        enable_b1 = "exact_name_country" in self.config.enabled_passes
        enable_b2 = "distinctive_token_country" in self.config.enabled_passes
        enable_b3 = "name_prefix_country" in self.config.enabled_passes
        enable_b4 = "name_token_address" in self.config.enabled_passes

        b2_min_len = self.config.min_token_len
        prefix_len = self.config.prefix_len
        b4_name_min = self.config.b4_name_min_len
        b4_addr_min = self.config.b4_addr_min_len
        b4_max_name = self.config.b4_max_name_tokens
        b4_max_addr = self.config.b4_max_addr_tokens

        for i, entity in enumerate(entities):
            idx = base_idx + i
            self.entity_ids.append(entity.entity_id)

            if not entity.business_address or not entity.business_address.strip():
                self.missing_address_count += 1

            # Pass 1 (B1): Exact Name + Country
            if enable_b1:
                key_b1 = build_exact_name_country_key(entity.business_name, entity.country)
                if key_b1:
                    self.b1_index[key_b1].append(idx)

            # Pass 2 (B2): Distinctive Tokens + Country
            if enable_b2:
                keys_b2 = build_distinctive_token_country_keys(
                    entity.business_name,
                    entity.country,
                    min_token_len=b2_min_len
                )
                for k in keys_b2:
                    self.b2_index[k].append(idx)

            # Pass 3 (B3): Character Prefix + Country
            if enable_b3:
                key_b3 = build_name_prefix_country_key(
                    entity.business_name,
                    entity.country,
                    prefix_len=prefix_len
                )
                if key_b3:
                    self.b3_index[key_b3].append(idx)

            # Pass 4 (B4): Name Token + Address Token (Additive)
            if enable_b4:
                keys_b4 = build_name_address_keys(
                    entity.business_name,
                    entity.business_address,
                    name_min_len=b4_name_min,
                    addr_min_len=b4_addr_min,
                    max_name_tokens=b4_max_name,
                    max_addr_tokens=b4_max_addr
                )
                for k in keys_b4:
                    self.b4_index[k].append(idx)

    def prune_b2_high_frequency_keys(self) -> Dict[str, int]:
        """
        Prune B2 posting lists whose document frequency exceeds max_token_doc_frequency.
        
        This prevents combinatorial explosion from generic words (e.g. 'health', 'group').
        Returns dictionary of pruned keys and their counts, sorted descending.
        """
        threshold = self.config.max_token_doc_frequency
        pruned: Dict[str, int] = {}
        keys_to_remove: List[str] = []

        for key, postings in self.b2_index.items():
            if len(postings) > threshold:
                pruned[key] = len(postings)
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.b2_index[key]

        self.pruned_b2_keys = dict(sorted(pruned.items(), key=lambda item: item[1], reverse=True))
        return self.pruned_b2_keys

    def query_b1(self, s1_entity: Entity) -> List[str]:
        """Query B1 pass only (exact normalized name + country)."""
        key = build_exact_name_country_key(s1_entity.business_name, s1_entity.country)
        if not key or key not in self.b1_index:
            return []
        matched_indices = self.b1_index[key]
        return [self.entity_ids[idx] for idx in matched_indices]

    def query_b2(self, s1_entity: Entity) -> List[str]:
        """Query B2 pass only (distinctive token + country, deduplicated)."""
        keys = build_distinctive_token_country_keys(
            s1_entity.business_name,
            s1_entity.country,
            min_token_len=self.config.min_token_len
        )
        if not keys:
            return []

        seen_indices: Set[int] = set()
        candidate_ids: List[str] = []

        for k in keys:
            if k in self.b2_index:
                for idx in self.b2_index[k]:
                    if idx not in seen_indices:
                        seen_indices.add(idx)
                        candidate_ids.append(self.entity_ids[idx])
        return candidate_ids

    def query_b3(self, s1_entity: Entity) -> List[str]:
        """Query B3 pass only (character prefix + country)."""
        key = build_name_prefix_country_key(
            s1_entity.business_name,
            s1_entity.country,
            prefix_len=self.config.prefix_len
        )
        if not key or key not in self.b3_index:
            return []
        matched_indices = self.b3_index[key]
        return [self.entity_ids[idx] for idx in matched_indices]

    def query_b4(self, s1_entity: Entity) -> List[str]:
        """
        Query B4 pass only (name token + address token, deduplicated).
        
        Returns empty list if address is missing.
        """
        keys = build_name_address_keys(
            s1_entity.business_name,
            s1_entity.business_address,
            name_min_len=self.config.b4_name_min_len,
            addr_min_len=self.config.b4_addr_min_len,
            max_name_tokens=self.config.b4_max_name_tokens,
            max_addr_tokens=self.config.b4_max_addr_tokens
        )
        if not keys:
            return []

        seen_indices: Set[int] = set()
        candidate_ids: List[str] = []

        for k in keys:
            if k in self.b4_index:
                for idx in self.b4_index[k]:
                    if idx not in seen_indices:
                        seen_indices.add(idx)
                        candidate_ids.append(self.entity_ids[idx])
        return candidate_ids

    def query_passes(self, s1_entity: Entity) -> Dict[str, List[str]]:
        """
        Query enabled passes and return candidates partitioned by pass.
        
        Returns:
            Dict with keys 'b1', 'b2', 'b3', 'b4',
            'cumulative_b1_b2', 'cumulative_b1_b2_b3', and 'cumulative' (B1+B2+B3+B4).
        """
        b1_cands = self.query_b1(s1_entity) if "exact_name_country" in self.config.enabled_passes else []
        b2_cands = self.query_b2(s1_entity) if "distinctive_token_country" in self.config.enabled_passes else []
        b3_cands = self.query_b3(s1_entity) if "name_prefix_country" in self.config.enabled_passes else []
        b4_cands = self.query_b4(s1_entity) if "name_token_address" in self.config.enabled_passes else []

        # Cumulative B1 + B2
        seen_12: Set[str] = set()
        cum_12: List[str] = []
        for cid in b1_cands:
            if cid not in seen_12: seen_12.add(cid); cum_12.append(cid)
        for cid in b2_cands:
            if cid not in seen_12: seen_12.add(cid); cum_12.append(cid)

        # Cumulative B1 + B2 + B3
        seen_123 = set(seen_12)
        cum_123 = list(cum_12)
        for cid in b3_cands:
            if cid not in seen_123: seen_123.add(cid); cum_123.append(cid)

        # Final Cumulative B1 + B2 + B3 + B4 (Additive)
        seen_all = set(seen_123)
        cum_all = list(cum_123)
        for cid in b4_cands:
            if cid not in seen_all: seen_all.add(cid); cum_all.append(cid)

        return {
            "b1": b1_cands,
            "b2": b2_cands,
            "b3": b3_cands,
            "b4": b4_cands,
            "cumulative_b1_b2": cum_12,
            "cumulative_b1_b2_b3": cum_123,
            "cumulative": cum_all
        }

    def query(self, s1_entity: Entity) -> List[str]:
        """
        Query all enabled passes and return uncapped deduplicated candidate IDs.
        """
        return self.query_passes(s1_entity)["cumulative"]

    def get_top_b3_keys(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Return top N most frequent B3 prefix keys with their posting counts."""
        sorted_keys = sorted(
            ((k, len(v)) for k, v in self.b3_index.items()),
            key=lambda item: item[1],
            reverse=True
        )
        return sorted_keys[:top_n]

    def get_top_b4_keys(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Return top N most frequent B4 name+address keys with their posting counts."""
        sorted_keys = sorted(
            ((k, len(v)) for k, v in self.b4_index.items()),
            key=lambda item: item[1],
            reverse=True
        )
        return sorted_keys[:top_n]

    def get_stats(self) -> Dict[str, Any]:
        """Return index summary statistics."""
        b4_postings = sum(len(v) for v in self.b4_index.values())
        total_ents = len(self.entity_ids)
        return {
            "source_name": self.source_name,
            "total_indexed": total_ents,
            "b1_unique_keys": len(self.b1_index),
            "b2_unique_keys": len(self.b2_index),
            "b2_pruned_keys": len(self.pruned_b2_keys),
            "b3_unique_keys": len(self.b3_index),
            "b4_unique_keys": len(self.b4_index),
            "b4_total_postings": b4_postings,
            "missing_addresses": self.missing_address_count,
            "missing_address_pct": round((self.missing_address_count / total_ents * 100), 2) if total_ents else 0.0,
        }


class CandidateGenerator:
    """
    Coordinates candidate generation across independent TargetIndex instances
    for Source 2 and Source 3.
    """

    def __init__(self, data_loader: DataLoader, config: Optional[BlockingConfig] = None):
        self.loader = data_loader
        self.config = config or BlockingConfig()
        self.s2_index = TargetIndex("source2", self.config)
        self.s3_index = TargetIndex("source3", self.config)

    def build_indexes(self, split: str = "train") -> Dict[str, float]:
        """
        Build independent S2 and S3 inverted indexes with B1, B2, B3, and B4 passes.
        Applies B2 high-frequency token pruning.
        Returns dict containing build elapsed times in seconds.
        """
        build_times: Dict[str, float] = {}

        # 1. Build Source 2 index
        s2_path = self.loader.get_file_path(split, "source2")
        t0 = time.perf_counter()
        for batch in self.loader.stream_entities(s2_path, chunk_size=self.config.chunk_size):
            self.s2_index.add_batch(batch)
        if "distinctive_token_country" in self.config.enabled_passes:
            self.s2_index.prune_b2_high_frequency_keys()
        build_times["source2"] = time.perf_counter() - t0

        # 2. Build Source 3 index
        s3_path = self.loader.get_file_path(split, "source3")
        t0 = time.perf_counter()
        for batch in self.loader.stream_entities(s3_path, chunk_size=self.config.chunk_size):
            self.s3_index.add_batch(batch)
        if "distinctive_token_country" in self.config.enabled_passes:
            self.s3_index.prune_b2_high_frequency_keys()
        build_times["source3"] = time.perf_counter() - t0

        return build_times

    def generate_candidates_passes(self, s1_entity: Entity) -> Dict[str, Tuple[List[str], List[str]]]:
        """
        Generate candidate pairs partitioned by pass name.
        
        Returns:
            Dict containing tuples of (s2_candidates, s3_candidates) for
            'b1', 'b2', 'b3', 'b4', 'cumulative_b1_b2', 'cumulative_b1_b2_b3', and 'cumulative'.
        """
        s2_passes = self.s2_index.query_passes(s1_entity)
        s3_passes = self.s3_index.query_passes(s1_entity)
        return {
            "b1": (s2_passes["b1"], s3_passes["b1"]),
            "b2": (s2_passes["b2"], s3_passes["b2"]),
            "b3": (s2_passes["b3"], s3_passes["b3"]),
            "b4": (s2_passes["b4"], s3_passes["b4"]),
            "cumulative_b1_b2": (s2_passes["cumulative_b1_b2"], s3_passes["cumulative_b1_b2"]),
            "cumulative_b1_b2_b3": (s2_passes["cumulative_b1_b2_b3"], s3_passes["cumulative_b1_b2_b3"]),
            "cumulative": (s2_passes["cumulative"], s3_passes["cumulative"])
        }

    def generate_candidates(self, s1_entity: Entity) -> Tuple[List[str], List[str]]:
        """
        Generate uncapped cumulative candidates for an S1 entity across S2 and S3.
        """
        passes = self.generate_candidates_passes(s1_entity)
        return passes["cumulative"]
