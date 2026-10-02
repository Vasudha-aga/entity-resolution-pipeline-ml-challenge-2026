"""
Candidate Generation and Blocking Module for Amazon ML Challenge 2026.

Supports:
- Pass 1 (B1): Exact Normalized Business Name + Country
- Pass 2 (B2): Distinctive Business Name Token + Country (with frequency pruning)
- Pass 3 (B3): Character Prefix (N-gram) + Country
- Pass 4 (B4): Name Token + Address/Locality (Additive pass)
- Pass 5A (B5-A): Building/House Number + Distinctive Locality/Address Token + Country (Address-only)
- Pass 5B (B5-B): Two Distinctive Non-Generic Address Tokens + Country (Address-only)

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
    extract_address_tokens,
    DEFAULT_NAME_STOPWORDS
)

# Standard generic address stopwords/markers to exclude from distinctive token pairs
DEFAULT_GENERIC_ADDR_TOKENS: Set[str] = {
    "road", "rd", "street", "st", "drive", "dr", "avenue", "ave", "lane", "ln",
    "unit", "floor", "fl", "suite", "ste", "apt", "apartment", "null", "box",
    "po", "no", "near", "opp", "opposite", "behind", "phase", "plot", "sector",
    "flat", "block", "house", "hno", "kh", "door", "st", "nd", "rd", "th",
    "east", "west", "north", "south", "city", "nagar", "colony", "bhavan",
    "complex", "building", "tower", "plaza", "market", "bazaar", "center", "centre",
    "delhi", "mumbai", "india", "state", "dist", "district", "post", "extn", "extension"
}


@dataclass
class BlockingConfig:
    """Configuration parameters for blocking passes."""
    enabled_passes: List[str] = field(
        default_factory=lambda: [
            "exact_name_country",            # B1
            "distinctive_token_country",     # B2
            "name_prefix_country",           # B3
            "name_token_address",            # B4
            "address_number_locality",       # B5-A
            "address_distinctive_pair"        # B5-B
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
    b5a_max_nums: int = 2  # Up to first N numeric/building tokens for Pass 5A (B5-A)
    b5a_max_toks: int = 3  # Up to first N distinctive locality tokens for Pass 5A (B5-A)
    b5b_max_toks: int = 4  # Up to first N distinctive tokens for pairwise combinations in Pass 5B (B5-B)
    max_b5_doc_frequency: int = 500  # Pruning threshold for high-frequency generic keys in B5
    generic_addr_tokens: Set[str] = field(default_factory=lambda: set(DEFAULT_GENERIC_ADDR_TOKENS))


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


def build_b5a_keys(
    address: Optional[str],
    country: Optional[str],
    max_nums: int = 2,
    max_toks: int = 3,
    generic_tokens: Optional[Set[str]] = None
) -> List[str]:
    """
    Construct Pass 5A (B5-A) blocking keys:
    Building/House Number + Distinctive Locality/Address Token + Country.
    
    Example: 'b5_num_68_rtnagar___india'
    
    Requirements:
    - Never uses business name.
    - Preserves numeric and alphanumeric building tokens (e.g. '68', '273', '196a', 'cc54').
    - Excludes generic address stopwords from distinctive tokens.
    - Preserves country as part of the key.
    """
    if not address or not isinstance(address, str) or not address.strip():
        return []

    generic_set = generic_tokens if generic_tokens is not None else DEFAULT_GENERIC_ADDR_TOKENS
    norm_country = country.strip().lower() if country and isinstance(country, str) else ""
    tokens = extract_address_tokens(address, min_token_len=2)
    if not tokens:
        return []

    num_tokens: List[str] = []
    distinctive_tokens: List[str] = []

    for t in tokens:
        if t.isdigit() or (len(t) >= 2 and any(c.isdigit() for c in t) and any(c.isalpha() for c in t)):
            if t not in num_tokens:
                num_tokens.append(t)
        elif len(t) >= 3 and t not in generic_set:
            if t not in distinctive_tokens:
                distinctive_tokens.append(t)

    selected_nums = num_tokens[:max_nums]
    selected_toks = distinctive_tokens[:max_toks]

    keys: List[str] = []
    for num in selected_nums:
        for tok in selected_toks:
            keys.append(f"b5_num_{num}_{tok}___{norm_country}")
    return keys


def build_b5b_keys(
    address: Optional[str],
    country: Optional[str],
    max_toks: int = 4,
    generic_tokens: Optional[Set[str]] = None
) -> List[str]:
    """
    Construct Pass 5B (B5-B) blocking keys:
    Two Distinctive Non-Generic Address Tokens + Country.
    
    Example: 'b5_pair_beturkar_pada___india'
    
    Requirements:
    - Never uses business name.
    - Extracts non-generic, non-purely numeric address tokens of length >= 3.
    - Excludes generic address stopwords.
    - Generates sorted pairs across top `max_toks` distinctive tokens.
    - Preserves country as part of the key.
    """
    if not address or not isinstance(address, str) or not address.strip():
        return []

    generic_set = generic_tokens if generic_tokens is not None else DEFAULT_GENERIC_ADDR_TOKENS
    norm_country = country.strip().lower() if country and isinstance(country, str) else ""
    tokens = extract_address_tokens(address, min_token_len=3)
    if not tokens:
        return []

    distinctive_tokens: List[str] = []
    for t in tokens:
        if t not in generic_set and not t.isdigit():
            if t not in distinctive_tokens:
                distinctive_tokens.append(t)

    selected = distinctive_tokens[:max_toks]
    if len(selected) < 2:
        return []

    keys: List[str] = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            t1, t2 = sorted([selected[i], selected[j]])
            keys.append(f"b5_pair_{t1}_{t2}___{norm_country}")
    return keys


class TargetIndex:
    """
    Inverted index for a single target candidate source (Source 2 or Source 3).
    
    Stores entity IDs in an integer-indexed master list and maps
    blocking keys across passes B1, B2, B3, B4, B5-A, and B5-B to compact array.array('I') posting lists.
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
        self.b5a_index: Dict[str, array.array] = defaultdict(lambda: array.array('I'))
        self.b5b_index: Dict[str, array.array] = defaultdict(lambda: array.array('I'))
        self.index = self.b1_index  # Backward-compatibility alias for B1-only access
        
        # Track pruned high-frequency keys
        self.pruned_b2_keys: Dict[str, int] = {}
        self.pruned_b5a_keys: Dict[str, int] = {}
        self.pruned_b5b_keys: Dict[str, int] = {}
        
        # Lightweight raw entity store for fast chunk-scoped feature extraction
        self.entity_store_raw: Dict[str, Tuple[str, str, str]] = {}
        
        # Track missing address statistics
        self.missing_address_count: int = 0

    def add_batch(self, entities: List[Entity]) -> None:
        """
        Streamingly ingest a batch of entities from DataLoader.
        
        Assigns sequential integer indices and updates the inverted indexes.
        Stores compact raw (name, address, country) tuples for O(1) lookup.
        """
        base_idx = len(self.entity_ids)
        enabled = set(self.config.enabled_passes)
        
        enable_b1 = "exact_name_country" in enabled or "b1" in enabled
        enable_b2 = "distinctive_token_country" in enabled or "b2" in enabled
        enable_b3 = "name_prefix_country" in enabled or "b3" in enabled
        enable_b4 = "name_token_address" in enabled or "b4" in enabled
        enable_b5a = (
            "address_number_locality" in enabled
            or "b5a" in enabled
            or "b5_a" in enabled
            or "b5a_num_locality_country" in enabled
        )
        enable_b5b = (
            "address_distinctive_pair" in enabled
            or "b5b" in enabled
            or "b5_b" in enabled
            or "b5b_distinctive_pair_country" in enabled
        )

        b2_min_len = self.config.min_token_len
        prefix_len = self.config.prefix_len
        b4_name_min = self.config.b4_name_min_len
        b4_addr_min = self.config.b4_addr_min_len
        b4_max_name = self.config.b4_max_name_tokens
        b4_max_addr = self.config.b4_max_addr_tokens
        b5a_max_nums = self.config.b5a_max_nums
        b5a_max_toks = self.config.b5a_max_toks
        b5b_max_toks = self.config.b5b_max_toks
        generic_tokens = self.config.generic_addr_tokens
        need_name = enable_b1 or enable_b2 or enable_b3 or enable_b4
        need_addr = enable_b4 or enable_b5a or enable_b5b

        for i, entity in enumerate(entities):
            idx = base_idx + i
            self.entity_ids.append(entity.entity_id)
            self.entity_store_raw[entity.entity_id] = (
                entity.business_name,
                entity.business_address,
                entity.country
            )

            has_addr = bool(entity.business_address and entity.business_address.strip())
            if not has_addr:
                self.missing_address_count += 1

            norm_country = entity.country.strip().lower() if entity.country and isinstance(entity.country, str) else ""

            # 1. Single-pass Name Normalization & Tokenization
            norm_name = ""
            name_tokens_clean: List[str] = []
            if need_name and entity.business_name:
                norm_name = normalize_business_name(entity.business_name)
                if norm_name:
                    raw_tokens = norm_name.split()
                    name_tokens_clean = [t for t in raw_tokens if t not in DEFAULT_NAME_STOPWORDS]

            # 2. Single-pass Address Tokenization (min_token_len=2)
            addr_tokens_all: List[str] = []
            if need_addr and has_addr:
                addr_tokens_all = extract_address_tokens(entity.business_address, min_token_len=2)

            # Pass 1 (B1): Exact Name + Country
            if enable_b1 and norm_name:
                self.b1_index[f"{norm_name}___{norm_country}"].append(idx)

            # Pass 2 (B2): Distinctive Tokens + Country
            if enable_b2 and name_tokens_clean:
                tokens_b2 = set(t for t in name_tokens_clean if len(t) >= b2_min_len)
                for t in tokens_b2:
                    self.b2_index[f"tok_{t}___{norm_country}"].append(idx)

            # Pass 3 (B3): Character Prefix + Country
            if enable_b3 and norm_name and len(norm_name) >= prefix_len:
                self.b3_index[f"pref_{norm_name[:prefix_len]}___{norm_country}"].append(idx)

            # Pass 4 (B4): Name Token + Address Token (Additive)
            if enable_b4 and name_tokens_clean and addr_tokens_all:
                selected_name = [t for t in name_tokens_clean if len(t) >= b4_name_min][:b4_max_name]
                selected_addr = [t for t in addr_tokens_all if len(t) >= b4_addr_min][:b4_max_addr]
                if selected_name and selected_addr:
                    seen_b4: Set[str] = set()
                    for nt in selected_name:
                        for at in selected_addr:
                            k = f"na_{nt}___{at}"
                            if k not in seen_b4:
                                seen_b4.add(k)
                                self.b4_index[k].append(idx)

            # Pass 5A (B5-A): Building/House Number + Distinctive Locality Token + Country
            if enable_b5a and addr_tokens_all:
                num_tokens: List[str] = []
                distinctive_tokens: List[str] = []
                for t in addr_tokens_all:
                    if t.isdigit() or (len(t) >= 2 and any(c.isdigit() for c in t) and any(c.isalpha() for c in t)):
                        if t not in num_tokens:
                            num_tokens.append(t)
                    elif len(t) >= 3 and t not in generic_tokens:
                        if t not in distinctive_tokens:
                            distinctive_tokens.append(t)

                selected_nums = num_tokens[:b5a_max_nums]
                selected_toks = distinctive_tokens[:b5a_max_toks]
                for num in selected_nums:
                    for tok in selected_toks:
                        self.b5a_index[f"b5_num_{num}_{tok}___{norm_country}"].append(idx)

            # Pass 5B (B5-B): Two Distinctive Non-Generic Address Tokens + Country
            if enable_b5b and addr_tokens_all:
                distinctive_b5b: List[str] = []
                for t in addr_tokens_all:
                    if len(t) >= 3 and t not in generic_tokens and not t.isdigit():
                        if t not in distinctive_b5b:
                            distinctive_b5b.append(t)

                selected_b5b = distinctive_b5b[:b5b_max_toks]
                if len(selected_b5b) >= 2:
                    for i1 in range(len(selected_b5b)):
                        for j1 in range(i1 + 1, len(selected_b5b)):
                            t1, t2 = sorted([selected_b5b[i1], selected_b5b[j1]])
                            self.b5b_index[f"b5_pair_{t1}_{t2}___{norm_country}"].append(idx)

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

    def prune_b5_high_frequency_keys(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Prune B5-A and B5-B posting lists whose document frequency exceeds max_b5_doc_frequency.
        
        Returns tuple of (pruned_b5a_keys, pruned_b5b_keys).
        """
        threshold = self.config.max_b5_doc_frequency

        # Prune B5-A
        pruned_a: Dict[str, int] = {}
        keys_to_del_a = [k for k, v in self.b5a_index.items() if len(v) > threshold]
        for k in keys_to_del_a:
            pruned_a[k] = len(self.b5a_index[k])
            del self.b5a_index[k]
        self.pruned_b5a_keys = dict(sorted(pruned_a.items(), key=lambda item: item[1], reverse=True))

        # Prune B5-B
        pruned_b: Dict[str, int] = {}
        keys_to_del_b = [k for k, v in self.b5b_index.items() if len(v) > threshold]
        for k in keys_to_del_b:
            pruned_b[k] = len(self.b5b_index[k])
            del self.b5b_index[k]
        self.pruned_b5b_keys = dict(sorted(pruned_b.items(), key=lambda item: item[1], reverse=True))

        return self.pruned_b5a_keys, self.pruned_b5b_keys

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

    def query_b5a(self, s1_entity: Entity) -> List[str]:
        """
        Query B5-A pass only (building number + distinctive locality token + country, deduplicated).
        """
        keys = build_b5a_keys(
            s1_entity.business_address,
            s1_entity.country,
            max_nums=self.config.b5a_max_nums,
            max_toks=self.config.b5a_max_toks,
            generic_tokens=self.config.generic_addr_tokens
        )
        if not keys:
            return []

        seen_indices: Set[int] = set()
        candidate_ids: List[str] = []

        for k in keys:
            if k in self.b5a_index:
                for idx in self.b5a_index[k]:
                    if idx not in seen_indices:
                        seen_indices.add(idx)
                        candidate_ids.append(self.entity_ids[idx])
        return candidate_ids

    def query_b5b(self, s1_entity: Entity) -> List[str]:
        """
        Query B5-B pass only (two distinctive non-generic address tokens + country, deduplicated).
        """
        keys = build_b5b_keys(
            s1_entity.business_address,
            s1_entity.country,
            max_toks=self.config.b5b_max_toks,
            generic_tokens=self.config.generic_addr_tokens
        )
        if not keys:
            return []

        seen_indices: Set[int] = set()
        candidate_ids: List[str] = []

        for k in keys:
            if k in self.b5b_index:
                for idx in self.b5b_index[k]:
                    if idx not in seen_indices:
                        seen_indices.add(idx)
                        candidate_ids.append(self.entity_ids[idx])
        return candidate_ids

    def query_passes(self, s1_entity: Entity) -> Dict[str, List[str]]:
        """
        Query enabled passes and return candidates partitioned by pass.
        
        Returns:
            Dict with keys:
            'b1', 'b2', 'b3', 'b4', 'b5a', 'b5b',
            'cumulative_b1_b2', 'cumulative_b1_b2_b3', 'cumulative_b1_b4',
            'cumulative_b1_b5a', and 'cumulative' (all enabled passes deduplicated).
        """
        enabled = set(self.config.enabled_passes)

        b1_cands = self.query_b1(s1_entity) if ("exact_name_country" in enabled or "b1" in enabled) else []
        b2_cands = self.query_b2(s1_entity) if ("distinctive_token_country" in enabled or "b2" in enabled) else []
        b3_cands = self.query_b3(s1_entity) if ("name_prefix_country" in enabled or "b3" in enabled) else []
        b4_cands = self.query_b4(s1_entity) if ("name_token_address" in enabled or "b4" in enabled) else []
        b5a_cands = self.query_b5a(s1_entity) if (
            "address_number_locality" in enabled
            or "b5a" in enabled
            or "b5_a" in enabled
            or "b5a_num_locality_country" in enabled
        ) else []
        b5b_cands = self.query_b5b(s1_entity) if (
            "address_distinctive_pair" in enabled
            or "b5b" in enabled
            or "b5_b" in enabled
            or "b5b_distinctive_pair_country" in enabled
        ) else []

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

        # Cumulative B1 + B2 + B3 + B4 (Additive)
        seen_b14 = set(seen_123)
        cum_b14 = list(cum_123)
        for cid in b4_cands:
            if cid not in seen_b14: seen_b14.add(cid); cum_b14.append(cid)

        # Cumulative B1..B4 + B5-A
        seen_b15a = set(seen_b14)
        cum_b15a = list(cum_b14)
        for cid in b5a_cands:
            if cid not in seen_b15a: seen_b15a.add(cid); cum_b15a.append(cid)

        # Final Cumulative B1..B5 (all enabled passes, fully deduplicated)
        seen_all = set(seen_b15a)
        cum_all = list(cum_b15a)
        for cid in b5b_cands:
            if cid not in seen_all: seen_all.add(cid); cum_all.append(cid)

        return {
            "b1": b1_cands,
            "b2": b2_cands,
            "b3": b3_cands,
            "b4": b4_cands,
            "b5a": b5a_cands,
            "b5b": b5b_cands,
            "cumulative_b1_b2": cum_12,
            "cumulative_b1_b2_b3": cum_123,
            "cumulative_b1_b4": cum_b14,
            "cumulative_b1_b2_b3_b4": cum_b14,  # Alias
            "cumulative_b1_b5a": cum_b15a,
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

    def get_top_b5a_keys(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Return top N most frequent B5-A keys with their posting counts."""
        sorted_keys = sorted(
            ((k, len(v)) for k, v in self.b5a_index.items()),
            key=lambda item: item[1],
            reverse=True
        )
        return sorted_keys[:top_n]

    def get_top_b5b_keys(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Return top N most frequent B5-B keys with their posting counts."""
        sorted_keys = sorted(
            ((k, len(v)) for k, v in self.b5b_index.items()),
            key=lambda item: item[1],
            reverse=True
        )
        return sorted_keys[:top_n]

    def get_stats(self) -> Dict[str, Any]:
        """Return index summary statistics."""
        b4_postings = sum(len(v) for v in self.b4_index.values())
        b5a_postings = sum(len(v) for v in self.b5a_index.values())
        b5b_postings = sum(len(v) for v in self.b5b_index.values())
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
            "b5a_unique_keys": len(self.b5a_index),
            "b5a_pruned_keys": len(self.pruned_b5a_keys),
            "b5a_total_postings": b5a_postings,
            "b5b_unique_keys": len(self.b5b_index),
            "b5b_pruned_keys": len(self.pruned_b5b_keys),
            "b5b_total_postings": b5b_postings,
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
        Build independent S2 and S3 inverted indexes with B1, B2, B3, B4, B5-A, and B5-B passes.
        Applies B2 and B5 high-frequency token pruning.
        Returns dict containing build elapsed times in seconds.
        """
        build_times: Dict[str, float] = {}
        enabled = set(self.config.enabled_passes)

        # 1. Build Source 2 index
        s2_path = self.loader.get_file_path(split, "source2")
        t0 = time.perf_counter()
        for batch in self.loader.stream_entities(s2_path, chunk_size=self.config.chunk_size):
            self.s2_index.add_batch(batch)
        if "distinctive_token_country" in enabled or "b2" in enabled:
            self.s2_index.prune_b2_high_frequency_keys()
        if (
            "address_number_locality" in enabled
            or "address_distinctive_pair" in enabled
            or "b5a" in enabled
            or "b5b" in enabled
        ):
            self.s2_index.prune_b5_high_frequency_keys()
        build_times["source2"] = time.perf_counter() - t0

        # 2. Build Source 3 index
        s3_path = self.loader.get_file_path(split, "source3")
        t0 = time.perf_counter()
        for batch in self.loader.stream_entities(s3_path, chunk_size=self.config.chunk_size):
            self.s3_index.add_batch(batch)
        if "distinctive_token_country" in enabled or "b2" in enabled:
            self.s3_index.prune_b2_high_frequency_keys()
        if (
            "address_number_locality" in enabled
            or "address_distinctive_pair" in enabled
            or "b5a" in enabled
            or "b5b" in enabled
        ):
            self.s3_index.prune_b5_high_frequency_keys()
        build_times["source3"] = time.perf_counter() - t0

        return build_times

    def build_indexes_from_entities(
        self,
        s2_entities: List[Entity],
        s3_entities: List[Entity]
    ) -> Dict[str, float]:
        """
        Build S2 and S3 inverted indexes from provided in-memory Entity lists.
        Used for fast deterministic integration testing (smoke test).
        Returns dict containing build elapsed times in seconds.
        """
        build_times: Dict[str, float] = {}
        enabled = set(self.config.enabled_passes)

        # 1. Build Source 2 index from entities
        t0 = time.perf_counter()
        self.s2_index.add_batch(s2_entities)
        if "distinctive_token_country" in enabled or "b2" in enabled:
            self.s2_index.prune_b2_high_frequency_keys()
        if (
            "address_number_locality" in enabled
            or "address_distinctive_pair" in enabled
            or "b5a" in enabled
            or "b5b" in enabled
        ):
            self.s2_index.prune_b5_high_frequency_keys()
        build_times["source2"] = time.perf_counter() - t0

        # 2. Build Source 3 index from entities
        t0 = time.perf_counter()
        self.s3_index.add_batch(s3_entities)
        if "distinctive_token_country" in enabled or "b2" in enabled:
            self.s3_index.prune_b2_high_frequency_keys()
        if (
            "address_number_locality" in enabled
            or "address_distinctive_pair" in enabled
            or "b5a" in enabled
            or "b5b" in enabled
        ):
            self.s3_index.prune_b5_high_frequency_keys()
        build_times["source3"] = time.perf_counter() - t0

        return build_times

    def generate_candidates_passes(self, s1_entity: Entity) -> Dict[str, Tuple[List[str], List[str]]]:
        """
        Generate candidate pairs partitioned by pass name.
        
        Returns:
            Dict containing tuples of (s2_candidates, s3_candidates) for
            'b1', 'b2', 'b3', 'b4', 'b5a', 'b5b',
            'cumulative_b1_b2', 'cumulative_b1_b2_b3', 'cumulative_b1_b4',
            'cumulative_b1_b5a', and 'cumulative'.
        """
        s2_passes = self.s2_index.query_passes(s1_entity)
        s3_passes = self.s3_index.query_passes(s1_entity)
        return {
            "b1": (s2_passes["b1"], s3_passes["b1"]),
            "b2": (s2_passes["b2"], s3_passes["b2"]),
            "b3": (s2_passes["b3"], s3_passes["b3"]),
            "b4": (s2_passes["b4"], s3_passes["b4"]),
            "b5a": (s2_passes["b5a"], s3_passes["b5a"]),
            "b5b": (s2_passes["b5b"], s3_passes["b5b"]),
            "cumulative_b1_b2": (s2_passes["cumulative_b1_b2"], s3_passes["cumulative_b1_b2"]),
            "cumulative_b1_b2_b3": (s2_passes["cumulative_b1_b2_b3"], s3_passes["cumulative_b1_b2_b3"]),
            "cumulative_b1_b4": (s2_passes["cumulative_b1_b4"], s3_passes["cumulative_b1_b4"]),
            "cumulative_b1_b5a": (s2_passes["cumulative_b1_b5a"], s3_passes["cumulative_b1_b5a"]),
            "cumulative": (s2_passes["cumulative"], s3_passes["cumulative"])
        }

    def generate_candidates(self, s1_entity: Entity) -> Tuple[List[str], List[str]]:
        """
        Generate uncapped cumulative candidates for an S1 entity across S2 and S3.
        """
        passes = self.generate_candidates_passes(s1_entity)
        return passes["cumulative"]

    def get_raw_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Fast O(1) lookup of raw Entity object for target entities in S2 or S3.
        Returns None if entity_id is not in index.
        """
        if entity_id.startswith("S2-") and entity_id in self.s2_index.entity_store_raw:
            name, addr, country = self.s2_index.entity_store_raw[entity_id]
            return Entity(entity_id=entity_id, business_name=name, business_address=addr, country=country)
        elif entity_id.startswith("S3-") and entity_id in self.s3_index.entity_store_raw:
            name, addr, country = self.s3_index.entity_store_raw[entity_id]
            return Entity(entity_id=entity_id, business_name=name, business_address=addr, country=country)
        return None

