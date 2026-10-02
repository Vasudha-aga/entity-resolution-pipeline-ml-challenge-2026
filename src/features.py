"""
Pairwise feature extraction module for Amazon ML Challenge 2026.

Extracts deterministic similarity features for candidate pairs (S1, S2) and (S1, S3),
reusing existing text normalization routines from src.normalization.
"""

import os
from typing import Dict, List, Tuple, Optional, Set, Iterator
from dataclasses import dataclass
import numpy as np

from src.normalization import TextNormalizer
from src.data_loader import DataLoader, Entity


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute exact Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    prev_row = list(range(len2 + 1))
    for i in range(len1):
        curr_row = [i + 1] + [0] * len2
        char1 = s1[i]
        for j in range(len2):
            cost = 0 if char1 == s2[j] else 1
            curr_row[j + 1] = min(
                curr_row[j] + 1,       # Insertion
                prev_row[j + 1] + 1,   # Deletion
                prev_row[j] + cost     # Substitution
            )
        prev_row = curr_row
    return prev_row[len2]


def normalized_levenshtein_similarity(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity: 1 - dist / max_len."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (dist / max_len)


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
    """
    Compute Jaro-Winkler similarity between two strings.
    Handles null/empty inputs safely.
    """
    if not s1 or not s2:
        return 1.0 if s1 == s2 else 0.0
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3.0

    l = 0
    for i in range(min(len1, len2, max_l)):
        if s1[i] == s2[i]:
            l += 1
        else:
            break

    return jaro + l * p * (1.0 - jaro)


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from a string."""
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 1.0 if set1 == set2 else 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def numeric_token_overlap(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute overlap ratio of numeric tokens between two token lists."""
    nums1 = {t for t in tokens1 if t.isdigit()}
    nums2 = {t for t in tokens2 if t.isdigit()}
    if not nums1 and not nums2:
        return 1.0
    if not nums1 or not nums2:
        return 0.0
    inter = len(nums1 & nums2)
    return inter / max(1, min(len(nums1), len(nums2)))


@dataclass(slots=True)
class NormalizedEntity:
    """Pre-normalized memory-efficient entity representation."""
    entity_id: str
    raw_name: str
    raw_address: str
    country: str
    norm_name: str
    norm_address: str
    name_tokens: List[str]
    addr_tokens: List[str]
    name_char_3grams: Set[str]
    addr_char_3grams: Set[str]

    @classmethod
    def from_entity(cls, entity: Entity, normalizer: TextNormalizer) -> "NormalizedEntity":
        norm_name = normalizer.normalize_name(entity.business_name)
        norm_addr = normalizer.normalize_addr(entity.business_address)
        name_tokens = normalizer.name_tokens(entity.business_name)
        addr_tokens = normalizer.addr_tokens(entity.business_address)
        name_ngrams = get_char_ngrams(norm_name, 3)
        addr_ngrams = get_char_ngrams(norm_addr, 3)

        return cls(
            entity_id=entity.entity_id,
            raw_name=entity.business_name,
            raw_address=entity.business_address,
            country=entity.country.strip().lower(),
            norm_name=norm_name,
            norm_address=norm_addr,
            name_tokens=name_tokens,
            addr_tokens=addr_tokens,
            name_char_3grams=name_ngrams,
            addr_char_3grams=addr_ngrams
        )


class PairwiseFeatureExtractor:
    """
    Extracts 25 pairwise similarity feature vectors for candidate pairs (S1, Candidate).
    """

    FEATURE_NAMES = [
        "name_exact_match",
        "name_token_jaccard",
        "name_token_dice",
        "name_token_overlap",
        "name_char_3gram_jaccard",
        "name_jaro_winkler",
        "name_levenshtein_sim",
        "name_prefix_match",
        "name_length_ratio",
        "name_len_diff",
        "s1_name_missing",
        "cand_name_missing",
        "addr_exact_match",
        "addr_token_jaccard",
        "addr_token_overlap",
        "addr_char_3gram_jaccard",
        "addr_numeric_overlap",
        "addr_len_diff",
        "s1_addr_missing",
        "cand_addr_missing",
        "both_addr_missing",
        "name_addr_combined_match",
        "country_exact_match",
        "is_s2",
        "is_s3"
    ]

    def __init__(self, normalizer: Optional[TextNormalizer] = None):
        self.normalizer = normalizer or TextNormalizer()

    def get_feature_names(self) -> List[str]:
        return list(self.FEATURE_NAMES)

    def extract_pair_features(self, e1: NormalizedEntity, e2: NormalizedEntity) -> np.ndarray:
        """
        Extract numerical feature vector for a single pair of normalized entities.
        Returns a 1D float32 numpy array of shape (25,).
        """
        # 1. Business Name Features
        s1_name_missing = 1.0 if not e1.norm_name else 0.0
        cand_name_missing = 1.0 if not e2.norm_name else 0.0

        name_exact = 1.0 if (e1.norm_name and e1.norm_name == e2.norm_name) else 0.0

        t1, t2 = set(e1.name_tokens), set(e2.name_tokens)
        if not t1 or not t2:
            token_jaccard = 1.0 if t1 == t2 else 0.0
            token_dice = 1.0 if t1 == t2 else 0.0
            token_overlap = 1.0 if t1 == t2 else 0.0
        else:
            inter = len(t1 & t2)
            token_jaccard = inter / len(t1 | t2)
            token_dice = (2.0 * inter) / (len(t1) + len(t2))
            token_overlap = inter / min(len(t1), len(t2))

        char_3gram_jaccard = jaccard_similarity(e1.name_char_3grams, e2.name_char_3grams)
        jw_sim = jaro_winkler_similarity(e1.norm_name, e2.norm_name)
        lev_sim = normalized_levenshtein_similarity(e1.norm_name, e2.norm_name)

        p1 = e1.name_tokens[0] if e1.name_tokens else ""
        p2 = e2.name_tokens[0] if e2.name_tokens else ""
        prefix_match = 1.0 if (p1 and p1 == p2) else 0.0

        l1, l2 = len(e1.norm_name), len(e2.norm_name)
        max_l = max(l1, l2)
        len_ratio = (min(l1, l2) / max_l) if max_l > 0 else 1.0
        len_diff = float(abs(l1 - l2))

        # 2. Address Features
        s1_addr_missing = 1.0 if not e1.norm_address else 0.0
        cand_addr_missing = 1.0 if not e2.norm_address else 0.0
        both_addr_missing = 1.0 if (s1_addr_missing and cand_addr_missing) else 0.0

        addr_exact = 1.0 if (e1.norm_address and e1.norm_address == e2.norm_address) else 0.0

        at1, at2 = set(e1.addr_tokens), set(e2.addr_tokens)
        if not at1 or not at2:
            addr_jaccard = 1.0 if at1 == at2 else 0.0
            addr_overlap = 1.0 if at1 == at2 else 0.0
        else:
            addr_inter = len(at1 & at2)
            addr_jaccard = addr_inter / len(at1 | at2)
            addr_overlap = addr_inter / min(len(at1), len(at2))

        addr_char_jaccard = jaccard_similarity(e1.addr_char_3grams, e2.addr_char_3grams)
        addr_num_overlap = numeric_token_overlap(e1.addr_tokens, e2.addr_tokens)
        addr_l1, addr_l2 = len(e1.norm_address), len(e2.norm_address)
        addr_len_diff = float(abs(addr_l1 - addr_l2))

        # Combined Interaction
        name_addr_combined = float(name_exact * addr_exact)

        # 3. Country & Source Features
        country_match = 1.0 if (e1.country and e1.country == e2.country) else 0.0
        is_s2 = 1.0 if e2.entity_id.startswith("S2-") else 0.0
        is_s3 = 1.0 if e2.entity_id.startswith("S3-") else 0.0

        return np.array([
            name_exact,
            token_jaccard,
            token_dice,
            token_overlap,
            char_3gram_jaccard,
            jw_sim,
            lev_sim,
            prefix_match,
            len_ratio,
            len_diff,
            s1_name_missing,
            cand_name_missing,
            addr_exact,
            addr_jaccard,
            addr_overlap,
            addr_char_jaccard,
            addr_num_overlap,
            addr_len_diff,
            s1_addr_missing,
            cand_addr_missing,
            both_addr_missing,
            name_addr_combined,
            country_match,
            is_s2,
            is_s3
        ], dtype=np.float32)

    def extract_batch_features(
        self,
        pairs: List[Tuple[str, str]],
        entity_store: Dict[str, NormalizedEntity]
    ) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
        """
        Extract features for a batch of candidate pairs (s1_id, cand_id).
        Returns (feature_matrix of shape (N, 25), list of valid pairs).
        """
        features_list = []
        valid_pairs = []

        for s1_id, cand_id in pairs:
            e1 = entity_store.get(s1_id)
            e2 = entity_store.get(cand_id)
            if e1 is None or e2 is None:
                continue

            feat = self.extract_pair_features(e1, e2)
            features_list.append(feat)
            valid_pairs.append((s1_id, cand_id))

        if not features_list:
            return np.empty((0, len(self.FEATURE_NAMES)), dtype=np.float32), []

        return np.vstack(features_list), valid_pairs
