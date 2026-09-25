"""
Blocking and Candidate Key Analysis for Amazon ML Challenge 2026.

Analyzes normalization strategies, token distributions, block size distributions,
and ground-truth key alignment across Source 1, Source 2, and Source 3.
"""

import re
import unicodedata
from typing import List, Set, Dict, Tuple, Optional
from collections import Counter, defaultdict
import statistics

# Common business legal suffixes across international markets (US, India, France, etc.)
LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
    "pvt", "private", "pvt ltd", "private limited", "co", "company",
    "llp", "gmbh", "sarl", "sas", "sa", "plc", "nv", "bv", "ag", "spa"
}

# Regex for stripping punctuation and extra whitespace
PUNCT_REGEX = re.compile(r"[^\w\s]")
WHITESPACE_REGEX = re.compile(r"\s+")


def normalize_text(text: str, remove_legal_suffix: bool = True) -> str:
    """
    Standardize text:
    - Unicode NFKD normalization (converts accented chars to base ASCII)
    - Lowercase
    - Replace '&' with 'and'
    - Strip punctuation and symbols
    - Optional legal suffix removal
    - Whitespace normalization
    """
    if not text:
        return ""

    # Unicode decomposition
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")

    # Lowercase & symbol expansions
    text = text.lower()
    text = text.replace("&", " and ")

    # Remove punctuation
    text = PUNCT_REGEX.sub(" ", text)
    text = WHITESPACE_REGEX.sub(" ", text).strip()

    if remove_legal_suffix:
        tokens = text.split()
        # Remove trailing legal suffix combinations (e.g., "pvt ltd", "inc")
        if len(tokens) >= 2 and f"{tokens[-2]} {tokens[-1]}" in LEGAL_SUFFIXES:
            tokens = tokens[:-2]
        elif tokens and tokens[-1] in LEGAL_SUFFIXES:
            tokens = tokens[:-1]
        text = " ".join(tokens).strip()

    return text


def extract_tokens(text: str, min_len: int = 2) -> List[str]:
    """
    Extract normalized alphanumeric tokens of minimum length.
    """
    normalized = normalize_text(text, remove_legal_suffix=False)
    tokens = [t for t in normalized.split() if len(t) >= min_len]
    return tokens


def extract_address_components(address: str) -> Dict[str, str]:
    """
    Parse generic address tokens and identify candidate postal/state tokens.
    """
    if not address or not address.strip():
        return {"raw_normalized": "", "tokens": [], "has_postal": False, "postal_token": ""}

    normalized = normalize_text(address, remove_legal_suffix=False)
    tokens = normalized.split()

    # Look for 5 or 6 digit postal codes (common in US & India)
    postal_token = ""
    for t in tokens:
        if t.isdigit() and len(t) in (5, 6):
            postal_token = t
            break

    return {
        "raw_normalized": normalized,
        "tokens": tokens,
        "has_postal": bool(postal_token),
        "postal_token": postal_token
    }


class BlockingAnalyzer:
    """Analyzes candidate key distributions and block size statistics."""

    def __init__(self):
        pass

    def compute_block_stats(self, key_counter: Counter) -> Dict[str, float]:
        """
        Compute summary statistics for block sizes.
        """
        if not key_counter:
            return {"total_blocks": 0, "max_block": 0, "mean_block": 0, "p95": 0, "p99": 0}

        sizes = sorted(key_counter.values())
        n = len(sizes)
        return {
            "total_unique_blocks": n,
            "min_block_size": min(sizes),
            "max_block_size": max(sizes),
            "mean_block_size": round(statistics.mean(sizes), 2),
            "median_block_size": round(statistics.median(sizes), 2),
            "p95_block_size": sizes[int(n * 0.95)] if n > 0 else 0,
            "p99_block_size": sizes[int(n * 0.99)] if n > 0 else 0,
            "blocks_over_100": sum(1 for s in sizes if s > 100),
            "blocks_over_500": sum(1 for s in sizes if s > 500),
            "blocks_over_1000": sum(1 for s in sizes if s > 1000),
        }
