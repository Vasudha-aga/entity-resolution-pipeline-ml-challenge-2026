"""
Shared text normalization module for Amazon ML Challenge 2026.

Provides deterministic, country-agnostic normalization for business names,
addresses, and token extraction used across blocking and feature engineering.
"""

import re
import unicodedata
from typing import List, Set, Optional, Dict, Any

# Multi-word and single-word legal corporate suffixes across major jurisdictions
MULTI_WORD_LEGAL_SUFFIXES = [
    "private limited",
    "pvt ltd",
    "pvt limited",
    "pvt. ltd.",
    "pvt. limited",
    "co ltd",
    "co. ltd.",
    "pty ltd",
    "holding corp",
    "holdings ltd"
]

SINGLE_WORD_LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
    "pvt", "private", "co", "company", "llp", "gmbh", "sarl", "sas", "sa",
    "plc", "nv", "bv", "ag", "sl", "srl", "cie"
}

# Generic grammatical stopwords (configurable)
DEFAULT_NAME_STOPWORDS = {
    "the", "a", "an", "of", "and", "in", "for", "on", "at", "by", "with"
}

# Precompiled regular expressions for performance
PUNCT_REGEX = re.compile(r"[^\w\s]")
WHITESPACE_REGEX = re.compile(r"\s+")
AMPERSAND_REGEX = re.compile(r"&")


def strip_accents_and_unicode(text: str) -> str:
    """
    Decompose Unicode characters into ASCII base representations.
    e.g., 'Léarning' -> 'Learning', 'Bordeaux Étoile' -> 'Bordeaux Etoile'
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    return decomposed.encode("ascii", "ignore").decode("utf-8")


def normalize_business_name(
    text: Optional[str],
    remove_legal_suffix: bool = True,
    legal_suffixes: Optional[Set[str]] = None
) -> str:
    """
    Standardize a business entity name:
    1. Null/empty safety
    2. Unicode NFKD accent stripping
    3. Lowercase
    4. '&' -> 'and'
    5. Punctuation removal
    6. Whitespace collapse
    7. Leading, trailing, and standalone legal corporate entity suffix stripping
    """
    if text is None or not isinstance(text, str) or not text.strip():
        return ""

    # 1. Unicode decomposition
    s = strip_accents_and_unicode(text)

    # 2. Lowercase & symbol expansions
    s = s.lower()
    s = AMPERSAND_REGEX.sub(" and ", s)

    # 3. Punctuation removal & whitespace normalization
    s = PUNCT_REGEX.sub(" ", s)
    s = WHITESPACE_REGEX.sub(" ", s).strip()

    if not s:
        return ""

    if remove_legal_suffix:
        # Strip trailing multi-word suffixes
        for m_suffix in MULTI_WORD_LEGAL_SUFFIXES:
            if s.endswith(f" {m_suffix}"):
                s = s[:-len(m_suffix) - 1].strip()
                break

        tokens = s.split()
        active_suffixes = legal_suffixes or SINGLE_WORD_LEGAL_SUFFIXES

        # Strip leading legal suffixes (e.g. 'LLC Moncada...' -> 'Moncada...')
        while len(tokens) > 1 and tokens[0] in active_suffixes:
            tokens.pop(0)

        # Strip trailing legal suffixes (e.g. '...Barbershop Inc' -> '...Barbershop')
        while len(tokens) > 1 and tokens[-1] in active_suffixes:
            tokens.pop()

        # Clean any internal standalone corporate designators if preceded & followed by business words
        # e.g., 'holloway peak inc seafood' -> 'holloway peak seafood'
        filtered_tokens = []
        for idx, t in enumerate(tokens):
            if t in active_suffixes and len(tokens) > 2:
                continue
            filtered_tokens.append(t)

        s = " ".join(filtered_tokens if filtered_tokens else tokens).strip()

    return s


def normalize_address(text: Optional[str]) -> str:
    """
    Standardize a business address string:
    1. Null/empty safety (returns empty string)
    2. Unicode NFKD accent stripping
    3. Lowercase
    4. Replace symbols/slashes/punctuations with spaces
    5. Preserve numeric tokens, building numbers, postal codes, street names
    6. Whitespace collapse
    """
    if text is None or not isinstance(text, str) or not text.strip():
        return ""

    s = strip_accents_and_unicode(text)
    s = s.lower()
    s = AMPERSAND_REGEX.sub(" and ", s)
    s = PUNCT_REGEX.sub(" ", s)
    s = WHITESPACE_REGEX.sub(" ", s).strip()
    return s


def tokenize_business_name(
    text: Optional[str],
    remove_stopwords: bool = False,
    min_token_len: int = 2,
    remove_legal_suffix: bool = True
) -> List[str]:
    """
    Extract meaningful normalized tokens from a business name.
    """
    norm_text = normalize_business_name(text, remove_legal_suffix=remove_legal_suffix)
    if not norm_text:
        return []

    tokens = norm_text.split()
    result = []
    for t in tokens:
        if len(t) < min_token_len:
            continue
        if remove_stopwords and t in DEFAULT_NAME_STOPWORDS:
            continue
        result.append(t)
    return result


def extract_address_tokens(
    text: Optional[str],
    min_token_len: int = 2
) -> List[str]:
    """
    Extract useful normalized address tokens (streets, localities, cities, numbers, postal codes).
    """
    norm_addr = normalize_address(text)
    if not norm_addr:
        return []

    tokens = norm_addr.split()
    return [t for t in tokens if len(t) >= min_token_len]


class TextNormalizer:
    """
    Configurable normalizer class initialized with custom settings.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        norm_cfg = self.config.get("normalization", {})
        self.remove_legal_suffix = norm_cfg.get("remove_legal_suffix", True)
        self.min_token_len = norm_cfg.get("min_token_len", 2)
        self.remove_stopwords = norm_cfg.get("remove_stopwords", False)

    def normalize_name(self, text: Optional[str]) -> str:
        return normalize_business_name(text, remove_legal_suffix=self.remove_legal_suffix)

    def normalize_addr(self, text: Optional[str]) -> str:
        return normalize_address(text)

    def name_tokens(self, text: Optional[str]) -> List[str]:
        return tokenize_business_name(
            text,
            remove_stopwords=self.remove_stopwords,
            min_token_len=self.min_token_len,
            remove_legal_suffix=self.remove_legal_suffix
        )

    def addr_tokens(self, text: Optional[str]) -> List[str]:
        return extract_address_tokens(text, min_token_len=self.min_token_len)
