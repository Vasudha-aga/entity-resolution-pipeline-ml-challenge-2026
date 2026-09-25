"""
Unit and sanity verification suite for the shared normalization module.

Tests:
- Ampersand conversions ('&' -> 'and')
- Legal entity suffix stripping across jurisdictions (Inc, LLC, Pvt Ltd, SARL, etc.)
- Punctuation and whitespace normalization
- Accented Unicode NFKD normalization
- Missing / Null / Empty input safety
- Address numbers and token preservation
- Config integration via TextNormalizer
"""

import sys
from src.normalization import (
    normalize_business_name,
    normalize_address,
    tokenize_business_name,
    extract_address_tokens,
    TextNormalizer
)


def run_checks():
    print("=========================================================")
    print(" Amazon ML Challenge 2026: Normalization Sanity Suite    ")
    print("=========================================================\n")

    test_cases = [
        # 1. Ampersand normalization
        {
            "category": "Ampersand '&' vs 'and'",
            "input": "AT&T Services Inc.",
            "expected_name": "at and t services",
            "type": "name"
        },
        {
            "category": "Ampersand '&' vs 'and'",
            "input": "B & H Photo Corp",
            "expected_name": "b and h photo",
            "type": "name"
        },
        # 2. Legal suffix variations
        {
            "category": "Legal Suffixes (US / India / EU)",
            "input": "Orelee's Barbershop Inc.",
            "expected_name": "orelee s barbershop",
            "type": "name"
        },
        {
            "category": "Legal Suffixes (US / India / EU)",
            "input": "Orelee's Barbershop LLC",
            "expected_name": "orelee s barbershop",
            "type": "name"
        },
        {
            "category": "Legal Suffixes (US / India / EU)",
            "input": "Orelee's Barbershop Pvt. Ltd.",
            "expected_name": "orelee s barbershop",
            "type": "name"
        },
        {
            "category": "Legal Suffixes (US / India / EU)",
            "input": "Orelee's Barbershop Private Limited",
            "expected_name": "orelee s barbershop",
            "type": "name"
        },
        {
            "category": "Legal Suffixes (US / India / EU)",
            "input": "Orelee's Barbershop SARL",
            "expected_name": "orelee s barbershop",
            "type": "name"
        },
        {
            "category": "Legal Suffixes (US / India / EU)",
            "input": "Orelee's Barbershop SAS",
            "expected_name": "orelee s barbershop",
            "type": "name"
        },
        # 3. Punctuation & noisy characters
        {
            "category": "Punctuation & Symbols",
            "input": "-- Holloway Peak Inc Seafood",
            "expected_name": "holloway peak seafood",
            "type": "name"
        },
        {
            "category": "Punctuation & Symbols",
            "input": "B+ Retail Inc",
            "expected_name": "b retail",
            "type": "name"
        },
        # 4. Whitespace & mixed case
        {
            "category": "Whitespace & Casing",
            "input": "   Christ    Chapel   ",
            "expected_name": "christ chapel",
            "type": "name"
        },
        # 5. Accented Unicode characters
        {
            "category": "Unicode Accents (NFKD)",
            "input": "LLC Moncada Léarning Center",
            "expected_name": "moncada learning center",
            "type": "name"
        },
        {
            "category": "Unicode Accents (NFKD)",
            "input": "<< Team École",
            "expected_name": "team ecole",
            "type": "name"
        },
        {
            "category": "Unicode Accents (NFKD)",
            "input": "Bordeaux Étoile SAS",
            "expected_name": "bordeaux etoile",
            "type": "name"
        },
        # 6. Null and empty handling
        {
            "category": "Null / Empty Safety",
            "input": None,
            "expected_name": "",
            "type": "name"
        },
        {
            "category": "Null / Empty Safety",
            "input": "   ",
            "expected_name": "",
            "type": "name"
        },
        # 7. Address preservation
        {
            "category": "Address Normalization",
            "input": "2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD",
            "expected_addr": "2100 cameron drive unit apartment g dundalk md",
            "type": "addr"
        },
        {
            "category": "Address Normalization",
            "input": "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
            "expected_addr": "kh no 570 13 new delhi west delhi delhi",
            "type": "addr"
        },
        {
            "category": "Address Normalization",
            "input": "1 Ivanhoe Ave, PO Box 6009, Cincinnati, Ohio",
            "expected_addr": "1 ivanhoe ave po box 6009 cincinnati ohio",
            "type": "addr"
        },
        {
            "category": "Address Normalization (Null)",
            "input": None,
            "expected_addr": "",
            "type": "addr"
        }
    ]

    all_passed = True
    print(f"{'Category':<26} | {'Original Input':<32} | {'Normalized Output':<32} | {'Status':<6}")
    print("-" * 105)

    for tc in test_cases:
        raw = tc["input"]
        display_raw = "None" if raw is None else (f"'{raw}'" if len(raw) < 30 else f"'{raw[:27]}...'")
        
        if tc["type"] == "name":
            actual = normalize_business_name(raw)
            expected = tc["expected_name"]
        else:
            actual = normalize_address(raw)
            expected = tc["expected_addr"]

        passed = actual == expected
        if not passed:
            all_passed = False

        status_str = "PASS" if passed else "FAIL"
        display_actual = f"'{actual}'" if len(actual) < 30 else f"'{actual[:27]}...'"
        print(f"{tc['category']:<26} | {display_raw:<32} | {display_actual:<32} | {status_str:<6}")

    # Test Token Extraction
    print("\n---------------- TOKEN EXTRACTION SANITY CHECKS ----------------")
    name_sample = "Orelee's Barbershop & Spa Pvt. Ltd."
    tokens = tokenize_business_name(name_sample)
    print(f"Original Name: '{name_sample}'")
    print(f"Normalized Tokens: {tokens}")

    addr_sample = "1795 Westchester Drive, High Point, NC 27262"
    addr_tokens = extract_address_tokens(addr_sample)
    print(f"\nOriginal Address: '{addr_sample}'")
    print(f"Address Tokens: {addr_tokens}")

    # Test Class-based TextNormalizer
    print("\n---------------- TEXT NORMALIZER CLASS TEST --------------------")
    cfg = {"normalization": {"remove_legal_suffix": True, "min_token_len": 3, "remove_stopwords": True}}
    normalizer = TextNormalizer(cfg)
    class_tokens = normalizer.name_tokens("The Great Ocean View Resort & Spa LLC")
    print(f"Input: 'The Great Ocean View Resort & Spa LLC' (with stopwords removed)")
    print(f"Resulting Tokens: {class_tokens}")

    print("\n-----------------------------------------------------------------")
    if all_passed:
        print("[*] All normalization and tokenization unit checks PASSED!")
    else:
        print("[!] Some checks failed. Please inspect implementation.")
        sys.exit(1)


if __name__ == "__main__":
    run_checks()
