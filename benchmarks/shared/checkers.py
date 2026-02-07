"""Answer checking utilities for benchmarks."""

import re
from typing import Callable


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    
    - Lowercase
    - Remove extra whitespace
    - Remove punctuation except apostrophes
    - Strip articles (a, an, the)
    """
    text = text.lower().strip()
    # Remove punctuation except apostrophes
    text = re.sub(r"[^\w\s']", " ", text)
    # Remove articles (with word boundaries)
    text = re.sub(r"\b(a|an|the)\b", "", text)
    # Normalize whitespace (after removing articles to clean up gaps)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def substring_checker(expected: str, retrieved: list[str]) -> bool:
    """
    Basic substring matching.
    
    Returns True if expected answer appears in any retrieved memory.
    """
    if not expected or not retrieved:
        return False
    
    expected_norm = normalize_text(expected)
    combined = " ".join(normalize_text(r) for r in retrieved)
    
    return expected_norm in combined


def phrase_checker(expected: str, retrieved: list[str]) -> bool:
    """
    Check for key phrases from expected answer.
    
    Splits expected on common delimiters and checks if any phrase matches.
    """
    if not expected or not retrieved:
        return False
    
    combined = " ".join(normalize_text(r) for r in retrieved)
    
    # Direct match on normalized expected
    expected_norm = normalize_text(expected)
    if expected_norm in combined:
        return True
    
    # Split on delimiters BEFORE normalizing, then normalize each phrase
    phrases = re.split(r"[,;|]", expected)
    for phrase in phrases:
        phrase_norm = normalize_text(phrase)
        if phrase_norm and len(phrase_norm) > 2 and phrase_norm in combined:
            return True
    
    return False


def fuzzy_checker(
    expected: str,
    retrieved: list[str],
    threshold: float = 0.75,
) -> bool:
    """
    Fuzzy matching using character-level similarity.
    
    Uses a simple implementation if rapidfuzz is not available.
    
    Args:
        expected: Expected answer
        retrieved: List of retrieved memory contents
        threshold: Minimum similarity ratio (0.0 to 1.0)
    
    Returns:
        True if fuzzy match found above threshold
    """
    if not expected or not retrieved:
        return False
    
    expected_norm = normalize_text(expected)
    combined = " ".join(normalize_text(r) for r in retrieved)
    
    # Try rapidfuzz if available
    try:
        from rapidfuzz import fuzz
        ratio = fuzz.partial_ratio(expected_norm, combined) / 100.0
        return ratio >= threshold
    except ImportError:
        pass
    
    # Fallback: simple token overlap
    expected_tokens = set(expected_norm.split())
    combined_tokens = set(combined.split())
    
    if not expected_tokens:
        return False
    
    overlap = len(expected_tokens & combined_tokens)
    ratio = overlap / len(expected_tokens)
    
    return ratio >= threshold


def abstention_checker(expected: str, retrieved: list[str]) -> bool:
    """
    Check for correct abstention (not hallucinating).
    
    For abstention cases, success means NOT finding relevant content.
    This is the inverse of normal retrieval checking.
    
    Args:
        expected: Expected "no information" type answer
        retrieved: Retrieved memories
    
    Returns:
        True if system correctly has no relevant information
    """
    # Common abstention indicators in expected answer
    abstention_phrases = [
        "no information",
        "don't know",
        "cannot determine",
        "not mentioned",
        "no record",
        "not specified",
        "unable to answer",
        "no prior conversation",
    ]
    
    expected_lower = expected.lower()
    is_abstention_expected = any(p in expected_lower for p in abstention_phrases)
    
    if not is_abstention_expected:
        # Not an abstention case, use normal checking
        return phrase_checker(expected, retrieved)
    
    # For abstention: success is having NO highly relevant content
    # If retrieved is empty or content is generic, that's correct
    if not retrieved:
        return True
    
    # Check if retrieved content is actually relevant
    # (This is a heuristic - LLM judge would be more accurate)
    combined = " ".join(normalize_text(r) for r in retrieved[:3])  # Top 3
    
    # If the combined content is short or generic, likely not relevant
    if len(combined) < 50:
        return True
    
    return False


def create_checker(
    method: str = "phrase",
    threshold: float = 0.75,
) -> Callable[[str, list[str]], bool]:
    """
    Factory function to create an answer checker.
    
    Args:
        method: One of "substring", "phrase", "fuzzy", "abstention"
        threshold: Threshold for fuzzy matching
    
    Returns:
        Checker function
    """
    if method == "substring":
        return substring_checker
    elif method == "phrase":
        return phrase_checker
    elif method == "fuzzy":
        return lambda e, r: fuzzy_checker(e, r, threshold)
    elif method == "abstention":
        return abstention_checker
    else:
        raise ValueError(f"Unknown checker method: {method}")
