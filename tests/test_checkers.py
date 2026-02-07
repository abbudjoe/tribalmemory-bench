"""Tests for answer checkers."""

import pytest
from benchmarks.shared.checkers import (
    normalize_text,
    substring_checker,
    phrase_checker,
    fuzzy_checker,
    abstention_checker,
    create_checker,
)


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello WORLD") == "hello world"
    
    def test_removes_punctuation(self):
        assert normalize_text("Hello, world!") == "hello world"
    
    def test_preserves_apostrophes(self):
        assert normalize_text("I'm here") == "i'm here"
    
    def test_removes_articles(self):
        assert normalize_text("the quick brown fox") == "quick brown fox"
        assert normalize_text("a cat and an apple") == "cat and apple"
    
    def test_normalizes_whitespace(self):
        assert normalize_text("hello   world") == "hello world"


class TestSubstringChecker:
    def test_exact_match(self):
        assert substring_checker("hello", ["hello world"])
    
    def test_case_insensitive(self):
        assert substring_checker("HELLO", ["hello world"])
    
    def test_no_match(self):
        assert not substring_checker("goodbye", ["hello world"])
    
    def test_empty_expected(self):
        assert not substring_checker("", ["hello"])
    
    def test_empty_retrieved(self):
        assert not substring_checker("hello", [])
    
    def test_multiple_retrieved(self):
        assert substring_checker("world", ["hello", "brave new world"])


class TestPhraseChecker:
    def test_single_phrase(self):
        assert phrase_checker("New York", ["I live in New York City"])
    
    def test_comma_separated(self):
        # Each phrase must be > 2 chars after normalization
        assert phrase_checker("apples, oranges", ["I bought some apples and oranges"])
    
    def test_semicolon_separated(self):
        # "blue" is > 2 chars, should match
        assert phrase_checker("red; blue", ["The sky is blue today"])
    
    def test_short_phrases_ignored(self):
        # Very short phrases (< 3 chars) should be ignored
        assert not phrase_checker("a, b", ["nothing here"])
    
    def test_direct_match_priority(self):
        # Direct match should work even if phrases don't
        assert phrase_checker("hello world", ["hello world is here"])


class TestFuzzyChecker:
    def test_exact_match(self):
        assert fuzzy_checker("hello world", ["hello world"], threshold=0.8)
    
    def test_partial_match(self):
        # "vegetarian" should match content containing it
        assert fuzzy_checker("vegetarian", ["I became vegetarian last year"])
    
    def test_below_threshold(self):
        assert not fuzzy_checker("completely different", ["hello world"], threshold=0.8)
    
    def test_empty_inputs(self):
        assert not fuzzy_checker("", ["hello"])
        assert not fuzzy_checker("hello", [])


class TestAbstentionChecker:
    def test_abstention_with_no_retrieval(self):
        expected = "There is no information to answer this question"
        assert abstention_checker(expected, [])
    
    def test_abstention_with_short_retrieval(self):
        expected = "I don't know the answer"
        assert abstention_checker(expected, ["ok"])  # Too short to be relevant
    
    def test_non_abstention_falls_back(self):
        # Regular question should use phrase checking
        expected = "Paris"
        assert abstention_checker(expected, ["The capital of France is Paris"])


class TestCreateChecker:
    def test_create_substring(self):
        checker = create_checker("substring")
        assert checker("hello", ["hello world"])
    
    def test_create_phrase(self):
        checker = create_checker("phrase")
        # "hello" phrase should match in "hello there"
        assert checker("hello; world", ["hello there and world"])
    
    def test_create_fuzzy(self):
        checker = create_checker("fuzzy", threshold=0.5)
        assert checker("hello", ["hello world"])
    
    def test_create_abstention(self):
        checker = create_checker("abstention")
        assert checker("no information", [])
    
    def test_invalid_method(self):
        with pytest.raises(ValueError):
            create_checker("invalid")
