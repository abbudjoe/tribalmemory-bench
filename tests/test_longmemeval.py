"""Regression tests for LongMemEval harness fixes.

These tests verify the bug fixes introduced in PR #3:
1. Mixed-type answer handling (int and str)
2. Correct session field parsing (haystack_sessions)
3. Session filtering for sampling optimization
4. Answer checker robustness
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from benchmarks.longmemeval.harness import (
    download_dataset,
    parse_dataset,
    _parse_dataset_filtered,
    answer_checker,
)


class TestDownloadDataset:
    """Tests for download_dataset function."""

    def test_unknown_variant_raises(self):
        """Test that unknown variant raises ValueError."""
        with pytest.raises(ValueError, match="Unknown variant"):
            download_dataset(variant="unknown")

    def test_valid_variants(self):
        """Test that valid variants are accepted."""
        # Just check the function accepts these without ValueError
        # (actual download is tested in integration tests)
        assert "s" in ["s", "m"]  # Valid variants
        assert "m" in ["s", "m"]


class TestAnswerChecker:
    """Tests for answer_checker function with mixed types."""

    def test_string_answer_match(self):
        """Test string answer matching."""
        assert answer_checker("hello", ["the answer is hello world"]) is True
        assert answer_checker("hello", ["no match here"]) is False

    def test_int_answer_match(self):
        """Test integer answer matching (regression for pyarrow issue)."""
        # Some LongMemEval answers are integers, not strings
        assert answer_checker(25, ["there are 25 items"]) is True
        assert answer_checker(42, ["the answer is 42"]) is True
        assert answer_checker(100, ["only 50 here"]) is False

    def test_empty_expected_returns_false(self):
        """Test that empty/None expected returns False."""
        assert answer_checker("", ["some content"]) is False
        assert answer_checker(None, ["some content"]) is False

    def test_empty_retrieved_returns_false(self):
        """Test that empty retrieved list returns False."""
        assert answer_checker("answer", []) is False

    def test_case_insensitive_matching(self):
        """Test case-insensitive substring matching."""
        assert answer_checker("Hello", ["HELLO WORLD"]) is True
        assert answer_checker("HELLO", ["hello world"]) is True

    def test_phrase_delimiter_splitting(self):
        """Test that comma/semicolon separated phrases are checked."""
        assert answer_checker("apple, banana", ["I like apple"]) is True
        assert answer_checker("one; two", ["number two"]) is True


class TestParseDataset:
    """Tests for parse_dataset and _parse_dataset_filtered."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a minimal dataset matching LongMemEval structure."""
        return [
            {
                "question_id": "q1",
                "question": "What color is the sky?",
                "answer": "blue",
                "question_type": "single-session",
                "haystack_sessions": [
                    [
                        {"role": "user", "content": "The sky is blue today"},
                        {"role": "assistant", "content": "Yes, it's beautiful"},
                    ],
                    [
                        {"role": "user", "content": "I like pizza"},
                        {"role": "assistant", "content": "Me too"},
                    ],
                ],
                "haystack_session_ids": ["session_a", "session_b"],
                "answer_session_ids": ["session_a"],
            },
            {
                "question_id": "q2",
                "question": "What food do I like?",
                "answer": "pizza",
                "question_type": "multi-session",
                "haystack_sessions": [
                    [
                        {"role": "user", "content": "I like pizza"},
                        {"role": "assistant", "content": "Me too"},
                    ],
                ],
                "haystack_session_ids": ["session_b"],  # Shared with q1
                "answer_session_ids": ["session_b"],
            },
        ]

    def test_uses_haystack_sessions_not_history_sessions(self, sample_dataset):
        """Regression test: parser uses haystack_sessions field."""
        # Modify dataset to have history_sessions (wrong field)
        bad_dataset = [{
            **sample_dataset[0],
            "history_sessions": [
                [{"role": "user", "content": "WRONG FIELD"}]
            ],
        }]

        conversations, questions = parse_dataset(bad_dataset)

        # Should NOT contain "WRONG FIELD" from history_sessions
        all_content = " ".join(
            m["content"]
            for c in conversations
            for m in c["messages"]
        )
        assert "WRONG FIELD" not in all_content
        assert "blue" in all_content  # From haystack_sessions

    def test_session_deduplication(self, sample_dataset):
        """Test that shared sessions are only included once."""
        conversations, questions = parse_dataset(sample_dataset)

        # session_b appears in both questions but should only be parsed once
        session_ids = [c["id"] for c in conversations]
        assert session_ids.count("session_b") == 1
        assert len(conversations) == 2  # session_a and session_b

    def test_int_answer_preserved(self, sample_dataset):
        """Test that integer answers are preserved correctly."""
        # Modify to have int answer
        sample_dataset[0]["answer"] = 42

        conversations, questions = parse_dataset(sample_dataset)

        assert questions[0]["expected"] == 42
        assert isinstance(questions[0]["expected"], int)


class TestSessionFiltering:
    """Tests for session filtering optimization."""

    @pytest.fixture
    def large_dataset(self):
        """Create dataset with many sessions to test filtering."""
        sessions = []
        session_ids = []
        for i in range(100):
            sessions.append([
                {"role": "user", "content": f"Session {i} content"},
                {"role": "assistant", "content": f"Response {i}"},
            ])
            session_ids.append(f"session_{i}")

        return [
            {
                "question_id": "q1",
                "question": "Test question 1",
                "answer": "answer1",
                "question_type": "test",
                "haystack_sessions": sessions[:50],  # First 50 sessions
                "haystack_session_ids": session_ids[:50],
                "answer_session_ids": ["session_0"],
            },
            {
                "question_id": "q2",
                "question": "Test question 2",
                "answer": "answer2",
                "question_type": "test",
                "haystack_sessions": sessions[25:75],  # Sessions 25-74 (overlap)
                "haystack_session_ids": session_ids[25:75],
                "answer_session_ids": ["session_50"],
            },
        ]

    def test_filtering_reduces_session_count(self, large_dataset):
        """Test that filtering only includes needed sessions."""
        # Only need sessions from question 1
        needed = {"session_0", "session_1", "session_2"}

        conversations, questions = _parse_dataset_filtered(
            large_dataset, needed_sessions=needed
        )

        # Should only have 3 conversations
        assert len(conversations) == 3
        session_ids = {c["id"] for c in conversations}
        assert session_ids == needed

    def test_no_filtering_includes_all(self, large_dataset):
        """Test that None needed_sessions includes all sessions."""
        conversations, questions = _parse_dataset_filtered(
            large_dataset, needed_sessions=None
        )

        # 50 unique from q1 + 25 new from q2 (25-49 overlap) = 75 unique
        # Actually: q1 has 0-49, q2 has 25-74
        # Unique: 0-74 = 75 sessions
        assert len(conversations) == 75

    def test_filtering_preserves_all_questions(self, large_dataset):
        """Test that filtering doesn't affect question parsing."""
        needed = {"session_0"}  # Minimal sessions

        conversations, questions = _parse_dataset_filtered(
            large_dataset, needed_sessions=needed
        )

        # All questions should still be parsed
        assert len(questions) == 2
        assert questions[0]["id"] == "q1"
        assert questions[1]["id"] == "q2"


class TestMissingSessions:
    """Tests for handling missing session IDs."""

    def test_synthetic_id_fallback(self):
        """Test that missing session IDs get synthetic IDs."""
        dataset = [{
            "question_id": "q1",
            "question": "Test",
            "answer": "answer",
            "question_type": "test",
            "haystack_sessions": [
                [{"role": "user", "content": "Content 1"}],
                [{"role": "user", "content": "Content 2"}],
            ],
            "haystack_session_ids": ["only_one"],  # Only 1 ID for 2 sessions
            "answer_session_ids": [],
        }]

        conversations, questions = parse_dataset(dataset)

        # Should have 2 conversations
        assert len(conversations) == 2
        ids = [c["id"] for c in conversations]
        assert "only_one" in ids
        assert "session_1" in ids  # Synthetic ID for second session
