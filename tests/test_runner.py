"""Tests for benchmark runner."""

import pytest
from benchmarks.shared.runner import (
    stratified_sample,
    chunk_conversations,
)


class TestStratifiedSample:
    def test_basic_sampling(self):
        questions = [
            {"question": "q1", "category": "A"},
            {"question": "q2", "category": "A"},
            {"question": "q3", "category": "B"},
            {"question": "q4", "category": "B"},
        ]
        sample = stratified_sample(questions, 2, seed=42)
        assert len(sample) == 2
    
    def test_proportional_categories(self):
        questions = [
            {"question": f"a{i}", "category": "A"} for i in range(80)
        ] + [
            {"question": f"b{i}", "category": "B"} for i in range(20)
        ]
        sample = stratified_sample(questions, 10, seed=42)
        
        # Should be roughly proportional (8:2)
        a_count = sum(1 for q in sample if q["category"] == "A")
        b_count = sum(1 for q in sample if q["category"] == "B")
        
        assert a_count >= 6  # At least 60%
        assert b_count >= 1  # At least 1
    
    def test_reproducible_with_seed(self):
        questions = [{"question": f"q{i}", "category": "A"} for i in range(100)]
        
        sample1 = stratified_sample(questions, 10, seed=42)
        sample2 = stratified_sample(questions, 10, seed=42)
        
        assert sample1 == sample2
    
    def test_different_seeds_different_results(self):
        questions = [{"question": f"q{i}", "category": "A"} for i in range(100)]
        
        sample1 = stratified_sample(questions, 10, seed=42)
        sample2 = stratified_sample(questions, 10, seed=99)
        
        assert sample1 != sample2
    
    def test_empty_questions(self):
        sample = stratified_sample([], 10, seed=42)
        assert sample == []
    
    def test_sample_larger_than_questions(self):
        questions = [{"question": "q1", "category": "A"}]
        sample = stratified_sample(questions, 10, seed=42)
        assert len(sample) == 1


class TestChunkConversations:
    def test_basic_chunking(self):
        conversations = [{
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "session": "s1",
        }]
        
        chunks = chunk_conversations(conversations)
        assert len(chunks) == 1
        assert "user: Hello" in chunks[0]["content"]
        assert "assistant: Hi there" in chunks[0]["content"]
    
    def test_multi_turn_chunking(self):
        conversations = [{
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "Good"},
                {"role": "user", "content": "Great"},
                {"role": "assistant", "content": "Thanks"},
            ],
            "session": "s1",
        }]
        
        chunks = chunk_conversations(conversations)
        # Should create multiple chunks (4 messages max per chunk)
        assert len(chunks) >= 2
    
    def test_session_context(self):
        conversations = [{
            "messages": [{"role": "user", "content": "Hello"}],
            "session": "session-123",
        }]
        
        chunks = chunk_conversations(conversations)
        assert chunks[0]["context"] == "session:session-123"
    
    def test_empty_conversations(self):
        chunks = chunk_conversations([])
        assert chunks == []
    
    def test_empty_messages(self):
        conversations = [{"messages": [], "session": "s1"}]
        chunks = chunk_conversations(conversations)
        assert chunks == []
