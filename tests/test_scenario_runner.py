"""Tests for scenario_runner module."""

import pytest
import tempfile
from pathlib import Path
import yaml

from benchmarks.shared.scenario_runner import (
    load_scenario,
    load_scenarios_from_dir,
    run_scenario,
    check_expected_behavior,
    ScenarioResult,
)
from benchmarks.shared.providers import Memory


class MockProvider:
    """Mock provider for testing."""
    
    def __init__(self, recall_results=None):
        self.stored = []
        self.recall_results = recall_results or []
    
    async def store(self, content, context=None):
        self.stored.append({"content": content, "context": context})
        return f"id-{len(self.stored)}"
    
    async def recall(self, query, limit=10):
        return [
            Memory(id=str(i), content=r, relevance=0.9 - i * 0.1)
            for i, r in enumerate(self.recall_results[:limit])
        ]


class TestLoadScenario:
    @pytest.mark.asyncio
    async def test_load_valid_yaml(self):
        """Test loading a valid YAML scenario."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump({
                "name": "test_scenario",
                "category": "test",
                "conversations": [],
                "task": {"query": "test query"},
            }, f)
            f.flush()
            
            scenario = await load_scenario(Path(f.name))
            
            assert scenario["name"] == "test_scenario"
            assert scenario["category"] == "test"


class TestLoadScenariosFromDir:
    @pytest.mark.asyncio
    async def test_load_multiple_scenarios(self):
        """Test loading multiple scenarios from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two scenario files
            for i in range(2):
                path = Path(tmpdir) / f"scenario_{i}.yaml"
                with open(path, 'w') as f:
                    yaml.dump({
                        "name": f"scenario_{i}",
                        "category": "test",
                    }, f)
            
            scenarios = await load_scenarios_from_dir(Path(tmpdir))
            
            assert len(scenarios) == 2
    
    @pytest.mark.asyncio
    async def test_skip_underscore_files(self):
        """Test that files starting with _ are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a regular file and a combined file
            regular = Path(tmpdir) / "scenario.yaml"
            combined = Path(tmpdir) / "_combined.yaml"
            
            with open(regular, 'w') as f:
                yaml.dump({"name": "regular"}, f)
            with open(combined, 'w') as f:
                yaml.dump({"name": "combined"}, f)
            
            scenarios = await load_scenarios_from_dir(Path(tmpdir))
            
            assert len(scenarios) == 1
            assert scenarios[0]["name"] == "regular"
    
    @pytest.mark.asyncio
    async def test_empty_directory(self):
        """Test loading from an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scenarios = await load_scenarios_from_dir(Path(tmpdir))
            assert scenarios == []


class TestCheckExpectedBehavior:
    def test_negative_test_no_retrieval(self):
        """Negative test passes when nothing retrieved."""
        passed, fm, desc = check_expected_behavior(
            expected={"should_retrieve": False},
            success={},
            failure_modes=[],
            retrieved=[],
        )
        assert passed is True
    
    def test_negative_test_short_retrieval(self):
        """Negative test passes with very short retrieval."""
        passed, fm, desc = check_expected_behavior(
            expected={"should_retrieve": False},
            success={},
            failure_modes=[],
            retrieved=["ok"],  # Very short, not relevant
        )
        assert passed is True
    
    def test_negative_test_false_positive(self):
        """Negative test fails when significant content retrieved."""
        passed, fm, desc = check_expected_behavior(
            expected={"should_retrieve": False},
            success={},
            failure_modes=[],
            retrieved=["This is a long piece of retrieved content"],
        )
        assert passed is False
        assert fm == "false_positive"
    
    def test_positive_test_no_retrieval(self):
        """Positive test fails when nothing retrieved."""
        passed, fm, desc = check_expected_behavior(
            expected={"should_retrieve": True},
            success={},
            failure_modes=[],
            retrieved=[],
        )
        assert passed is False
        assert fm == "no_retrieval"
    
    def test_positive_test_contains_indicator(self):
        """Positive test passes when indicator found."""
        passed, fm, desc = check_expected_behavior(
            expected={"should_retrieve": True},
            success={"contains": ["vegetarian"]},
            failure_modes=[],
            retrieved=["I am vegetarian now"],
        )
        assert passed is True
    
    def test_positive_test_missing_indicator(self):
        """Positive test fails when indicator not found."""
        passed, fm, desc = check_expected_behavior(
            expected={"should_retrieve": True},
            success={"contains": ["vegetarian"]},
            failure_modes=[],
            retrieved=["I love steak"],
        )
        assert passed is False
        assert fm == "missing_expected"
    
    def test_unexpected_content(self):
        """Test fails when forbidden content found."""
        passed, fm, desc = check_expected_behavior(
            expected={"should_retrieve": True},
            success={
                "contains": ["restaurant"],
                "not_contains": ["steak"],
            },
            failure_modes=[],
            retrieved=["steakhouse restaurant"],
        )
        assert passed is False
        assert fm == "unexpected_content"
        assert "steak" in desc


class TestRunScenario:
    @pytest.mark.asyncio
    async def test_run_simple_scenario(self):
        """Test running a simple scenario."""
        scenario = {
            "name": "test_scenario",
            "category": "test",
            "conversations": [
                {
                    "session": "1",
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi"},
                    ],
                },
            ],
            "task": {
                "query": "test query",
                "expected_behavior": {"should_retrieve": True},
                "success": {"contains": ["hello"]},
            },
        }
        
        provider = MockProvider(recall_results=["hello world"])
        result = await run_scenario(scenario, provider)
        
        assert result.name == "test_scenario"
        assert result.category == "test"
        assert result.passed is True
        assert len(provider.stored) == 1  # One chunk stored
    
    @pytest.mark.asyncio
    async def test_run_scenario_failure(self):
        """Test running a scenario that fails."""
        scenario = {
            "name": "failing_scenario",
            "category": "test",
            "conversations": [],
            "task": {
                "query": "test query",
                "expected_behavior": {"should_retrieve": True},
                "success": {"contains": ["specific_answer"]},
            },
        }
        
        provider = MockProvider(recall_results=["wrong content"])
        result = await run_scenario(scenario, provider)
        
        assert result.passed is False
        assert result.failure_mode == "missing_expected"


class TestScenarioResult:
    def test_scenario_result_creation(self):
        result = ScenarioResult(
            name="test",
            category="cat",
            passed=True,
            latency_ms=50.0,
        )
        
        assert result.name == "test"
        assert result.category == "cat"
        assert result.passed is True
        assert result.failure_mode is None
        assert result.latency_ms == 50.0
    
    def test_scenario_result_failure(self):
        result = ScenarioResult(
            name="test",
            category="cat",
            passed=False,
            failure_mode="stale_retrieval",
            failure_description="Used outdated info",
        )
        
        assert result.passed is False
        assert result.failure_mode == "stale_retrieval"
