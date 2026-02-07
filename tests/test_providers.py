"""Tests for providers module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from benchmarks.shared.providers import (
    Provider,
    TribalMemoryProvider,
    Memory,
)


class TestMemory:
    def test_memory_defaults(self):
        mem = Memory(id="123", content="test content")
        assert mem.id == "123"
        assert mem.content == "test content"
        assert mem.relevance == 0.0
    
    def test_memory_with_relevance(self):
        mem = Memory(id="123", content="test", relevance=0.95)
        assert mem.relevance == 0.95


class TestTribalMemoryProvider:
    def test_init_defaults(self):
        provider = TribalMemoryProvider()
        assert provider.base_url == "http://127.0.0.1:18790"
        assert provider.instance.startswith("bench-")
        assert provider.timeout == 120.0
        assert provider.max_retries == 3
    
    def test_init_custom(self):
        provider = TribalMemoryProvider(
            base_url="http://custom:8080",
            instance="test-instance",
            timeout=60.0,
            max_retries=5,
        )
        assert provider.base_url == "http://custom:8080"
        assert provider.instance == "test-instance"
        assert provider.timeout == 60.0
        assert provider.max_retries == 5
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager support."""
        async with TribalMemoryProvider() as provider:
            assert provider is not None
        # Client should be closed after exiting context


class TestTribalMemoryProviderRetry:
    @pytest.mark.asyncio
    async def test_no_retry_on_404(self):
        """404 errors should not be retried."""
        provider = TribalMemoryProvider()
        
        # Mock the client to return 404
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found",
            request=MagicMock(),
            response=mock_response,
        )
        
        provider.client.post = AsyncMock(return_value=mock_response)
        
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await provider._request_with_retry("post", "/test")
        
        # Should only be called once (no retries for 4xx)
        assert provider.client.post.call_count == 1
        await provider.close()
    
    @pytest.mark.asyncio
    async def test_no_retry_on_400(self):
        """400 errors should not be retried."""
        provider = TribalMemoryProvider()
        
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Request",
            request=MagicMock(),
            response=mock_response,
        )
        
        provider.client.get = AsyncMock(return_value=mock_response)
        
        with pytest.raises(httpx.HTTPStatusError):
            await provider._request_with_retry("get", "/test")
        
        assert provider.client.get.call_count == 1
        await provider.close()


class TestTribalMemoryProviderStore:
    @pytest.mark.asyncio
    async def test_store_success(self):
        """Test successful memory storage."""
        provider = TribalMemoryProvider()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"memory_id": "abc123"}
        
        provider.client.post = AsyncMock(return_value=mock_response)
        
        memory_id = await provider.store("test content", context="test context")
        
        assert memory_id == "abc123"
        provider.client.post.assert_called_once()
        call_kwargs = provider.client.post.call_args[1]
        assert call_kwargs["json"]["content"] == "test content"
        assert call_kwargs["json"]["context"] == "test context"
        await provider.close()


class TestTribalMemoryProviderRecall:
    @pytest.mark.asyncio
    async def test_recall_success(self):
        """Test successful memory recall."""
        provider = TribalMemoryProvider()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "memory": {"id": "1", "content": "first"},
                    "relevance": 0.9,
                },
                {
                    "memory": {"id": "2", "content": "second"},
                    "relevance": 0.7,
                },
            ]
        }
        
        provider.client.post = AsyncMock(return_value=mock_response)
        
        memories = await provider.recall("test query", limit=5)
        
        assert len(memories) == 2
        assert memories[0].id == "1"
        assert memories[0].content == "first"
        assert memories[0].relevance == 0.9
        await provider.close()
    
    @pytest.mark.asyncio
    async def test_recall_empty(self):
        """Test recall with no results."""
        provider = TribalMemoryProvider()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"results": []}
        
        provider.client.post = AsyncMock(return_value=mock_response)
        
        memories = await provider.recall("no match")
        
        assert memories == []
        await provider.close()


class TestTribalMemoryProviderStoreBatch:
    @pytest.mark.asyncio
    async def test_store_batch_success(self):
        """Test successful batch storage."""
        provider = TribalMemoryProvider()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"memory_ids": ["1", "2", "3"]}
        
        provider.client.post = AsyncMock(return_value=mock_response)
        
        memories = [
            {"content": "mem1"},
            {"content": "mem2"},
            {"content": "mem3"},
        ]
        
        ids = await provider.store_batch(memories)
        
        assert ids == ["1", "2", "3"]
        await provider.close()
    
    @pytest.mark.asyncio
    async def test_store_batch_fallback_on_404(self):
        """Test fallback to sequential on 404."""
        provider = TribalMemoryProvider()
        
        # First call returns 404 (batch not supported)
        mock_404 = MagicMock()
        mock_404.status_code = 404
        mock_404.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_404
        )
        
        # Sequential calls succeed
        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.raise_for_status = MagicMock()
        mock_success.json.return_value = {"memory_id": "seq"}
        
        call_count = 0
        async def mock_post(path, **kwargs):
            nonlocal call_count
            call_count += 1
            if path == "/v1/remember/batch":
                return mock_404
            return mock_success
        
        provider.client.post = mock_post
        
        memories = [{"content": "mem1"}, {"content": "mem2"}]
        ids = await provider.store_batch(memories)
        
        # Should have called batch once, then individual twice
        assert call_count == 3
        assert ids == ["seq", "seq"]
        await provider.close()
