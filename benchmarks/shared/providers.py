"""Memory provider abstractions for benchmarking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import asyncio
import httpx
import os
import uuid


@dataclass
class Memory:
    """A stored memory."""
    id: str
    content: str
    relevance: float = 0.0


class Provider(ABC):
    """Abstract base class for memory providers."""
    
    @abstractmethod
    async def store(self, content: str, context: Optional[str] = None) -> str:
        """Store a memory, return its ID."""
        pass
    
    @abstractmethod
    async def store_batch(self, memories: list[dict]) -> list[str]:
        """Store multiple memories, return their IDs."""
        pass
    
    @abstractmethod
    async def recall(self, query: str, limit: int = 10) -> list[Memory]:
        """Recall relevant memories for a query."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all memories (for test isolation)."""
        pass
    
    @abstractmethod
    async def stats(self) -> dict:
        """Get provider stats."""
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close any open connections."""
        pass


class TribalMemoryProvider(Provider):
    """TribalMemory provider implementation."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        instance: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url or os.environ.get(
            "TRIBALMEMORY_URL", "http://127.0.0.1:18790"
        )
        # Use UUID for isolation if no instance specified
        self.instance = instance or f"bench-{uuid.uuid4().hex[:8]}"
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)
    
    async def _request_with_retry(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> httpx.Response:
        """Make HTTP request with exponential backoff retry."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await getattr(self.client, method)(path, **kwargs)
                response.raise_for_status()
                return response
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    await asyncio.sleep(wait_time)
        raise last_error
    
    async def store(self, content: str, context: Optional[str] = None) -> str:
        """Store a memory in TribalMemory."""
        payload = {
            "content": content,
            "source_type": "auto_capture",
            "instance_id": self.instance,
        }
        if context:
            payload["context"] = context
        
        response = await self._request_with_retry("post", "/v1/remember", json=payload)
        data = response.json()
        return data.get("memory_id", "")
    
    async def store_batch(self, memories: list[dict]) -> list[str]:
        """
        Store multiple memories in TribalMemory.
        
        Falls back to sequential storage if batch endpoint not available.
        """
        # Try batch endpoint first
        try:
            payload = {
                "memories": [
                    {
                        "content": m.get("content", ""),
                        "source_type": "auto_capture",
                        "instance_id": self.instance,
                        "context": m.get("context"),
                    }
                    for m in memories
                ]
            }
            response = await self._request_with_retry("post", "/v1/remember/batch", json=payload)
            data = response.json()
            return data.get("memory_ids", [])
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Batch endpoint not available, fall back to sequential
                ids = []
                for m in memories:
                    mid = await self.store(m.get("content", ""), m.get("context"))
                    ids.append(mid)
                return ids
            raise
    
    async def recall(self, query: str, limit: int = 10) -> list[Memory]:
        """Recall memories from TribalMemory."""
        response = await self._request_with_retry(
            "post",
            "/v1/recall",
            json={
                "query": query,
                "limit": limit,
                "instance_id": self.instance,
            }
        )
        data = response.json()
        
        memories = []
        for result in data.get("results", []):
            mem = result.get("memory", {})
            memories.append(Memory(
                id=mem.get("id", ""),
                content=mem.get("content", ""),
                relevance=result.get("relevance", 0.0),
            ))
        return memories
    
    async def clear(self) -> None:
        """
        Clear all memories for this instance.
        
        Uses unique instance ID per run for isolation.
        If bulk delete is needed, use a fresh instance.
        """
        # Try clear endpoint if available
        try:
            await self._request_with_retry(
                "delete",
                f"/v1/memories",
                params={"instance_id": self.instance}
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise
            # Clear not available - we rely on unique instance IDs for isolation
    
    async def stats(self) -> dict:
        """Get TribalMemory stats."""
        response = await self._request_with_retry(
            "get",
            "/v1/stats",
            params={"instance_id": self.instance}
        )
        return response.json()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Future providers:
# class Mem0Provider(Provider): ...
# class ZepProvider(Provider): ...
# class SupermemoryProvider(Provider): ...
