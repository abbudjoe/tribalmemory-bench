"""Memory provider abstractions for benchmarking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import httpx
import os


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


class TribalMemoryProvider(Provider):
    """TribalMemory provider implementation."""
    
    def __init__(self, base_url: Optional[str] = None, instance: str = "benchmark"):
        self.base_url = base_url or os.environ.get(
            "TRIBALMEMORY_URL", "http://127.0.0.1:18790"
        )
        self.instance = instance
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)
    
    async def store(self, content: str, context: Optional[str] = None) -> str:
        """Store a memory in TribalMemory."""
        payload = {
            "content": content,
            "source_type": "auto_capture",
            "instance_id": self.instance,
        }
        if context:
            payload["context"] = context
        
        response = await self.client.post("/v1/remember", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("memory_id", "")
    
    async def recall(self, query: str, limit: int = 10) -> list[Memory]:
        """Recall memories from TribalMemory."""
        response = await self.client.post(
            "/v1/recall",
            json={
                "query": query,
                "limit": limit,
                "instance_id": self.instance,
            }
        )
        response.raise_for_status()
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
        """Clear all memories for this instance."""
        # TribalMemory doesn't have a bulk delete, but we can use a fresh instance
        # For benchmarks, we use unique instance IDs per run
        pass
    
    async def stats(self) -> dict:
        """Get TribalMemory stats."""
        response = await self.client.get(
            "/v1/stats",
            params={"instance_id": self.instance}
        )
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Future providers:
# class Mem0Provider(Provider): ...
# class ZepProvider(Provider): ...
# class SupermemoryProvider(Provider): ...
