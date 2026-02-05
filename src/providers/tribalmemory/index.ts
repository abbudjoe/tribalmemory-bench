/**
 * TribalMemory provider for MemoryBench.
 *
 * Connects to a running TribalMemory HTTP server (tribalmemory serve).
 * Designed for local-first, cross-agent memory with graph-enriched search.
 *
 * Environment:
 *   TRIBALMEMORY_BASE_URL - Server URL (default: http://localhost:8765)
 *   TRIBALMEMORY_API_KEY  - Optional API key (reserved for future auth)
 */

import type {
  Provider,
  ProviderConfig,
  IngestOptions,
  IngestResult,
  SearchOptions,
  IndexingProgressCallback,
} from "../../types/provider"
import type { UnifiedSession } from "../../types/unified"
import { logger } from "../../utils/logger"

interface TribalMemoryStoreResponse {
  success: boolean
  memory_id?: string
  duplicate_of?: string
  error?: string
}

interface TribalMemoryRecallResult {
  memory: {
    id: string
    content: string
    source_instance: string
    source_type: string
    created_at: string
    updated_at: string
    tags: string[]
    context?: Record<string, unknown>
    confidence: number
  }
  similarity_score: number
  retrieval_time_ms: number
  retrieval_method?: string
}

interface TribalMemoryRecallResponse {
  results: TribalMemoryRecallResult[]
  query: string
  total_time_ms: number
  error?: string
}

export class TribalMemoryProvider implements Provider {
  name = "tribalmemory"
  private baseUrl: string = "http://localhost:8765"
  private apiKey: string = ""
  // Track memory IDs per container for cleanup
  private containerMemories: Map<string, string[]> = new Map()

  async initialize(config: ProviderConfig): Promise<void> {
    this.baseUrl = config.baseUrl || "http://localhost:8765"
    this.apiKey = config.apiKey || ""

    // Health check
    try {
      const response = await fetch(`${this.baseUrl}/v1/health`)
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`)
      }
      const health = await response.json()
      logger.info(
        `Initialized TribalMemory provider at ${this.baseUrl} ` +
          `(instance: ${health.instance_id})`
      )
    } catch (error) {
      throw new Error(
        `Failed to connect to TribalMemory at ${this.baseUrl}: ${error}`
      )
    }
  }

  async ingest(
    sessions: UnifiedSession[],
    options: IngestOptions
  ): Promise<IngestResult> {
    const memoryIds: string[] = []
    const containerTag = options.containerTag

    for (const session of sessions) {
      // Convert each message in the session to a memory
      for (const message of session.messages) {
        // Build context with session metadata (serialized as JSON string)
        const contextObj: Record<string, unknown> = {
          sessionId: session.sessionId,
          role: message.role,
          speaker: message.speaker,
          containerTag,
          ...session.metadata,
          ...options.metadata,
        }

        if (message.timestamp) {
          contextObj.timestamp = message.timestamp
        }

        const body = {
          content: message.content,
          source_type: "auto_capture",  // TribalMemory SourceType enum value
          context: JSON.stringify(contextObj),  // API expects string, not object
          tags: [containerTag, `session:${session.sessionId}`, message.role],
        }

        try {
          const response = await fetch(`${this.baseUrl}/v1/remember`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              ...(this.apiKey && { Authorization: `Bearer ${this.apiKey}` }),
            },
            body: JSON.stringify(body),
          })

          if (!response.ok) {
            logger.warn(
              `Failed to store message: ${response.status} ${response.statusText}`
            )
            continue
          }

          const result: TribalMemoryStoreResponse = await response.json()

          if (result.success && result.memory_id) {
            memoryIds.push(result.memory_id)
          } else if (result.duplicate_of) {
            // Dedup detected, track the original
            memoryIds.push(result.duplicate_of)
          }
        } catch (error) {
          logger.warn(`Error storing message: ${error}`)
        }
      }
    }

    // Track memories for this container (for cleanup)
    const existing = this.containerMemories.get(containerTag) || []
    this.containerMemories.set(containerTag, [...existing, ...memoryIds])

    logger.info(
      `Ingested ${sessions.length} sessions (${memoryIds.length} memories) ` +
        `for container: ${containerTag}`
    )

    return { documentIds: memoryIds }
  }

  async awaitIndexing(
    result: IngestResult,
    _containerTag: string,
    onProgress?: IndexingProgressCallback
  ): Promise<void> {
    // TribalMemory indexes synchronously on store, no async indexing needed
    const total = result.documentIds.length
    onProgress?.({
      completedIds: result.documentIds,
      failedIds: [],
      total,
    })
  }

  async search(query: string, options: SearchOptions): Promise<unknown[]> {
    const body = {
      query,
      limit: options.limit || 30,
      min_relevance: options.threshold || 0.3,
      tags: [options.containerTag],
    }

    try {
      const response = await fetch(`${this.baseUrl}/v1/recall`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(this.apiKey && { Authorization: `Bearer ${this.apiKey}` }),
        },
        body: JSON.stringify(body),
      })

      if (!response.ok) {
        throw new Error(`Search failed: ${response.status}`)
      }

      const data: TribalMemoryRecallResponse = await response.json()

      if (data.error) {
        logger.warn(`Search returned error: ${data.error}`)
        return []
      }

      // Return results in a format the benchmark can use
      return data.results.map((r) => {
        // Parse context if it's a JSON string (we store it stringified)
        let parsedContext: Record<string, unknown> = {}
        if (r.memory.context) {
          try {
            parsedContext = typeof r.memory.context === 'string' 
              ? JSON.parse(r.memory.context) 
              : r.memory.context
          } catch {
            // If parsing fails, use as-is or empty
            parsedContext = {}
          }
        }
        
        return {
          id: r.memory.id,
          content: r.memory.content,
          score: r.similarity_score,
          metadata: {
            ...parsedContext,
            tags: r.memory.tags,
            created_at: r.memory.created_at,
            retrieval_method: r.retrieval_method,
          },
        }
      })
    } catch (error) {
      logger.error(`Search error: ${error}`)
      return []
    }
  }

  async clear(containerTag: string): Promise<void> {
    const memoryIds = this.containerMemories.get(containerTag) || []

    let deleted = 0
    for (const memoryId of memoryIds) {
      try {
        const response = await fetch(
          `${this.baseUrl}/v1/forget/${memoryId}`,
          {
            method: "DELETE",
            headers: {
              ...(this.apiKey && { Authorization: `Bearer ${this.apiKey}` }),
            },
          }
        )

        if (response.ok) {
          deleted++
        }
      } catch {
        // Ignore errors on cleanup
      }
    }

    this.containerMemories.delete(containerTag)
    logger.info(`Cleared ${deleted}/${memoryIds.length} memories for: ${containerTag}`)
  }
}

export default TribalMemoryProvider
