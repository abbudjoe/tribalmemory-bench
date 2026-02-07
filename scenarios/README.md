# Scenario-Based Evaluation

Task-completion benchmarks for conversational memory systems.

## Philosophy

Traditional benchmarks ask: "Did you retrieve matching text?"
We ask: "Did the agent complete the task correctly?"

This includes testing when memory should NOT be used.

## Scenario Types

### Positive (`positive/`)
Memory should be retrieved and used correctly.
- Multi-session recall
- User facts and preferences
- Entity relationships

### Negative (`negative/`)
Memory should NOT be retrieved or should be ignored.
- **Abstention**: No relevant memory exists, don't hallucinate
- **Stale rejection**: Memory exists but is outdated
- **Scope isolation**: Memory exists but for different context

### Temporal (`temporal/`)
Track changes over time correctly.
- Preference updates (was X, now Y)
- Knowledge corrections
- Time-sensitive facts

### Privacy (`privacy/`)
Memory isolation between contexts.
- Cross-user leakage
- Cross-session scope

### Degradation (`degradation/`)
Behavior at scale.
- 1k, 10k, 100k memories
- Latency, accuracy, cost curves

## Scenario Format

```yaml
name: "vegetarian_preference_update"
description: "User changed from meat-eater to vegetarian"
category: "temporal"

# Conversation history to ingest
conversations:
  - session: 1
    timestamp: "2024-01-15"
    messages:
      - role: user
        content: "I love a good steak"
      - role: assistant
        content: "Nice! Any favorite cuts?"
        
  - session: 2
    timestamp: "2024-06-20"  
    messages:
      - role: user
        content: "I've gone vegetarian now"
      - role: assistant
        content: "That's great! What prompted the change?"

# Task to evaluate
task:
  query: "Recommend a restaurant for dinner"
  
  # What SHOULD happen
  expected_behavior:
    should_retrieve: true
    should_use: "vegetarian preference (session 2)"
    should_ignore: "steak preference (session 1)"
    
  # Success criteria
  success:
    contains: ["vegetarian", "veggie", "plant-based"]
    not_contains: ["steak", "steakhouse", "meat"]

# Failure classification
failure_modes:
  - type: "stale_retrieval"
    description: "Retrieved outdated steak preference"
  - type: "temporal_confusion"
    description: "Mixed old and new preferences"
```

## Evaluation

```bash
# Run scenario suite
tribench scenarios positive/ --provider tribalmemory
tribench scenarios negative/ --provider tribalmemory

# Compare providers
tribench scenarios all/ --provider tribalmemory --provider mem0
```

## Sources

- `positive/`: Adapted from LongMemEval, ConvoMem
- `negative/abstention/`: Extracted from ConvoMem abstention category
- `negative/stale/`: Original (built from temporal patterns)
- `privacy/`: Original
- `degradation/`: Original
