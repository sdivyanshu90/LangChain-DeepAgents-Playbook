# 05 — Context Pruning Strategies

> **Previous:** [04 → Multi-User Isolation](04-multi-user-isolation.md) | **Next:** [Module 2.4 Complete — Proceed to DeepAgents Track](../../../deepagents/README.md)

---

## Real-World Analogy

A project manager's notebook has 200 pages of meeting notes.
Before a new meeting, they don't re-read all 200 pages.
They read the last page of notes plus a one-page summary of key decisions.

Context pruning is the same act: keep recent turns in full,
compress old turns into a summary, and discard small talk that has no lasting value.

---

## Why Context Window Limits Are a Real Problem

```
Turn 1: user asks about vacation policy     → 200 tokens
Turn 2: AI answers vacation policy          → 300 tokens
...
Turn 50: user asks about remote work        → 200 tokens
Turn 51: AI answer + full 50-turn history   → 10,000 tokens

gpt-4o-mini context window: 128,000 tokens
Monthly cost with 1000 daily users, 50-turn conversations: significant
Latency penalty: prompt processing time grows with token count

Also: old conversation turns often add noise, not signal.
Turn 3's discussion of vacation policy is irrelevant to turn 51's remote work question.
```

---

## The Token Budget Model

Define your budget before implementing pruning:

```python
MAX_HISTORY_TOKENS   = 2000   # how many tokens to allow for message history
SUMMARY_TARGET_WORDS = 150    # target length of a compressed summary
RECENT_TURNS_TO_KEEP = 4      # how many full recent turns to always keep (each = 2 messages)

# Example token breakdown:
# System prompt:    ~300 tokens (fixed)
# State context:    ~200 tokens (varies)
# History:          2000 tokens (budgeted)
# Current message:  ~100 tokens
# ─────────────────────────────────────
# Total:            ~2600 tokens → leaves ~4000 for model output (gpt-4o-mini)
```

---

## Strategy 1 — Window Buffer (Last N Turns)

Simplest approach: keep only the last N turns, discard everything older.

```python
from langchain_core.messages import BaseMessage

def window_prune(
    messages: list[BaseMessage],
    keep_turns: int = 4,
) -> list[BaseMessage]:
    """
    Keep only the last `keep_turns` turns (each turn = 1 human + 1 AI message).
    Oldest messages are silently dropped.
    """
    keep_messages = keep_turns * 2    # each turn is 2 messages
    if len(messages) <= keep_messages:
        return messages               # history fits in window; no pruning needed
    return messages[-keep_messages:]  # keep the most recent messages

# Usage: apply before building the prompt
raw_messages = history.messages
pruned_messages = window_prune(raw_messages, keep_turns=4)
# Passes pruned_messages to the chain instead of the full history
```

**When to use:** When the most recent context is sufficient and older turns are genuinely not needed.
**Trade-off:** Information in older turns (e.g., the user's name stated in turn 1) is lost.

---

## Strategy 2 — Summary + Recent Turns

Keep full recent turns. Compress older turns into a one-paragraph summary.
The summary is prepended to the recent turns as a system message.

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

summary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are summarising a support conversation to keep only the essential information.\n"
        "Focus on: decisions made, facts established, user preferences stated.\n"
        "Omit: pleasantries, repeated clarifications, unsuccessful attempts.\n"
        f"Target: under {SUMMARY_TARGET_WORDS} words."
    )),
    ("human", "Summarise this conversation:\n\n{conversation}"),
])

def format_conversation_for_summary(messages: list[BaseMessage]) -> str:
    """Format a list of messages as readable text for the summariser."""
    lines = []
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)

def summarise_old_turns(old_messages: list[BaseMessage]) -> str:
    """Use the LLM to compress old turns into a short paragraph."""
    conversation_text = format_conversation_for_summary(old_messages)
    result = summary_llm.invoke(
        SUMMARY_PROMPT.format_messages(conversation=conversation_text)
    )
    return result.content

def summary_and_recent_prune(
    messages: list[BaseMessage],
    keep_turns: int = 4,
) -> list[BaseMessage]:
    """
    Prune a message history to summary + recent turns.

    Structure of returned list:
      [SystemMessage("Summary of earlier conversation: ..."), recent_messages...]

    Why SystemMessage for the summary:
      - SystemMessages appear before history in the prompt
      - They don't look like a "previous turn"; they look like context
      - The model treats them as persistent background knowledge
    """
    keep_messages = keep_turns * 2

    if len(messages) <= keep_messages:
        return messages   # fits without pruning; no summary needed

    old_messages    = messages[:-keep_messages]
    recent_messages = messages[-keep_messages:]

    summary_text = summarise_old_turns(old_messages)
    summary_msg  = SystemMessage(
        content=f"Summary of earlier conversation:\n{summary_text}"
    )

    return [summary_msg] + recent_messages

# Usage
raw_messages   = history.messages
pruned_history = summary_and_recent_prune(raw_messages, keep_turns=4)
# Returns: [SystemMessage(summary), last 4 turns (8 messages)]
```

---

## Strategy 3 — Selective Retention (Keep Decisions, Drop Small Talk)

Use an LLM classifier to decide which messages are "important" before summarising:

```python
from pydantic import BaseModel, Field

class MessageImportance(BaseModel):
    """Classification of a single message's importance for retention."""
    keep:   bool = Field(description="True if this message should be retained verbatim")
    reason: str  = Field(description="Brief reason for the keep/drop decision")

importance_classifier = summary_llm.with_structured_output(MessageImportance)

CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Decide if this message should be retained verbatim in a long conversation summary.\n\n"
        "KEEP if the message contains:\n"
        "  - A decision made\n"
        "  - A fact stated by the user (name, preference, account detail)\n"
        "  - An error or failure that must be remembered\n"
        "  - A confirmed agreement\n\n"
        "DROP if the message contains only:\n"
        "  - Greetings, pleasantries ('thanks', 'sure', 'got it')\n"
        "  - Repeated clarification questions\n"
        "  - Unsuccessful attempts that were superseded\n"
    )),
    ("human", "MESSAGE: {message}"),
])

def selectively_retain(
    messages: list[BaseMessage],
    keep_turns: int = 4,
) -> list[BaseMessage]:
    """
    Classify each old message and keep only the important ones.
    Always keep the most recent `keep_turns` turns in full.
    """
    keep_messages = keep_turns * 2

    if len(messages) <= keep_messages:
        return messages

    old_messages    = messages[:-keep_messages]
    recent_messages = messages[-keep_messages:]

    retained = []
    for msg in old_messages:
        result = importance_classifier.invoke(
            CLASSIFY_PROMPT.format_messages(message=msg.content)
        )
        if result.keep:
            retained.append(msg)

    return retained + recent_messages
```

---

## Implementing TruncatingSummaryMemory Pattern

A production-grade class that combines all three strategies adaptively:

```python
from langchain_core.messages import BaseMessage
from langchain_community.chat_message_histories import InMemoryChatMessageHistory
import tiktoken

def count_tokens(messages: list[BaseMessage], model: str = "gpt-4o-mini") -> int:
    """Count tokens for a list of messages."""
    enc = tiktoken.encoding_for_model(model)
    total = 0
    for msg in messages:
        total += 4   # per-message overhead
        total += len(enc.encode(str(msg.content)))
    return total

class TruncatingSummaryHistory:
    """
    A drop-in replacement for InMemoryChatMessageHistory that automatically
    prunes history when token budget is exceeded.

    Strategy:
      1. If under budget → return as-is.
      2. If over budget but recent turns fit → keep recent turns + window prune.
      3. If still over budget → compress old turns to a summary.
    """

    def __init__(
        self,
        max_tokens: int = 2000,
        keep_turns: int = 4,
    ) -> None:
        self._inner    = InMemoryChatMessageHistory()
        self.max_tokens = max_tokens
        self.keep_turns = keep_turns

    def add_user_message(self, content: str) -> None:
        self._inner.add_user_message(content)

    def add_ai_message(self, content: str) -> None:
        self._inner.add_ai_message(content)

    @property
    def messages(self) -> list[BaseMessage]:
        """Return pruned messages, never exceeding the token budget."""
        all_msgs = self._inner.messages

        if count_tokens(all_msgs) <= self.max_tokens:
            return all_msgs   # fits; no pruning needed

        pruned = summary_and_recent_prune(all_msgs, keep_turns=self.keep_turns)

        if count_tokens(pruned) <= self.max_tokens:
            return pruned

        # Even summary + recent exceeds budget; hard window truncation
        return window_prune(pruned, keep_turns=self.keep_turns)

    def clear(self) -> None:
        self._inner.clear()
```

---

## Common Pitfalls

| Pitfall                                          | What goes wrong                                                       | Fix                                                                                  |
| ------------------------------------------------ | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Window pruning drops the user's name from turn 1 | Model stops using the user's name mid-conversation                    | Move profile facts to `UserState`; don't rely on history for permanent facts         |
| Summarising every turn                           | Many LLM calls; high cost and latency                                 | Only summarise when token budget is exceeded (check before summarising)              |
| Not persisting the summary                       | Summary is lost on process restart; next call re-reads full history   | Store the summary as a `SystemMessage` in the history backend alongside recent turns |
| Using selective retention on short conversations | Classifying each message when history is short adds unnecessary cost  | Only use selective retention after history exceeds 2× the keep_turns threshold       |
| Not accounting for summary token cost            | The summary itself might be 300 tokens; total may still exceed budget | Count tokens AFTER pruning; apply hard window if still over budget                   |

---

## Mini Summary

- Context grows with every turn; without pruning, token costs and latency increase without bound.
- Window buffer: keep last N turns. Simple; loses old facts. Use when old context is not needed.
- Summary + recent: compress old turns into a paragraph, keep full recent turns. Best balance.
- Selective retention: classify each message; keep decisions and facts, discard pleasantries.
- `TruncatingSummaryHistory` combines all three strategies and activates pruning only when needed.
- Profile facts and decisions should live in `UserState`, not in conversation history — they must survive pruning.
