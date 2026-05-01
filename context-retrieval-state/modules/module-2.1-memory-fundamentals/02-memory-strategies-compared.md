# 02 — Memory Strategies Compared

> **Previous:** [01 → Why LLMs Are Stateless](01-why-llms-are-stateless.md) | **Next:** [03 → RunnableWithMessageHistory](03-runnable-with-message-history.md)

---

## Real-World Analogy

A journalist covering a long conference has three options for their notes:

1. **Full transcript**: write down everything. Runs out of notebook pages eventually.
2. **Recent pages only**: keep only the last 10 pages. The earliest context is lost.
3. **Summary + recent**: distill early sessions into a paragraph; keep recent sessions in full.

Option 3 retains the most value per page.
In LLM memory design, the same three strategies exist.

---

## Why Strategy Matters

The wrong strategy causes one of two failures:

```
Failure Mode A: Context overflow
─────────────────────────────────
  Full buffer grows without limit.
  Request eventually exceeds context window.
  Error: "This model's maximum context length is exceeded."
  Result: application crashes on long sessions.

Failure Mode B: Amnesia
────────────────────────
  Window buffer drops early turns.
  User: "Let's go back to what I said about the project requirements."
  App: (those messages were dropped 50 turns ago — not in context)
  Result: model gives a hallucinated or irrelevant response.
```

Neither is acceptable in production.
The right strategy depends on session length, content value, and cost constraints.

---

## Strategy 1 — Full Buffer Memory

**Keep everything.** Every message from the start of the session.

```
Turn 1:  [H1, A1]
Turn 5:  [H1, A1, H2, A2, H3, A3, H4, A4, H5]
Turn 50: [H1, A1, H2, A2, ... H50]  ← 50 turns
```

### Token Math (Full Buffer)

```
Assumptions:
  Average turn = 200 tokens (100 user + 100 AI)
  Context window = 128,000 tokens (gpt-4o-mini)
  System message = 50 tokens

  Safe history budget = 128,000 - 50 - 1,000 (response buffer) = 126,950 tokens
  Max turns before overflow = 126,950 / 200 ≈ 634 turns

  At 10 turns/day: safe for 63 days
  At 100 turns/day: overflows in 6 days
```

### Implementation

```python
from langchain_core.chat_history import InMemoryChatMessageHistory

# Full buffer is the default — just append messages without trimming
history = InMemoryChatMessageHistory()

history.add_user_message("What is Python?")
history.add_ai_message("Python is a programming language.")
history.add_user_message("What version is latest?")
history.add_ai_message("Python 3.12.")

# history.messages returns ALL stored messages
print(len(history.messages))   # 4 — grows without bound
```

**When to use:** Short-lived sessions (< 50 turns); testing and development.
**When NOT to use:** Multi-turn support conversations; long research sessions.

---

## Strategy 2 — Window Buffer Memory

**Keep only the last N turns.** When the buffer exceeds N, drop the oldest.

```
Window size = 3 turns (for illustration):

After turn 1: [H1, A1]
After turn 2: [H1, A1, H2, A2]
After turn 3: [H1, A1, H2, A2, H3, A3]
After turn 4: [       H2, A2, H3, A3, H4, A4]  ← H1/A1 dropped
After turn 5: [              H3, A3, H4, A4, H5, A5]  ← H2/A2 dropped
```

### Token Math (Window Buffer)

```
Window size = 20 turns
  Memory tokens = 20 × 200 = 4,000 tokens — constant regardless of session length.
  Context budget for response: 128,000 - 4,000 - 50 = 123,950 tokens.
  Session can be arbitrarily long without overflow.
```

### Implementation

```python
from langchain_core.messages import trim_messages
from langchain_core.chat_history import InMemoryChatMessageHistory

def get_windowed_messages(
    history: InMemoryChatMessageHistory,
    max_tokens: int = 4000,
) -> list:
    """Return recent messages trimmed to fit within max_tokens."""
    return trim_messages(
        history.messages,
        max_tokens=max_tokens,
        token_counter=len,            # use len(messages) as a fast proxy for testing
        # In production: use a real token counter
        # token_counter=ChatOpenAI(model="gpt-4o-mini").get_num_tokens_from_messages
        strategy="last",              # keep the LAST N messages (drop oldest)
        include_system=True,          # never drop the system message
        allow_partial=False,          # never split a turn mid-message
    )
```

### The Amnesia Trade-Off

```
Session: product requirements discussion

Turn 1:  User: "We need user authentication with OAuth."
Turn 2:  AI: "Got it. OAuth 2.0 implementation notes..."
...
Turn 25: User: "Let's revisit the auth requirements from the start."
         AI: ???

With window = 20 turns:
  Turns 1-5 have been dropped.
  The model cannot recall the OAuth discussion.
  It will say: "I don't have that in my current context."
  Or worse: it will hallucinate "requirements" it never saw.
```

**When to use:** Customer support chats; form-filling workflows; any session where
recent context matters much more than historical context.

---

## Strategy 3 — Summary Memory

**Summarise old turns; keep recent turns in full.**
This is the most sophisticated strategy and the one most suitable for long sessions.

```
Window size = 10 recent turns  |  Summary of everything older

Turn 30:
  [SystemMessage("You are a helpful assistant.")]
  [SystemMessage("Summary of turns 1-20: The user is building an e-commerce API.
    They've decided on FastAPI, PostgreSQL, and Stripe for payments. Key constraints:
    GDPR compliance required; deployment target is AWS ECS.")]
  [H21, A21, H22, A22, ... H30]   ← last 10 turns in full
```

### Token Math (Summary Memory)

```
Without summary:  30 turns × 200 tokens = 6,000 tokens
With summary:     summary (300 tokens) + 10 recent turns (2,000 tokens) = 2,300 tokens
                  Savings: 3,700 tokens (62%)

At 100 turns: Without summary = 20,000 tokens
              With summary    = 300 + 2,000 = 2,300 tokens
              Savings: 89%
```

### Implementation

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def summarise_history(
    messages: list,
    existing_summary: str = "",
) -> str:
    """
    Summarise old conversation turns into a compact paragraph.
    If a prior summary exists, extend it rather than re-summarising from scratch.
    """
    # Format messages for the summary prompt
    conversation_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in messages
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a conversation summariser. "
            "Produce a concise, factual summary of the key information and decisions "
            "from the conversation. Preserve specific facts, numbers, and decisions. "
            "Discard small talk and filler."
        )),
        ("human", (
            "{existing_summary_section}"
            "Conversation to summarise:\n{conversation}"
        )),
    ])

    existing_section = (
        f"Prior summary:\n{existing_summary}\n\nNew conversation to add:\n"
        if existing_summary else ""
    )

    result = llm.invoke(
        prompt.format_messages(
            existing_summary_section=existing_section,
            conversation=conversation_text,
        )
    )
    return result.content

def get_messages_with_summary(
    all_messages: list,
    recent_window: int = 10,
    existing_summary: str = "",
) -> tuple[list, str]:
    """
    Split messages into old and recent.
    Summarise old messages; return recent messages + updated summary.
    Returns: (message_list_to_send, new_summary_string)
    """
    if len(all_messages) <= recent_window:
        # Not enough messages to warrant summarisation
        return all_messages, existing_summary

    old_messages  = all_messages[:-recent_window]   # everything before the recent window
    recent_messages = all_messages[-recent_window:] # the last N messages

    # Summarise the old messages
    new_summary = summarise_history(old_messages, existing_summary)

    # Build the message list: system summary + recent turns
    return [
        SystemMessage(content=f"Summary of earlier conversation:\n{new_summary}"),
        *recent_messages,
    ], new_summary
```

---

## Strategy Comparison Table

| Dimension                     | Full Buffer                 | Window Buffer                | Summary Memory                           |
| ----------------------------- | --------------------------- | ---------------------------- | ---------------------------------------- |
| **Recall depth**              | All turns (until overflow)  | Last N turns only            | All turns (via summary) + recent in full |
| **Token cost**                | Grows linearly              | Constant                     | Near-constant                            |
| **Overflow risk**             | Yes — crashes long sessions | No                           | No                                       |
| **Implementation complexity** | Very low                    | Low                          | Medium                                   |
| **Information loss**          | None (until overflow)       | Loses early context entirely | Compressed context; detail lost          |
| **Best for**                  | Dev/testing, short sessions | Support chat, form wizards   | Research, consulting, long workflows     |

---

## Choosing a Strategy in Practice

```
Is the session guaranteed to be short (< 30 turns)?
  → Yes: Full Buffer. Simple, no information loss.
  → No: continue...

Is only the recent context relevant (e.g., support chat)?
  → Yes: Window Buffer. Simple, predictable cost.
  → No: continue...

Does the user need the model to recall information from 50+ turns ago?
  → Yes: Summary Memory. More work to implement, but the only option that scales.
```

---

## Common Pitfalls

| Pitfall                                     | What goes wrong                                      | Fix                                       |
| ------------------------------------------- | ---------------------------------------------------- | ----------------------------------------- |
| Using full buffer in production             | Context overflow on long sessions; silent truncation | Implement window or summary strategy      |
| Window too small (< 5 turns)                | Model loses context mid-conversation                 | Use at minimum 10-15 turns                |
| Summarising every turn                      | Summary model costs add up; summaries accumulate     | Summarise in batches (every 20+ turns)    |
| Including the summary in the stored history | Summary + history = double storage                   | Store summary and recent turns separately |
| Using `len(messages)` as token counter      | Imprecise; messages vary widely in length            | Use a proper tokenizer for production     |

---

## Mini Summary

- Full Buffer keeps everything: simple but overflows on long sessions.
- Window Buffer keeps only recent turns: constant cost but permanent amnesia for old context.
- Summary Memory compresses old turns into a paragraph: scales indefinitely with minimal loss.
- Choose your strategy based on session length and the value of historical context.
- Token math matters: model the worst-case session length before choosing a strategy.
