# 04 — Retry and Fallback for Parse Failures

> **Previous:** [03 → .with_structured_output()](03-with-structured-output.md) | **Next:** [05 → Schema Design Patterns](05-schema-design-patterns.md)

---

## Real-World Analogy

Imagine a data entry operator who transcribes paper forms into a database.
When a form is unreadable, a well-designed system does not silently skip it or crash.
It routes the form to a second operator with a note: "Field 3 was illegible — please re-read."
The second operator looks at the original form and the note, then tries again.

That is exactly the retry-with-context pattern for structured output failures.

---

## Why Parse Failures Happen

Even with `.with_structured_output()` and tool calling,
parse failures occur in three situations:

```
Situation 1: Model produces invalid JSON
─────────────────────────────────────────
  Rare with tool calling, but happens with:
  - Local models (Ollama) using json_mode
  - Very long outputs that get truncated
  - Models that haven't been trained on tool calling

Situation 2: Pydantic validation rejects the output
─────────────────────────────────────────────────────
  The JSON is valid but violates schema constraints:
  - score = 15 when ge=1, le=10
  - sentiment = "strongly positive" when enum is POSITIVE/NEGATIVE/NEUTRAL
  - Required field missing from the response

Situation 3: Intermittent API failures
───────────────────────────────────────
  - Network timeouts
  - Rate limiting (429)
  - Model returns empty content
```

You need different strategies for each.

---

## The `OutputParserException`

When parsing fails, LangChain raises `OutputParserException`:

```python
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class ProductInfo(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD", gt=0)
    in_stock: bool = Field(description="Whether the product is currently available")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(ProductInfo)

try:
    result = structured_llm.invoke([
        HumanMessage(content="Tell me about a laptop")
    ])
except OutputParserException as e:
    # e.llm_output contains the raw string that failed to parse
    print(f"Parse failed: {e}")
    print(f"Raw output was: {e.llm_output}")
```

But catching exceptions is not a retry strategy — it is just error handling.
You need an active recovery loop.

---

## The Basic Retry Loop Pattern

The simplest reliable pattern: retry up to N times, stop on success:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError
import logging

logger = logging.getLogger(__name__)

class ExtractedData(BaseModel):
    company_name: str = Field(description="Name of the company")
    founding_year: int = Field(
        description="Year the company was founded",
        ge=1800, le=2100,
    )
    employee_count_estimate: str = Field(
        description="Rough size: 'startup' (<50), 'small' (50-500), "
                    "'medium' (500-5000), 'large' (>5000)"
    )

def extract_with_retry(
    text: str,
    llm: ChatOpenAI,
    max_attempts: int = 3,
) -> ExtractedData:
    """
    Extract structured data with up to max_attempts retries.
    Each failed attempt logs the error before retrying.
    """
    structured_llm = llm.with_structured_output(ExtractedData, include_raw=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract company information from the provided text."),
        ("human", "{text}"),
    ])
    chain = prompt | structured_llm

    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            result = chain.invoke({"text": text})

            if result["parsing_error"] is not None:
                # include_raw=True surfaces parse errors without raising
                raise result["parsing_error"]

            if result["parsed"] is None:
                raise OutputParserException("Parsed result was None")

            return result["parsed"]   # success — return immediately

        except (OutputParserException, ValidationError) as e:
            last_error = e
            logger.warning(
                f"Attempt {attempt}/{max_attempts} failed: {e}. "
                f"Raw output: {result.get('raw', {})}"
            )
            # Loop continues to next attempt automatically

    # All attempts exhausted
    raise OutputParserException(
        f"Failed to parse after {max_attempts} attempts. "
        f"Last error: {last_error}"
    )
```

---

## Feedback-as-Context: The Smarter Retry

Simply retrying the same prompt often produces the same failure.
The effective pattern is to include the previous failure as context
so the model understands what went wrong:

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def extract_with_feedback_retry(
    text: str,
    llm: ChatOpenAI,
    max_attempts: int = 3,
) -> ExtractedData:
    """
    Retry with the error fed back as context so the model can self-correct.
    Each retry includes:
    1. The original question
    2. The model's previous (failed) response
    3. A description of what was wrong
    """
    structured_llm = llm.with_structured_output(ExtractedData, include_raw=True)

    # Build the message history manually for feedback injection
    messages = [
        SystemMessage(content="Extract company information from the provided text."),
        HumanMessage(content=text),
    ]

    for attempt in range(1, max_attempts + 1):
        result = structured_llm.invoke(messages)

        if result["parsing_error"] is None and result["parsed"] is not None:
            return result["parsed"]   # success

        # Failure — construct error feedback
        error_msg = str(result["parsing_error"] or "Unknown parse error")
        raw_content = ""
        if result.get("raw"):
            # Extract whatever the model actually returned
            raw_content = str(result["raw"].tool_calls or result["raw"].content)

        logger.warning(f"Attempt {attempt} failed: {error_msg}")

        if attempt < max_attempts:
            # Append the failed attempt and the correction instruction
            messages.append(
                AIMessage(content=raw_content)   # what the model tried
            )
            messages.append(
                HumanMessage(
                    content=(
                        f"That response was invalid. Error: {error_msg}\n\n"
                        f"Please try again. Make sure all required fields are present "
                        f"and values conform to the schema constraints."
                    )
                )
            )

    raise OutputParserException(
        f"Could not extract valid data after {max_attempts} attempts."
    )
```

### Why Feedback Works

```
Attempt 1:
  Input:    "Tell me about Apple Inc."
  Output:   {"company_name": "Apple", "founding_year": "1976", ...}
                                                    ↑ string, not int
  Error:    "founding_year: value is not a valid integer"

Attempt 2 (with feedback):
  Input:    [original messages] +
            AIMessage: {"company_name": "Apple", "founding_year": "1976"}
            HumanMessage: "founding_year must be an integer, not a string"
  Output:   {"company_name": "Apple", "founding_year": 1976, ...}
                                                    ↑ correct
```

The model sees its mistake and the specific error description.
Self-correction rate is typically 85-95% on attempt 2.

---

## Using `OutputFixingParser` from LangChain

LangChain provides a built-in wrapper that implements the feedback pattern:

```python
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain_openai import ChatOpenAI

# Note: OutputFixingParser works with PydanticOutputParser, not structured_output
# It's best for json_mode or prompt-based extraction (not tool calling)

base_parser = PydanticOutputParser(pydantic_object=ExtractedData)

fixing_parser = OutputFixingParser.from_llm(
    llm=llm,              # the model that will attempt the fix
    parser=base_parser,   # the original parser that defines the schema
    max_retries=3,        # number of fix attempts before raising
)

# Use in a chain like any other parser
chain = prompt | llm | fixing_parser
result = chain.invoke({"text": "Apple was founded in 1976 by Steve Jobs..."})
# result is an ExtractedData instance
```

`OutputFixingParser` is most useful when using `json_mode` or prompt-based parsers.
For tool-calling structured output, use the manual feedback loop above.

---

## Choosing a Fallback Strategy

When all retries are exhausted, you have options:

```python
def extract_with_fallback(
    text: str,
    llm: ChatOpenAI,
) -> ExtractedData:
    """Try extraction, return a safe default if all retries fail."""

    try:
        return extract_with_feedback_retry(text, llm, max_attempts=3)

    except OutputParserException:
        logger.error(f"Extraction failed for text: {text[:100]}...")

        # Option A: Return a sentinel object with null fields
        # Callers must check for this case
        return ExtractedData(
            company_name="UNKNOWN",
            founding_year=1900,       # obviously wrong — triggers downstream checks
            employee_count_estimate="startup",
        )

        # Option B: Re-raise so the caller decides (preferred for APIs)
        # raise

        # Option C: Queue for human review
        # review_queue.append({"text": text, "error": str(e)})
        # return None
```

Choose your fallback based on the risk profile:

- High-stakes data (legal, medical, financial) → fail loudly and queue for review.
- Low-stakes data (content tagging, summaries) → return safe defaults with logging.

---

## Common Pitfalls

| Pitfall                                      | What goes wrong                                                     | Fix                                                        |
| -------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------- |
| Retrying without feedback                    | Same wrong output every time                                        | Include previous failure and error in messages             |
| Infinite retry loop                          | Cost explodes; application hangs                                    | Always cap retries (3 is a good default)                   |
| Swallowing exceptions silently               | Bad data enters the system undetected                               | Log every failure; alert on high failure rates             |
| Retrying network errors the same way         | Rate limit retries need backoff; parse retries don't                | Distinguish `OutputParserException` from `httpx.HTTPError` |
| Using `OutputFixingParser` with tool calling | `OutputFixingParser` is designed for text parsers, not tool calling | Use manual feedback loop for `with_structured_output()`    |
| Not logging the raw output on failure        | You can't debug what failed                                         | Always log `result["raw"]` when `include_raw=True`         |

---

## Mini Summary

- `OutputParserException` fires when Pydantic rejects the model's output.
- The basic retry loop (up to 3 attempts) catches transient failures.
- Feedback-as-context passes the error message back to the model — self-correction rate is high.
- `OutputFixingParser` implements feedback automatically for prompt-based parsers.
- Cap retries at 3; never retry indefinitely in production.
- Choose a fallback strategy (fail loudly, return defaults, or queue) based on risk level.
