# 05 — Schema Design Patterns

> **Previous:** [04 → Retry and Fallback](04-retry-and-fallback.md) | **Next:** [Module 2.1 → Memory Fundamentals](../../../context-retrieval-state/modules/module-2.1-memory-fundamentals/README.md)

---

## Real-World Analogy

A database table schema for a well-designed application is not just a list of columns.
It has constraints (NOT NULL, UNIQUE, CHECK), foreign keys, and types chosen with care.
A poorly designed table lets you insert garbage. A well-designed table refuses it.

Your Pydantic schema is the same.
A bad schema lets models produce garbage that validates.
A good schema makes garbage fail validation and forces the model to try harder.

---

## The Schema Design Principle

**Every field you leave vague is a field the model will hallucinate.**

Good schema design is about making the schema as specific as possible,
so the model has no room to make up plausible-sounding nonsense.

```
Vague schema:
  class Result(BaseModel):
    data: dict            ← can be anything

Specific schema:
  class Result(BaseModel):
    company_name: str = Field(description="Legal company name as registered")
    industry: IndustryEnum = Field(description="Primary industry vertical")
    headcount_range: str = Field(
        description="One of: '1-10', '11-50', '51-200', '201-500', '500+'",
        pattern=r"^\d+-\d+$|^500\+$"
    )
```

The specific schema reduces hallucination surface area.

---

## Flat vs Nested Schemas

### Flat Schema (Preferred for Simple Extraction)

```python
from pydantic import BaseModel, Field
from typing import List, Optional

# Good: flat, 5-6 fields, each clear
class JobPosting(BaseModel):
    """Information extracted from a job posting."""
    job_title: str = Field(
        description="The exact job title as listed in the posting"
    )
    company: str = Field(
        description="Company name posting the job"
    )
    location: str = Field(
        description="City and country, or 'Remote' if fully remote"
    )
    salary_min: Optional[int] = Field(
        default=None,
        description="Minimum annual salary in USD, or null if not specified",
        ge=0,
    )
    salary_max: Optional[int] = Field(
        default=None,
        description="Maximum annual salary in USD, or null if not specified",
        ge=0,
    )
    required_skills: List[str] = Field(
        default_factory=list,
        description="List of required technical skills mentioned",
        max_length=10,     # cap at 10 — forces prioritisation
    )
```

### Nested Schema (Use When Hierarchy Is Real)

```python
from pydantic import BaseModel, Field
from typing import List

class ContactInfo(BaseModel):
    """Contact details for a person or company."""
    email: Optional[str] = Field(
        default=None,
        description="Primary email address"
    )
    phone: Optional[str] = Field(
        default=None,
        description="Phone number in international format (+1-xxx-xxx-xxxx)"
    )
    linkedin_url: Optional[str] = Field(
        default=None,
        description="LinkedIn profile URL if mentioned"
    )

class LeadProfile(BaseModel):
    """Sales lead extracted from a prospect email."""
    full_name: str = Field(description="Prospect's full name")
    title: str = Field(description="Current job title")
    company: str = Field(description="Current employer")
    contact: ContactInfo = Field(
        description="All contact information for this person"
    )
    pain_points: List[str] = Field(
        default_factory=list,
        description="Business problems the prospect mentioned",
        max_length=5,
    )
    urgency: str = Field(
        description="Buying urgency: 'immediate', 'within_3_months', 'exploring', 'unknown'"
    )
```

**Rule of thumb:** nest when the nested data forms a logical unit (ContactInfo, Address, DateRange).
Do not nest just to group fields — keep it flat.

---

## Good vs Bad Field Design

### Bad Field: Overloaded Single Field

```python
# BAD: one string trying to capture multiple pieces of info
class BadResult(BaseModel):
    date_info: str = Field(
        description="The date mentioned (could be DD/MM/YYYY or written out)"
    )
    # The model will produce inconsistent formats. Your code must parse this string.
```

```python
# GOOD: separate fields with clear types
class GoodResult(BaseModel):
    event_date: Optional[str] = Field(
        default=None,
        description="Date in ISO 8601 format: YYYY-MM-DD. Null if not mentioned.",
        pattern=r"^\d{4}-\d{2}-\d{2}$",   # enforces the format
    )
    date_is_approximate: bool = Field(
        default=False,
        description="True if the date was expressed as approximate ('early 2024')"
    )
```

---

### Bad Field: Unbounded Text Where Structure Exists

```python
# BAD: list of items stuffed into a single string
class BadExtraction(BaseModel):
    features: str = Field(
        description="All product features, comma-separated"
    )
    # Returns: "fast, reliable, cheap" — 3 different types of data as one string
```

```python
# GOOD: list with typed items
class GoodExtraction(BaseModel):
    features: List[str] = Field(
        default_factory=list,
        description="Each distinct product feature as a separate list item",
        max_length=8,
    )
```

---

## Confidence Fields

Adding an explicit confidence field shifts some of the model's uncertainty
from "hallucinate a confident answer" to "provide a calibrated estimate":

```python
from pydantic import BaseModel, Field
from enum import Enum

class ConfidenceLevel(str, Enum):
    HIGH   = "high"     # clear evidence in source text
    MEDIUM = "medium"   # inferred from context
    LOW    = "low"      # not mentioned; best guess

class ExtractedFact(BaseModel):
    claim: str = Field(
        description="The specific fact being extracted"
    )
    evidence_quote: Optional[str] = Field(
        default=None,
        description="The exact sentence from the source that supports this claim. "
                    "Null if the claim is inferred rather than explicitly stated.",
        max_length=300,
    )
    confidence: ConfidenceLevel = Field(
        description="How confident the extraction is, based on evidence in the text"
    )
```

In downstream processing, filter out or flag `LOW` confidence extractions
for human review rather than acting on them automatically.

---

## Nullable Fields Done Right

Two common mistakes with nullable fields:

```python
# MISTAKE 1: Every field is Optional "just in case"
class TooPermissive(BaseModel):
    name: Optional[str] = None       # always present in practice, but not enforced
    score: Optional[int] = None      # model can just return None and "succeed"
    category: Optional[str] = None   # model never has to commit to a category

# MISTAKE 2: Required field for something genuinely optional
class TooStrict(BaseModel):
    phone_number: str   # required, but many records have no phone number
                        # model will hallucinate one just to satisfy the schema
```

```python
# CORRECT: Required for always-present data; Optional only when genuinely absent
class CorrectDesign(BaseModel):
    # Always present in the source — required, no default
    name: str = Field(description="Full legal name")

    # Optional: may not appear in all sources
    phone_number: Optional[str] = Field(
        default=None,
        description="Phone number if mentioned; null if not present in the text",
    )

    # Confidence field for uncertain extractions
    name_confidence: ConfidenceLevel = Field(
        description="Confidence in the name extraction"
    )
```

---

## Enum Constraints for Category Fields

Every field that represents a category, status, or type should be an Enum.
Never use `str` for fields with a fixed valid set:

```python
from enum import Enum

class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"

class Department(str, Enum):
    ENGINEERING   = "engineering"
    MARKETING     = "marketing"
    SALES         = "sales"
    CUSTOMER_OPS  = "customer_ops"
    FINANCE       = "finance"
    HR            = "hr"
    OTHER         = "other"   # always include an OTHER bucket for unknowns

class SupportTicket(BaseModel):
    ticket_id: str = Field(description="Ticket identifier extracted from text")
    priority: Priority = Field(description="Severity level of the issue")
    department: Department = Field(
        description="Which department should handle this ticket"
    )
    issue_summary: str = Field(
        max_length=200,
        description="One-sentence summary of the problem"
    )
    requires_escalation: bool = Field(
        description="True if the issue requires management involvement"
    )
```

---

## Real Extraction Examples

### Pattern 1 — Invoice Extraction

```python
from typing import List, Optional
from pydantic import BaseModel, Field
from decimal import Decimal

class LineItem(BaseModel):
    description: str = Field(description="Product or service description")
    quantity: float = Field(description="Number of units", gt=0)
    unit_price: float = Field(description="Price per unit in USD", ge=0)
    total: float = Field(description="quantity × unit_price", ge=0)

class InvoiceExtraction(BaseModel):
    invoice_number: str = Field(description="Invoice identifier (e.g., INV-2024-0042)")
    vendor_name: str = Field(description="Company issuing the invoice")
    invoice_date: str = Field(
        description="Date in YYYY-MM-DD format",
        pattern=r"^\d{4}-\d{2}-\d{2}$"
    )
    due_date: Optional[str] = Field(
        default=None,
        description="Payment due date in YYYY-MM-DD format, null if not specified",
        pattern=r"^\d{4}-\d{2}-\d{2}$"
    )
    line_items: List[LineItem] = Field(
        default_factory=list,
        description="All line items on the invoice"
    )
    subtotal: float = Field(description="Total before tax", ge=0)
    tax_amount: Optional[float] = Field(default=None, ge=0)
    total_amount: float = Field(description="Final amount due", ge=0)
    currency: str = Field(
        default="USD",
        description="Three-letter currency code (USD, EUR, GBP)"
    )
```

### Pattern 2 — Meeting Notes Extraction

```python
from typing import List
from pydantic import BaseModel, Field
from enum import Enum

class ActionItemStatus(str, Enum):
    OPEN       = "open"
    IN_PROGRESS = "in_progress"
    DONE       = "done"

class ActionItem(BaseModel):
    owner: str = Field(description="Person responsible for this action")
    task: str = Field(
        description="Clear description of what needs to be done",
        max_length=200
    )
    due_date: Optional[str] = Field(
        default=None,
        description="Deadline in YYYY-MM-DD, null if not mentioned"
    )
    status: ActionItemStatus = Field(
        default=ActionItemStatus.OPEN,
        description="Current status of the action item"
    )

class MeetingNotes(BaseModel):
    meeting_title: str = Field(description="Subject or title of the meeting")
    attendees: List[str] = Field(
        default_factory=list,
        description="Full names of everyone who attended"
    )
    key_decisions: List[str] = Field(
        default_factory=list,
        description="Decisions made during the meeting (not actions — facts agreed upon)",
        max_length=10,
    )
    action_items: List[ActionItem] = Field(
        default_factory=list,
        description="All follow-up tasks assigned to specific people"
    )
    next_meeting_date: Optional[str] = Field(
        default=None,
        description="Date of next meeting in YYYY-MM-DD, null if not scheduled"
    )
```

---

## Common Pitfalls

| Pitfall                            | What goes wrong                                      | Fix                                                      |
| ---------------------------------- | ---------------------------------------------------- | -------------------------------------------------------- |
| Using `str` for categorical fields | Model invents new category names                     | Use `Enum` for all fixed-value fields                    |
| No `max_length` on lists           | Model returns 30 items when you asked for key points | Always cap list fields with `max_length`                 |
| `Optional` on every field          | Model never has to commit; returns all nulls         | Only mark genuinely optional fields                      |
| No `description` on nested models  | Model doesn't understand the nested object's purpose | Add `description` to both the field and the nested class |
| Date fields as `datetime`          | Serialisation issues between providers               | Use `str` with a `pattern` constraint for dates          |
| Mixing concerns in one schema      | 15+ field schemas confuse the model                  | Split into 2-3 focused schemas; extract in stages        |

---

## Mini Summary

- Specificity is the goal: the more precise your schema, the less room for hallucination.
- Flat schemas are easier for models than deeply nested ones; keep nesting ≤ 2 levels.
- Use `Enum` for every categorical field — never use `str` for "one of these values".
- Add `confidence` fields to signal uncertainty rather than forcing confident hallucination.
- Use `Optional` only for genuinely absent fields; required fields enforce presence.
- Cap lists with `max_length` to prevent runaway output.
- Build focused schemas with 5-8 fields; split large extractions into multiple passes.
