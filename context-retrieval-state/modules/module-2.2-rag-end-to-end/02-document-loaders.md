# 02 — Document Loaders

> **Previous:** [01 → Why RAG Exists](01-why-rag-exists.md) | **Next:** [03 → Text Splitters](03-text-splitters.md)

---

## Real-World Analogy

Before a librarian can help you find information, someone must shelve the books.
Document loaders are the process of bringing raw source material into the system.
The quality of what gets loaded determines everything downstream.
A loader that silently drops pages or loses metadata is a reliability problem,
not a format problem.

---

## Why Loaders (Not Just `open(file)`)

You could open files with Python's standard library.
But LangChain document loaders do more:

```
Python open() gives you:          LangChain loaders give you:
  ┌───────────────┐                ┌────────────────────────────────────────┐
  │  raw string   │                │  Document(                             │
  │               │                │    page_content="the actual text...",  │
  │               │                │    metadata={                          │
  │               │                │      "source": "/docs/policy.pdf",     │
  │               │                │      "page": 4,                        │
  │               │                │      "author": "Legal Team",           │
  │               │                │      "last_modified": "2024-01-15",    │
  │               │                │    }                                   │
  └───────────────┘                │  )                                     │
                                   └────────────────────────────────────────┘
```

Metadata is what makes citations possible.
Without `metadata["source"]` and `metadata["page"]`, you cannot tell the user
where the answer came from.

---

## The Document Object

Every loader returns a list of `Document` objects.
Understand this structure before using any loader:

```python
from langchain_core.documents import Document

# A Document has exactly two fields:
doc = Document(
    page_content="The quarterly revenue increased by 23%...",   # the text
    metadata={
        "source": "Q3-2024-earnings.pdf",    # where this came from
        "page": 7,                           # which page
        "section": "Financial Highlights",   # any extra structure you add
    },
)

print(doc.page_content[:50])   # first 50 chars of text
print(doc.metadata["source"])  # "Q3-2024-earnings.pdf"
```

---

## Loader 1 — PyPDFLoader

Extracts text from PDF files. Returns one `Document` per page.

```python
from langchain_community.document_loaders import PyPDFLoader

# Load a single PDF file
loader = PyPDFLoader(file_path="./documents/employee-handbook.pdf")
pages = loader.load()
# ↑ Returns a list — one Document per page in the PDF

print(f"Loaded {len(pages)} pages")

# Inspect a single page
print(pages[0].page_content[:200])    # first 200 characters of page 1
print(pages[0].metadata)
# {
#   "source": "./documents/employee-handbook.pdf",
#   "page": 0   ← 0-indexed page number
# }

# Add custom metadata before indexing
for doc in pages:
    doc.metadata["document_type"] = "hr_policy"   # custom enrichment
    doc.metadata["department"] = "hr"
```

### Why Per-Page Documents?

```
Full 200-page PDF as one Document:
  - Can't retrieve a relevant section without loading all 200 pages
  - Single metadata entry; can't cite which page

Per-page Documents (200 Documents):
  - Retrieval returns only the relevant pages
  - Each Document has a page number in metadata
  - Enables exact citations: "page 47 of employee-handbook.pdf"
```

---

## Loader 2 — WebBaseLoader

Loads text from web pages. Strips HTML; returns the plain text content.

```python
from langchain_community.document_loaders import WebBaseLoader
import bs4   # BeautifulSoup4 — for HTML parsing

# Load a single web page
loader = WebBaseLoader(
    web_paths=["https://docs.python.org/3/library/pathlib.html"],
    # bs_kwargs: passed to BeautifulSoup for HTML filtering
    # Use this to target specific HTML elements and exclude nav/footer noise
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("section", "body", "article")
            # ↑ Only extract content inside these CSS classes
            # Without this, you get navigation menus and page chrome mixed in
        )
    ),
)

docs = loader.load()
print(f"Loaded {len(docs)} document(s)")
print(docs[0].metadata)
# {"source": "https://docs.python.org/3/library/pathlib.html"}
print(docs[0].page_content[:300])

# Load multiple pages in one call
multi_loader = WebBaseLoader(
    web_paths=[
        "https://python.langchain.com/docs/introduction",
        "https://python.langchain.com/docs/get_started",
    ]
)
all_docs = multi_loader.load()   # returns 2 Documents, one per URL
```

### WebBaseLoader vs Scraping APIs

`WebBaseLoader` does a straightforward HTTP GET.
It does not handle JavaScript-rendered content, authentication, or rate limiting.
For production web scraping, use a dedicated service or Playwright-based loader.

---

## Loader 3 — CSVLoader

Treats each row of a CSV as a separate document.
Useful for structured records: product catalogue, FAQ list, customer data.

```python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="./data/products.csv",
    # source_column tells the loader which CSV column to use as the "source" metadata
    # Without this, the source is just the filename
    source_column="product_id",
    # csv_args: passed to csv.DictReader for custom delimiters etc.
    csv_args={"delimiter": ",", "quotechar": '"'},
)

rows = loader.load()
print(f"Loaded {len(rows)} product records")

# Each Document contains all the row's fields as formatted text
print(rows[0].page_content)
# "product_id: P001\nname: Wireless Keyboard\nprice: 49.99\ncategory: peripherals"
# ↑ Format: key: value\nkey: value... for each column in the row

print(rows[0].metadata)
# {"source": "P001", "row": 0}
#   source = value from "product_id" column (because source_column="product_id")
#   row    = 0-indexed row number

# Add metadata post-load for all documents
for doc in rows:
    doc.metadata["data_type"] = "product_catalog"
    doc.metadata["loaded_at"] = "2024-01-15"
```

---

## Loader 4 — DirectoryLoader

Loads all matching files in a directory. Useful for bulk indexing.

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Load all PDF files in a directory (recursively)
loader = DirectoryLoader(
    path="./documents",          # root directory to scan
    glob="**/*.pdf",             # glob pattern — ** matches any subdirectory
    loader_cls=PyPDFLoader,      # which loader to use for each matched file
    show_progress=True,          # print a progress bar (useful for large directories)
    use_multithreading=True,     # load multiple files in parallel
    silent_errors=True,          # skip files that fail to load; don't crash
)

all_docs = loader.load()
print(f"Loaded {len(all_docs)} page(s) from all PDFs")

# Mix file types with multiple loaders
from langchain_community.document_loaders import TextLoader

text_loader = DirectoryLoader(
    path="./documents",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
txt_docs = text_loader.load()
```

### Directory Loader File Type Reference

| File type | `loader_cls`     | Notes                          |
| --------- | ---------------- | ------------------------------ |
| `.pdf`    | `PyPDFLoader`    | Most reliable for PDFs         |
| `.txt`    | `TextLoader`     | Plain text; set `encoding`     |
| `.md`     | `TextLoader`     | Markdown treated as plain text |
| `.csv`    | `CSVLoader`      | One Document per row           |
| `.html`   | `BSHTMLLoader`   | Strips tags                    |
| `.docx`   | `Docx2txtLoader` | Requires `docx2txt` package    |

---

## Metadata Propagation: Why It Matters for Citations

When a user asks "What does your policy say about overtime?",
they need to know: which policy? which page? which section?

Without metadata, you can only say: "The policy says X."
With metadata, you can say: "The Employee Handbook, page 12, Section 3.4 states: X."

```python
# Enriching metadata after loading — the standard pattern
def load_and_enrich_pdf(
    file_path: str,
    document_type: str,
    department: str,
) -> list:
    """Load a PDF and enrich every page with custom metadata."""
    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load()

    for doc in docs:
        # Enrich: add custom fields alongside the loader's built-in metadata
        doc.metadata["document_type"] = document_type
        doc.metadata["department"]    = department
        # source and page are already set by PyPDFLoader

    return docs

# Usage:
hr_docs = load_and_enrich_pdf(
    "./docs/employee-handbook.pdf",
    document_type="hr_policy",
    department="hr",
)
# Each Document now has:
# metadata = {
#   "source": "./docs/employee-handbook.pdf",   ← from PyPDFLoader
#   "page": N,                                  ← from PyPDFLoader
#   "document_type": "hr_policy",               ← our enrichment
#   "department": "hr",                         ← our enrichment
# }
```

---

## Common Pitfalls

| Pitfall                                    | What goes wrong                                                   | Fix                                                                     |
| ------------------------------------------ | ----------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Not setting `source_column` in CSVLoader   | All rows have the same source metadata; can't distinguish records | Always set `source_column` to a unique identifier column                |
| WebBaseLoader on JavaScript-rendered pages | Empty or garbled content — JS hasn't executed                     | Use a headless browser loader or a scraping API                         |
| Loading PDFs with scanned images only      | PyPDFLoader returns empty strings for image-only pages            | Use an OCR step before loading (e.g., pytesseract)                      |
| No `silent_errors=True` in DirectoryLoader | One corrupt file crashes the entire indexing run                  | Enable `silent_errors` and log which files failed                       |
| No metadata enrichment                     | Can't cite which document or page an answer came from             | Always add `document_type`, `department`, or other context at load time |

---

## Mini Summary

- All loaders return `list[Document]`, where each `Document` has `page_content` and `metadata`.
- `PyPDFLoader` returns one `Document` per PDF page — ideal for citation by page number.
- `WebBaseLoader` strips HTML and returns the text content; use `bs_kwargs` to filter noise.
- `CSVLoader` returns one `Document` per row; set `source_column` for meaningful metadata.
- `DirectoryLoader` bulk-loads an entire directory; use `show_progress` for large datasets.
- Always enrich metadata after loading — it is the foundation of citation capability.
