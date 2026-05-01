# shared/utils

Shared utility modules used across all curriculum levels.

| Module               | Purpose                                                            |
| -------------------- | ------------------------------------------------------------------ |
| `langsmith_setup.py` | One-call LangSmith tracing bootstrap                               |
| `model_factory.py`   | Provider-agnostic chat model factory (OpenAI / Anthropic / Google) |
| `logging_config.py`  | Structured logging with optional JSON mode                         |

## Usage

```python
from shared.utils.langsmith_setup import setup_langsmith
from shared.utils.model_factory import get_chat_model
from shared.utils.logging_config import configure_logging

setup_langsmith("my-project-name")
llm = get_chat_model(provider="openai", temperature=0)
log = configure_logging()
```

Add `shared/` to your `PYTHONPATH` or install it as an editable package to use
these utilities from any project in the repo.
