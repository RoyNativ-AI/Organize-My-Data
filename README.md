# Organize My Data

Transform unstructured text into structured data using AI with guaranteed schema compliance.

## Features

- **Structured Outputs** - Guaranteed JSON schema compliance using OpenAI's native Structured Outputs
- **Multi-Provider Support** - OpenAI, Anthropic Claude, Ollama, vLLM
- **Pydantic Integration** - Define schemas with Python classes or JSON Schema
- **Validation Layer** - Automatic output validation with Pydantic
- **Retry Mechanism** - Exponential backoff with configurable attempts
- **Type Safety** - Full type hints and runtime validation

## Installation

```bash
pip install organize-my-data
```

## Quick Start

### Using Pydantic Models

```python
from organize_my_data import SchemaOrganizer
from pydantic import BaseModel
from typing import Optional

class Product(BaseModel):
    name: str
    description: str
    price: Optional[float] = None
    url: Optional[str] = None

text = """
Smart Home Security Camera
Crystal-clear 1080p video quality.
Price: $129.99
https://example.com/camera
"""

organizer = SchemaOrganizer(provider="openai", model="gpt-4o-mini")
result = organizer.organize(text, Product)
# {'name': 'Smart Home Security Camera', 'description': '...', 'price': 129.99, 'url': '...'}
```

### Using JSON Schema

```python
from organize_my_data import SchemaOrganizer

schema = {
    "type": "object",
    "properties": {
        "full_name": {"type": "string"},
        "email": {"type": "string"},
        "phone": {"type": "string"}
    }
}

text = "Contact John Smith at john@example.com or call 555-1234"

organizer = SchemaOrganizer(provider="openai")
result = organizer.organize(text, schema)
# {'full_name': 'John Smith', 'email': 'john@example.com', 'phone': '555-1234'}
```

## Supported Providers

| Provider | Structured Outputs | Setup |
|----------|-------------------|-------|
| OpenAI | Native support | `OPENAI_API_KEY` |
| Claude | Tool-based | `ANTHROPIC_API_KEY` |
| Ollama | JSON mode | Local install |
| vLLM | Guided decoding | Local server |

### Provider Examples

```python
# OpenAI (default)
organizer = SchemaOrganizer(provider="openai", model="gpt-4o")

# Claude
organizer = SchemaOrganizer(provider="claude", model="claude-sonnet-4-20250514")

# Ollama (local)
organizer = SchemaOrganizer(provider="ollama", model="llama3.1")

# vLLM (local server)
organizer = SchemaOrganizer(provider="vllm", base_url="http://localhost:8000")
```

## Configuration

```python
from organize_my_data import SchemaOrganizer, OrganizerConfig, RetryConfig

config = OrganizerConfig(
    provider="openai",
    model="gpt-4o",
    temperature=0.1,
    validate_output=True,
    strict_mode=True,
    retry=RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        max_delay=30.0
    )
)

organizer = SchemaOrganizer(config=config)
```

## CLI Usage

```bash
# Basic usage
organize-my-data --input data.txt --schema schema.json --output result.json

# With specific provider
organize-my-data -i data.txt -s schema.json -p claude -m claude-sonnet-4-20250514
```

## Batch Processing

```python
texts = [
    "Product A - $10",
    "Product B - $20",
    "Product C - $30"
]

results = organizer.organize_batch(texts, schema)
```

## Verification (LLM-as-Judge)

Cross-model verification with confidence scores and source citations:

```python
from organize_my_data import VerifiedOrganizer

organizer = VerifiedOrganizer(
    extractor_provider="openai",
    extractor_model="gpt-4o-mini",
    verifier_provider="openai",      # Use different model for verification
    verifier_model="gpt-4o",
    confidence_threshold=0.8
)

result = organizer.organize_and_verify(text, schema)

# Result includes verification details:
# {
#   "result": {"name": "John", "price": 129.99},
#   "verification": {
#     "overall_confidence": 0.95,
#     "needs_review": false,
#     "fields": [
#       {
#         "field": "name",
#         "confidence": 1.0,
#         "source_quote": "My name is John",
#         "status": "verified"
#       },
#       ...
#     ]
#   },
#   "needs_review": false
# }
```

## Error Handling

```python
from organize_my_data import OrganizeError, SchemaValidationError, ProviderError

try:
    result = organizer.organize(text, schema)
except SchemaValidationError as e:
    print(f"Output validation failed: {e}")
except ProviderError as e:
    print(f"LLM provider error: {e}")
except OrganizeError as e:
    print(f"Organization failed: {e}")
```

## License

**Proprietary Software** - All rights reserved by Officely AI.

This software requires a commercial license for any use beyond evaluation.
See [LICENSE](LICENSE) for details.

Contact: roy@officely.ai

---

Built by [Officely AI](https://officely.ai)
