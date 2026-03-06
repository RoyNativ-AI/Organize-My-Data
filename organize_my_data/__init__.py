"""
Organize My Data - Transform unstructured text to structured data using AI
"""

from .schema_organizer import (
    SchemaOrganizer,
    OrganizerConfig,
    RetryConfig,
    OrganizeError,
    SchemaValidationError,
    ProviderError,
)
from .providers import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    VLLMProvider,
    get_provider,
)
from .verification import (
    Verifier,
    VerifiedOrganizer,
    VerificationResult,
    FieldVerification,
    VerificationStatus,
)

__version__ = "2.1.0"
__all__ = [
    # Core
    "SchemaOrganizer",
    "OrganizerConfig",
    "RetryConfig",
    # Errors
    "OrganizeError",
    "SchemaValidationError",
    "ProviderError",
    # Providers
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "VLLMProvider",
    "get_provider",
    # Verification
    "Verifier",
    "VerifiedOrganizer",
    "VerificationResult",
    "FieldVerification",
    "VerificationStatus",
]
