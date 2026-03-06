"""
Verification Module - LLM-as-Judge for validating extraction results
Uses a different model to verify extractions and provide confidence scores
"""

import json
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from enum import Enum

from .providers import LLMProvider, get_provider


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    UNCERTAIN = "uncertain"
    INVALID = "invalid"


class FieldVerification(BaseModel):
    """Verification result for a single field."""
    field_name: str
    extracted_value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    source_quote: Optional[str] = None
    status: VerificationStatus
    reason: Optional[str] = None


class VerificationResult(BaseModel):
    """Complete verification result for an extraction."""
    field_verifications: List[FieldVerification]
    overall_confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool
    verified_count: int
    uncertain_count: int
    invalid_count: int


class VerificationResponse(BaseModel):
    """Schema for LLM verification response."""
    verifications: List[Dict[str, Any]]
    overall_assessment: str


class Verifier:
    """
    Verifies extraction results using a different LLM model.

    Uses LLM-as-Judge pattern to validate that extracted data
    matches the source text with confidence scores and citations.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        confidence_threshold: float = 0.8,
    ):
        """
        Initialize the verifier.

        Args:
            provider: LLM provider for verification (default: anthropic for cross-model)
            model: Model name
            api_key: API key for the provider
            base_url: Custom base URL
            confidence_threshold: Minimum confidence to mark as verified (0.0-1.0)
        """
        self.provider: LLMProvider = get_provider(
            provider_name=provider,
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.confidence_threshold = confidence_threshold

    def verify(
        self,
        original_text: str,
        extracted_data: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """
        Verify extracted data against the original text.

        Args:
            original_text: The source text that was processed
            extracted_data: The data extracted by the primary LLM
            schema: Optional schema for context

        Returns:
            VerificationResult with confidence scores and citations
        """
        verification_prompt = self._build_verification_prompt(
            original_text, extracted_data, schema
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a verification assistant. Your job is to check if "
                    "extracted data accurately matches the source text. "
                    "For each field, provide:\n"
                    "1. confidence (0.0-1.0) - how confident you are the extraction is correct\n"
                    "2. source_quote - the exact text from the source that supports this value\n"
                    "3. status - 'verified', 'uncertain', or 'invalid'\n"
                    "4. reason - explanation if uncertain or invalid\n\n"
                    "Be strict. If the value doesn't appear in the text, mark it invalid. "
                    "If it's inferred but not explicit, mark it uncertain."
                )
            },
            {"role": "user", "content": verification_prompt}
        ]

        response_schema = {
            "type": "object",
            "properties": {
                "verifications": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field_name": {"type": "string"},
                            "confidence": {"type": "number"},
                            "source_quote": {"type": "string"},
                            "status": {"type": "string", "enum": ["verified", "uncertain", "invalid"]},
                            "reason": {"type": "string"}
                        },
                        "required": ["field_name", "confidence", "status", "source_quote", "reason"],
                        "additionalProperties": False
                    }
                },
                "overall_assessment": {"type": "string"}
            },
            "required": ["verifications", "overall_assessment"],
            "additionalProperties": False
        }

        result = self.provider.generate_json(
            messages=messages,
            schema=response_schema,
            model=self.model,
            temperature=0.1,
        )

        return self._parse_verification_result(result, extracted_data)

    def _build_verification_prompt(
        self,
        original_text: str,
        extracted_data: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the verification prompt."""
        prompt = f"""Verify the following extraction:

## Source Text:
{original_text}

## Extracted Data:
{json.dumps(extracted_data, indent=2, ensure_ascii=False)}
"""
        if schema:
            prompt += f"""
## Expected Schema:
{json.dumps(schema, indent=2)}
"""
        prompt += """
For each field in the extracted data, verify if the value is correct based on the source text.
Provide your verification in JSON format."""

        return prompt

    def _parse_verification_result(
        self,
        llm_response: Dict[str, Any],
        extracted_data: Dict[str, Any],
    ) -> VerificationResult:
        """Parse LLM response into VerificationResult."""
        field_verifications = []
        verified_count = 0
        uncertain_count = 0
        invalid_count = 0
        total_confidence = 0.0

        verifications = llm_response.get("verifications", [])

        for v in verifications:
            field_name = v.get("field_name", "")
            confidence = float(v.get("confidence", 0.0))
            status_str = v.get("status", "uncertain")

            try:
                status = VerificationStatus(status_str)
            except ValueError:
                status = VerificationStatus.UNCERTAIN

            field_verification = FieldVerification(
                field_name=field_name,
                extracted_value=extracted_data.get(field_name),
                confidence=confidence,
                source_quote=v.get("source_quote"),
                status=status,
                reason=v.get("reason"),
            )
            field_verifications.append(field_verification)
            total_confidence += confidence

            if status == VerificationStatus.VERIFIED:
                verified_count += 1
            elif status == VerificationStatus.UNCERTAIN:
                uncertain_count += 1
            else:
                invalid_count += 1

        num_fields = len(field_verifications) or 1
        overall_confidence = total_confidence / num_fields

        needs_review = (
            overall_confidence < self.confidence_threshold
            or invalid_count > 0
            or uncertain_count > num_fields * 0.3
        )

        return VerificationResult(
            field_verifications=field_verifications,
            overall_confidence=overall_confidence,
            needs_review=needs_review,
            verified_count=verified_count,
            uncertain_count=uncertain_count,
            invalid_count=invalid_count,
        )


class VerifiedOrganizer:
    """
    Schema organizer with built-in verification.

    Extracts data with one model and verifies with another.
    """

    def __init__(
        self,
        extractor_provider: str = "openai",
        extractor_model: Optional[str] = None,
        verifier_provider: str = "anthropic",
        verifier_model: Optional[str] = None,
        extractor_api_key: Optional[str] = None,
        verifier_api_key: Optional[str] = None,
        confidence_threshold: float = 0.8,
        auto_retry_on_invalid: bool = True,
        max_retries: int = 2,
    ):
        from .schema_organizer import SchemaOrganizer

        self.extractor = SchemaOrganizer(
            provider=extractor_provider,
            model=extractor_model,
            api_key=extractor_api_key,
        )

        self.verifier = Verifier(
            provider=verifier_provider,
            model=verifier_model,
            api_key=verifier_api_key,
            confidence_threshold=confidence_threshold,
        )

        self.auto_retry = auto_retry_on_invalid
        self.max_retries = max_retries

    def organize_and_verify(
        self,
        text: str,
        schema: Any,
        custom_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract and verify data with confidence scores.

        Args:
            text: Text to extract from
            schema: JSON Schema or Pydantic model
            custom_prompt: Optional custom extraction prompt

        Returns:
            Dict with 'result', 'verification', and 'needs_review'
        """
        json_schema = None
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            json_schema = schema.model_json_schema()
        else:
            json_schema = schema

        best_result = None
        best_verification = None

        for attempt in range(self.max_retries + 1):
            result = self.extractor.organize(text, schema, custom_prompt)
            verification = self.verifier.verify(text, result, json_schema)

            if best_verification is None or verification.overall_confidence > best_verification.overall_confidence:
                best_result = result
                best_verification = verification

            if not verification.needs_review:
                break

            if not self.auto_retry or attempt == self.max_retries:
                break

        return {
            "result": best_result,
            "verification": {
                "overall_confidence": best_verification.overall_confidence,
                "needs_review": best_verification.needs_review,
                "verified_count": best_verification.verified_count,
                "uncertain_count": best_verification.uncertain_count,
                "invalid_count": best_verification.invalid_count,
                "fields": [
                    {
                        "field": fv.field_name,
                        "confidence": fv.confidence,
                        "source_quote": fv.source_quote,
                        "status": fv.status.value,
                        "reason": fv.reason,
                    }
                    for fv in best_verification.field_verifications
                ]
            },
            "needs_review": best_verification.needs_review,
        }
