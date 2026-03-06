"""
Schema Organizer - Transform unstructured text to structured data
Uses LLM providers with Structured Outputs for guaranteed schema compliance
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, ValidationError, create_model
from pydantic.fields import FieldInfo

from .providers import LLMProvider, get_provider


class OrganizeError(Exception):
    """Base exception for organization errors."""
    pass


class SchemaValidationError(OrganizeError):
    """Raised when output doesn't match schema."""
    pass


class ProviderError(OrganizeError):
    """Raised when LLM provider fails."""
    pass


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0


class OrganizerConfig(BaseModel):
    """Configuration for the schema organizer."""
    provider: str = "openai"
    model: Optional[str] = None
    temperature: float = 0.1
    max_tokens_per_chunk: int = 4000
    retry: RetryConfig = RetryConfig()
    validate_output: bool = True
    strict_mode: bool = True


class SchemaOrganizer:
    """
    Transform unstructured text into structured data using LLMs.

    Supports multiple providers (OpenAI, Claude, Ollama, vLLM) with
    Structured Outputs for guaranteed schema compliance.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        config: Optional[OrganizerConfig] = None,
        base_url: Optional[str] = None,
    ):
        self.config = config or OrganizerConfig(provider=provider, model=model)
        self.logger = self._setup_logger()

        self.provider: LLMProvider = get_provider(
            provider_name=self.config.provider,
            api_key=api_key,
            base_url=base_url,
        )

        self.logger.info(f"Initialized with provider: {self.config.provider}")

    @staticmethod
    def _setup_logger() -> logging.Logger:
        logger = logging.getLogger("SchemaOrganizer")
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry."""
        last_error = None
        retry_config = self.config.retry

        for attempt in range(retry_config.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < retry_config.max_attempts - 1:
                    delay = min(
                        retry_config.base_delay * (retry_config.exponential_base ** attempt),
                        retry_config.max_delay
                    )
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(f"All {retry_config.max_attempts} attempts failed")

        raise ProviderError(f"Failed after {retry_config.max_attempts} attempts: {last_error}")

    def _build_messages(
        self,
        text: str,
        schema: Dict[str, Any],
        custom_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build the message list for the LLM."""
        system_prompt = custom_prompt or (
            "You are a data extraction assistant. Analyze the given text and "
            "extract information according to the provided schema. Be precise "
            "and only include information that is explicitly stated or can be "
            "directly inferred from the text. If a field cannot be determined, "
            "use null or an appropriate default value."
        )

        user_content = (
            f"Extract structured data from the following text according to this schema:\n\n"
            f"Schema:\n{json.dumps(schema, indent=2)}\n\n"
            f"Text to analyze:\n{text}"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _json_schema_to_pydantic(self, schema: Dict[str, Any]) -> Type[BaseModel]:
        """Convert JSON Schema to a dynamic Pydantic model."""
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        field_definitions = {}

        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        for field_name, field_spec in properties.items():
            field_type = type_mapping.get(field_spec.get("type", "string"), Any)

            if field_name in required:
                field_definitions[field_name] = (field_type, ...)
            else:
                field_definitions[field_name] = (Optional[field_type], None)

        model_name = schema.get("title", "DynamicModel").replace(" ", "")
        return create_model(model_name, **field_definitions)

    def organize(
        self,
        text: str,
        schema: Union[Dict[str, Any], Type[BaseModel]],
        custom_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Organize unstructured text into structured data.

        Args:
            text: The unstructured text to organize
            schema: JSON Schema dict or Pydantic model class
            custom_prompt: Optional custom system prompt

        Returns:
            Structured data as a dictionary
        """
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            pydantic_model = schema
            json_schema = schema.model_json_schema()
        else:
            json_schema = schema
            pydantic_model = self._json_schema_to_pydantic(schema)

        messages = self._build_messages(text, json_schema, custom_prompt)

        def _execute():
            if self.config.strict_mode:
                result = self.provider.generate_structured(
                    messages=messages,
                    response_model=pydantic_model,
                    model=self.config.model,
                    temperature=self.config.temperature,
                )
                return result.model_dump()
            else:
                return self.provider.generate_json(
                    messages=messages,
                    schema=json_schema,
                    model=self.config.model,
                    temperature=self.config.temperature,
                )

        result = self._retry_with_backoff(_execute)

        if self.config.validate_output:
            self._validate_result(result, pydantic_model)

        return result

    def organize_with_model(
        self,
        text: str,
        response_model: Type[BaseModel],
        custom_prompt: Optional[str] = None,
    ) -> BaseModel:
        """
        Organize text and return a validated Pydantic model instance.

        Args:
            text: The unstructured text to organize
            response_model: Pydantic model class for the response
            custom_prompt: Optional custom system prompt

        Returns:
            Validated Pydantic model instance
        """
        result = self.organize(text, response_model, custom_prompt)
        return response_model.model_validate(result)

    def _validate_result(
        self,
        result: Dict[str, Any],
        model: Type[BaseModel],
    ) -> None:
        """Validate that the result matches the expected schema."""
        try:
            model.model_validate(result)
            self.logger.debug("Output validation passed")
        except ValidationError as e:
            self.logger.error(f"Validation failed: {e}")
            raise SchemaValidationError(f"Output validation failed: {e}")

    def organize_file(
        self,
        input_file: str,
        schema: Union[Dict[str, Any], Type[BaseModel]],
        output_file: Optional[str] = None,
        custom_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Organize text from a file.

        Args:
            input_file: Path to input text file
            schema: JSON Schema dict or Pydantic model class
            output_file: Optional path to save JSON output
            custom_prompt: Optional custom system prompt

        Returns:
            Structured data as a dictionary
        """
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()

        result = self.organize(text, schema, custom_prompt)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Output saved to {output_file}")

        return result

    def organize_batch(
        self,
        texts: List[str],
        schema: Union[Dict[str, Any], Type[BaseModel]],
        custom_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Organize multiple texts with the same schema.

        Args:
            texts: List of unstructured texts
            schema: JSON Schema dict or Pydantic model class
            custom_prompt: Optional custom system prompt

        Returns:
            List of structured data dictionaries
        """
        results = []
        for i, text in enumerate(texts):
            self.logger.info(f"Processing item {i + 1}/{len(texts)}")
            try:
                result = self.organize(text, schema, custom_prompt)
                results.append(result)
            except OrganizeError as e:
                self.logger.error(f"Failed to process item {i + 1}: {e}")
                results.append({"_error": str(e)})

        return results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Organize unstructured text into structured data"
    )
    parser.add_argument("--input", "-i", default="input_text.txt", help="Input text file")
    parser.add_argument("--schema", "-s", default="my_schema.json", help="JSON schema file")
    parser.add_argument("--output", "-o", default="output.json", help="Output JSON file")
    parser.add_argument("--provider", "-p", default="openai", help="LLM provider")
    parser.add_argument("--model", "-m", help="Model name")
    parser.add_argument("--base-url", help="Custom API base URL")

    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

    with open(args.schema, "r") as f:
        schema = json.load(f)

    organizer = SchemaOrganizer(
        api_key=api_key,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
    )

    result = organizer.organize_file(
        input_file=args.input,
        schema=schema,
        output_file=args.output,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
