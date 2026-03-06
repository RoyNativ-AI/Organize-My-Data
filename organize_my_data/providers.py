"""
LLM Provider Abstraction Layer
Supports: OpenAI, Anthropic Claude, Ollama, vLLM
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel
import httpx


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate_structured(
        self,
        messages: list[Dict[str, str]],
        response_model: Type[BaseModel],
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> BaseModel:
        """Generate structured output matching the Pydantic model."""
        pass

    @abstractmethod
    def generate_json(
        self,
        messages: list[Dict[str, str]],
        schema: Dict[str, Any],
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Generate JSON output matching the schema."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider with native Structured Outputs support."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        from openai import OpenAI
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.default_model = "gpt-4o"

    def generate_structured(
        self,
        messages: list[Dict[str, str]],
        response_model: Type[BaseModel],
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> BaseModel:
        """Use OpenAI's native Structured Outputs with Pydantic."""
        response = self.client.beta.chat.completions.parse(
            model=model or self.default_model,
            messages=messages,
            response_format=response_model,
            temperature=temperature,
        )

        if response.choices[0].message.refusal:
            raise ValueError(f"Model refused: {response.choices[0].message.refusal}")

        return response.choices[0].message.parsed

    def generate_json(
        self,
        messages: list[Dict[str, str]],
        schema: Dict[str, Any],
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Use OpenAI's JSON Schema response format."""
        schema_name = schema.get("title", "response").lower().replace(" ", "_")

        response = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": self._prepare_schema(schema),
                }
            },
            temperature=temperature,
        )

        return json.loads(response.choices[0].message.content)

    def _prepare_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare schema for OpenAI Structured Outputs (add required fields)."""
        prepared = schema.copy()

        if prepared.get("type") == "object" and "properties" in prepared:
            if "required" not in prepared:
                prepared["required"] = list(prepared["properties"].keys())
            if "additionalProperties" not in prepared:
                prepared["additionalProperties"] = False

        return prepared


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider with tool-based structured outputs."""

    def __init__(self, api_key: Optional[str] = None):
        import anthropic
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.default_model = "claude-sonnet-4-20250514"

    def generate_structured(
        self,
        messages: list[Dict[str, str]],
        response_model: Type[BaseModel],
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> BaseModel:
        """Use Claude with tool-based extraction for structured output."""
        schema = response_model.model_json_schema()
        result = self.generate_json(messages, schema, model, temperature)
        return response_model.model_validate(result)

    def generate_json(
        self,
        messages: list[Dict[str, str]],
        schema: Dict[str, Any],
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Use Claude's tool calling for structured JSON output."""
        system_msg = None
        claude_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                claude_messages.append(msg)

        tool_name = schema.get("title", "extract_data").lower().replace(" ", "_")

        response = self.client.messages.create(
            model=model or self.default_model,
            max_tokens=4096,
            system=system_msg or "Extract structured data from the input.",
            messages=claude_messages,
            temperature=temperature,
            tools=[{
                "name": tool_name,
                "description": "Extract and structure the data according to the schema",
                "input_schema": schema
            }],
            tool_choice={"type": "tool", "name": tool_name}
        )

        for block in response.content:
            if block.type == "tool_use":
                return block.input

        raise ValueError("No structured output received from Claude")


class OllamaProvider(LLMProvider):
    """Ollama provider for local LLMs with JSON mode."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.default_model = "llama3.1"

    def generate_structured(
        self,
        messages: list[Dict[str, str]],
        response_model: Type[BaseModel],
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> BaseModel:
        """Generate structured output using Ollama."""
        schema = response_model.model_json_schema()
        result = self.generate_json(messages, schema, model, temperature)
        return response_model.model_validate(result)

    def generate_json(
        self,
        messages: list[Dict[str, str]],
        schema: Dict[str, Any],
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Use Ollama's JSON format mode with schema guidance."""
        enhanced_messages = messages.copy()
        schema_instruction = f"\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"

        if enhanced_messages and enhanced_messages[-1]["role"] == "user":
            enhanced_messages[-1] = {
                "role": "user",
                "content": enhanced_messages[-1]["content"] + schema_instruction
            }

        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model or self.default_model,
                    "messages": enhanced_messages,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": temperature}
                }
            )
            response.raise_for_status()
            result = response.json()

        return json.loads(result["message"]["content"])


class VLLMProvider(LLMProvider):
    """vLLM provider with OpenAI-compatible API."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "dummy"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=f"{base_url}/v1")
        self.default_model = "default"

    def generate_structured(
        self,
        messages: list[Dict[str, str]],
        response_model: Type[BaseModel],
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> BaseModel:
        """Generate structured output using vLLM."""
        schema = response_model.model_json_schema()
        result = self.generate_json(messages, schema, model, temperature)
        return response_model.model_validate(result)

    def generate_json(
        self,
        messages: list[Dict[str, str]],
        schema: Dict[str, Any],
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Use vLLM's guided decoding for structured output."""
        enhanced_messages = messages.copy()
        schema_instruction = f"\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"

        if enhanced_messages and enhanced_messages[-1]["role"] == "user":
            enhanced_messages[-1] = {
                "role": "user",
                "content": enhanced_messages[-1]["content"] + schema_instruction
            }

        response = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=enhanced_messages,
            temperature=temperature,
            extra_body={"guided_json": schema}
        )

        return json.loads(response.choices[0].message.content)


def get_provider(
    provider_name: str = "openai",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMProvider:
    """Factory function to get the appropriate LLM provider."""
    providers = {
        "openai": lambda: OpenAIProvider(api_key=api_key, base_url=base_url),
        "anthropic": lambda: AnthropicProvider(api_key=api_key),
        "claude": lambda: AnthropicProvider(api_key=api_key),
        "ollama": lambda: OllamaProvider(base_url=base_url or "http://localhost:11434"),
        "vllm": lambda: VLLMProvider(base_url=base_url or "http://localhost:8000", api_key=api_key or "dummy"),
    }

    provider_name = provider_name.lower()
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. Supported: {list(providers.keys())}")

    return providers[provider_name]()
