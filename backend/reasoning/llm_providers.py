"""
LLM provider integrations for reasoning engine.

Supports:
- Google Gemini (with thinking mode)
- OpenAI GPT-4
- Anthropic Claude
"""
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import asyncio

from backend.core.config import settings
from backend.core.logging_config import logger


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate response from LLM."""
        pass

    @abstractmethod
    async def generate_with_thinking(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """Generate response with thinking/reasoning steps."""
        pass


class GeminiProvider(LLMProvider):
    """Google Gemini provider."""

    def __init__(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.genai = genai
            self.model_name = settings.GEMINI_MODEL
            logger.info(f"Initialized Gemini provider: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate response from Gemini."""
        try:
            model = self.genai.GenerativeModel(self.model_name)

            # Combine system prompt and user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Generate response
            response = await asyncio.to_thread(
                model.generate_content,
                full_prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens or 2048,
                },
            )

            return response.text

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise

    async def generate_with_thinking(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """Generate response with thinking mode (Gemini 2.0 Flash Thinking)."""
        try:
            # Use thinking model if available
            thinking_model = "gemini-2.0-flash-thinking-exp"

            model = self.genai.GenerativeModel(thinking_model)

            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            response = await asyncio.to_thread(
                model.generate_content,
                full_prompt,
            )

            # Extract thinking and response
            # Gemini thinking mode returns thoughts in the response
            return {
                "thinking": "Thinking process embedded in response",
                "response": response.text,
            }

        except Exception as e:
            logger.warning(f"Gemini thinking mode failed, falling back: {e}")
            # Fallback to regular generation
            response = await self.generate(prompt, system_prompt)
            return {
                "thinking": "",
                "response": response,
            }


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    def __init__(self):
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self.model_name = settings.OPENAI_MODEL
            logger.info(f"Initialized OpenAI provider: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            raise

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate response from OpenAI."""
        try:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 2048,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

    async def generate_with_thinking(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """Generate response with chain-of-thought prompting."""
        try:
            # Add chain-of-thought instruction
            thinking_prompt = (
                f"{prompt}\n\n"
                "Please think through this step by step before providing your final answer. "
                "Format your response as:\n"
                "THINKING: <your reasoning steps>\n"
                "ANSWER: <your final answer>"
            )

            response = await self.generate(thinking_prompt, system_prompt)

            # Parse thinking and answer
            if "THINKING:" in response and "ANSWER:" in response:
                parts = response.split("ANSWER:")
                thinking = parts[0].replace("THINKING:", "").strip()
                answer = parts[1].strip()
            else:
                thinking = ""
                answer = response

            return {
                "thinking": thinking,
                "response": answer,
            }

        except Exception as e:
            logger.error(f"OpenAI thinking generation failed: {e}")
            raise


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self):
        try:
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
            self.model_name = settings.CLAUDE_MODEL
            logger.info(f"Initialized Claude provider: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Claude: {e}")
            raise

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate response from Claude."""
        try:
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens or 2048,
                temperature=temperature,
                system=system_prompt or "",
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Claude generation failed: {e}")
            raise

    async def generate_with_thinking(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """Generate response with extended thinking."""
        try:
            # Use extended thinking with Claude
            thinking_prompt = (
                f"{prompt}\n\n"
                "Please provide your detailed reasoning process before your final answer."
            )

            response = await self.generate(thinking_prompt, system_prompt, max_tokens=4096)

            return {
                "thinking": "Reasoning embedded in response",
                "response": response,
            }

        except Exception as e:
            logger.error(f"Claude thinking generation failed: {e}")
            raise


# Provider factory
_providers: Dict[str, Optional[LLMProvider]] = {
    "gemini": None,
    "openai": None,
    "anthropic": None,
}


def get_llm_provider(provider_name: Optional[str] = None) -> LLMProvider:
    """
    Get LLM provider instance.

    Args:
        provider_name: Provider name (gemini, openai, anthropic)
                      Defaults to settings.DEFAULT_LLM_PROVIDER

    Returns:
        LLMProvider instance

    Raises:
        ValueError: If provider not supported or not configured
    """
    global _providers

    provider_name = provider_name or settings.DEFAULT_LLM_PROVIDER

    # Create provider if not cached
    if _providers.get(provider_name) is None:
        if provider_name == "gemini":
            if not settings.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY not configured")
            _providers["gemini"] = GeminiProvider()
        elif provider_name == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not configured")
            _providers["openai"] = OpenAIProvider()
        elif provider_name == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not configured")
            _providers["anthropic"] = ClaudeProvider()
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")

    return _providers[provider_name]
