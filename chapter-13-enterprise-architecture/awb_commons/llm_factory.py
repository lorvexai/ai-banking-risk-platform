"""awb_commons/llm_factory.py
LLM client factory — DORA Art.28 multi-provider strategy.
AWB distribution: Gemini 3.5 Flash 68% | Claude Sonnet 4.6 17%
                  GPT-5.5 15%  — no provider exceeds 70%.
Automatic failover: if Gemini error rate > 50% for 5 min,
route new requests to GPT-5 Mini fallback.
"""
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Approved model IDs — master_prompt_v3.1 §LLM MODEL STANDARDS
APPROVED_MODELS = {
    "gemini-3.5-flash":    "google",
    "gemini-3.1-pro":      "google",
    "gpt-5.5":             "openai",
    "gpt-5-mini":        "openai",
    "claude-sonnet-4-6":   "anthropic",
    "claude-haiku-4-5-20251001": "anthropic",
}

FALLBACK_MODEL = "gpt-5-mini"
FAILOVER_ERROR_THRESHOLD = 0.5   # 50% errors triggers failover
FAILOVER_WINDOW_SECONDS = 300    # 5-minute rolling window


@dataclass
class LLMResponse:
    """Standardised response wrapper across all providers."""
    content: str
    model_id: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_gbp: float = 0.0


@dataclass
class _ProviderHealth:
    errors: list[float] = field(default_factory=list)
    calls: list[float] = field(default_factory=list)

    def record(self, success: bool) -> None:
        now = time.time()
        self.calls.append(now)
        if not success:
            self.errors.append(now)
        cutoff = now - FAILOVER_WINDOW_SECONDS
        self.errors = [t for t in self.errors if t > cutoff]
        self.calls = [t for t in self.calls if t > cutoff]

    def error_rate(self) -> float:
        if len(self.calls) < 5:
            return 0.0
        return len(self.errors) / len(self.calls)


class AWBLLMFactory:
    """Multi-provider LLM client factory.

    Usage:
        factory = AWBLLMFactory()
        response = factory.generate(
            model_id="gemini-3.5-flash",
            prompt="Summarise this credit pack.",
            max_tokens=1000,
        )

    Args:
        dry_run: If True, return stub responses (CI/CD testing).
    """

    # GBP cost per 1M tokens (June 2026) — master_prompt_v3.1
    _INPUT_COST = {
        "gemini-3.5-flash":  0.24,
        "gemini-3.1-pro":    1.50,
        "gpt-5.5":           1.57,
        "gpt-5-mini":      0.31,
        "claude-sonnet-4-6": 2.36,
        "claude-haiku-4-5-20251001": 0.79,
    }
    _OUTPUT_COST = {
        "gemini-3.5-flash":  1.97,
        "gemini-3.1-pro":    5.91,
        "gpt-5.5":           6.30,
        "gpt-5-mini":      1.26,
        "claude-sonnet-4-6": 11.81,
        "claude-haiku-4-5-20251001": 3.94,
    }

    def __init__(self, dry_run: bool = False) -> None:
        self._dry_run = dry_run or (
            os.environ.get("AGENT_DRY_RUN", "").lower() == "true"
        )
        self._health: dict[str, _ProviderHealth] = {
            p: _ProviderHealth()
            for p in set(APPROVED_MODELS.values())
        }

    def generate(
        self,
        model_id: str,
        prompt: str,
        system: str = "",
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Generate a completion.

        Args:
            model_id:   One of APPROVED_MODELS keys.
            prompt:     User message.
            system:     Optional system prompt.
            max_tokens: Output token budget.

        Returns:
            LLMResponse with content and cost attribution.

        Raises:
            ValueError: If model_id not in approved list.
        """
        if model_id not in APPROVED_MODELS:
            raise ValueError(
                f"Model '{model_id}' not approved. "
                f"Use: {list(APPROVED_MODELS)}"
            )

        provider = APPROVED_MODELS[model_id]
        # DORA Art.28 failover check
        if self._health[provider].error_rate() > FAILOVER_THRESHOLD:
            logger.warning(
                "llm_provider_failover",
                extra={
                    "original": model_id,
                    "fallback": FALLBACK_MODEL,
                    "error_rate": round(
                        self._health[provider].error_rate(), 2
                    ),
                },
            )
            model_id = FALLBACK_MODEL
            provider = APPROVED_MODELS[model_id]

        if self._dry_run:
            return LLMResponse(
                content=f"[DRY RUN] {model_id}: stub response",
                model_id=model_id,
                provider=provider,
            )

        try:
            content, in_tok, out_tok = self._call_provider(
                provider, model_id, system, prompt, max_tokens
            )
            self._health[provider].record(success=True)
            cost = (
                in_tok / 1_000_000 * self._INPUT_COST.get(model_id, 0)
                + out_tok / 1_000_000
                * self._OUTPUT_COST.get(model_id, 0)
            )
            logger.info(
                "llm_call_success",
                extra={
                    "model": model_id,
                    "in_tokens": in_tok,
                    "out_tokens": out_tok,
                    "cost_gbp": round(cost, 6),
                },
            )
            return LLMResponse(
                content=content,
                model_id=model_id,
                provider=provider,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_gbp=cost,
            )
        except Exception as exc:
            self._health[provider].record(success=False)
            logger.error(
                "llm_call_failed",
                extra={
                    "model": model_id,
                    "error": str(exc),
                },
            )
            raise

    def _call_provider(
        self,
        provider: str,
        model_id: str,
        system: str,
        prompt: str,
        max_tokens: int,
    ) -> tuple[str, int, int]:
        """Route to provider SDK. Returns (content, in_tok, out_tok)."""
        if provider == "google":
            return self._call_google(
                model_id, system, prompt, max_tokens
            )
        if provider == "openai":
            return self._call_openai(
                model_id, system, prompt, max_tokens
            )
        if provider == "anthropic":
            return self._call_anthropic(
                model_id, system, prompt, max_tokens
            )
        raise ValueError(f"Unknown provider: {provider}")

    def _call_google(
        self,
        model_id: str,
        system: str,
        prompt: str,
        max_tokens: int,
    ) -> tuple[str, int, int]:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel(
            model_id,
            system_instruction=system or None,
        )
        resp = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens},
        )
        text = resp.text or ""
        return (
            text,
            resp.usage_metadata.prompt_token_count or 0,
            resp.usage_metadata.candidates_token_count or 0,
        )

    def _call_openai(
        self,
        model_id: str,
        system: str,
        prompt: str,
        max_tokens: int,
    ) -> tuple[str, int, int]:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
        )
        return (
            resp.choices[0].message.content or "",
            resp.usage.prompt_tokens,
            resp.usage.completion_tokens,
        )

    def _call_anthropic(
        self,
        model_id: str,
        system: str,
        prompt: str,
        max_tokens: int,
    ) -> tuple[str, int, int]:
        import anthropic
        client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        text = "".join(
            b.text for b in resp.content
            if hasattr(b, "text")
        )
        return (
            text,
            resp.usage.input_tokens,
            resp.usage.output_tokens,
        )


# Module-level constant referenced in llm_factory
FAILOVER_THRESHOLD = FAILOVER_ERROR_THRESHOLD
