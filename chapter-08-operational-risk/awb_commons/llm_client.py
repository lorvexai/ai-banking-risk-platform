"""
awb_commons.llm_client — LLM client factory for AWB-AI-2025.
Defaults to Gemini 3.5 Flash; override with MODEL_ID env var.
Avon & Wessex Bank plc (AWB), Bristol, UK.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

MODEL_ID = os.getenv("MODEL_ID", "gemini-3.5-flash")


class LLMClient:
    """
    Thin wrapper around the LLM provider API.
    Supports Gemini 3.5 Flash (default), GPT-5.5, Claude Sonnet 4.6.
    Change MODEL_ID environment variable to switch models.
    """

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        logger.info("LLMClient initialised: model=%s", model_id)

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> str:
        """
        Send a completion request to the configured LLM.

        Args:
            prompt: User-facing prompt text.
            system: System instruction (optional).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = deterministic).

        Returns:
            Generated text response.
        """
        # Production: call the appropriate provider API.
        # Test/dry-run: return stub response.
        dry_run = os.getenv("LLM_DRY_RUN", "false").lower() == "true"
        if dry_run:
            logger.debug("LLM_DRY_RUN: returning stub response")
            return '{"stub": true, "result": "dry_run"}'
        raise NotImplementedError(
            "Set LLM_DRY_RUN=true for tests or configure "
            "provider credentials for production."
        )
