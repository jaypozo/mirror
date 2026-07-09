"""Pluggable LLM client used by summary and draft generation."""

from __future__ import annotations

import asyncio
import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Sequence

from dotenv import load_dotenv


Message = dict[str, str]


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or str(default))
    except (TypeError, ValueError):
        return default


class LLMClient(ABC):
    name: str
    model: str

    @abstractmethod
    async def complete(self, messages: Sequence[Message]) -> str:
        """Return a text completion for chat-style messages."""


class CodexLLMClient(LLMClient):
    """Drafts via GPT-5.5 through `codex exec` (stateless, on-demand).

    Codex authenticates through ChatGPT OAuth, so no OpenAI API key is needed.
    Each call is a fresh, one-shot `codex exec` — no
    session is maintained between drafts (the owner's decision).
    """

    name = "codex"

    def __init__(
        self,
        model: str = "gpt-5.5",
        reasoning_effort: str = "high",
        binary: str = "codex",
        timeout_seconds: int = 300,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.binary = binary
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _flatten(messages: Sequence[Message]) -> str:
        parts: list[str] = []
        for message in messages:
            role = message.get("role", "user").upper()
            content = message.get("content", "")
            if role == "SYSTEM":
                parts.append(f"[SYSTEM INSTRUCTIONS]\n{content}")
            elif role == "USER":
                parts.append(f"[TASK]\n{content}")
            else:
                parts.append(content)
        return "\n\n".join(parts)

    async def complete(self, messages: Sequence[Message]) -> str:
        prompt = self._flatten(messages)
        with tempfile.NamedTemporaryFile("r", suffix=".txt", delete=False) as handle:
            out_path = handle.name

        cmd = [
            self.binary,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "read-only",
            "-m",
            self.model,
            "-c",
            f"model_reasoning_effort={self.reasoning_effort}",
            "-o",
            out_path,
            prompt,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout_seconds
                )
            except asyncio.TimeoutError as exc:
                proc.kill()
                raise RuntimeError(
                    f"codex exec timed out after {self.timeout_seconds}s"
                ) from exc

            if proc.returncode != 0:
                raise RuntimeError(
                    f"codex exec failed (exit {proc.returncode}): "
                    f"{stderr.decode(errors='replace')[-500:]}"
                )

            try:
                with open(out_path, "r", encoding="utf-8") as fh:
                    return fh.read().strip()
            except OSError as exc:
                raise RuntimeError(f"codex exec produced no output file: {exc}") from exc
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass


class OpenAILLMClient(LLMClient):
    name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def complete(self, messages: Sequence[Message]) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            temperature=0.3,
        )
        return response.choices[0].message.content or ""


class DryRunLLMClient(LLMClient):
    name = "dry-run"

    def __init__(self, model: str = "dry-run") -> None:
        self.model = model

    async def complete(self, messages: Sequence[Message]) -> str:
        user_text = "\n".join(message["content"] for message in messages if message["role"] == "user")
        if "JSON object" in user_text:
            return (
                '{"goal":"Resolve the current thread clearly.",'
                '"now":"the owner has a new message that may need a response.",'
                '"next":"Send a concise reply that moves the thread forward.",'
                '"open":["Confirm any missing facts before sending."]}'
            )
        return "Got it. I will take a look and follow up with the next step."


def build_llm_client(
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
    timeout_seconds: int | None = None,
) -> LLMClient:
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "codex").lower().strip()

    if provider == "codex":
        return CodexLLMClient(
            model=model or os.getenv("LLM_MODEL", "gpt-5.5"),
            reasoning_effort=reasoning_effort
            or os.getenv("CODEX_REASONING_EFFORT", "high"),
            timeout_seconds=timeout_seconds
            if timeout_seconds is not None
            else _int_env("CODEX_TIMEOUT_SECONDS", 300),
        )

    model = model or os.getenv("LLM_MODEL", "gpt-4.1-mini")
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

    if provider == "openai" and api_key:
        return OpenAILLMClient(api_key=api_key, model=model)
    if provider in {"dry-run", "dryrun"} or not api_key:
        return DryRunLLMClient()
    raise SystemExit(f"Unsupported LLM_PROVIDER={provider!r}.")
