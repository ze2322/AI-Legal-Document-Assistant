"""Answer generation through a locally hosted Ollama model."""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import ollama

from services.exceptions import LLMUnavailableError
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an experienced legal AI assistant.

You answer questions ONLY from the document context supplied by the user.

Rules:
- Base every statement on the provided context; never invent facts, clauses or
  figures that are not there.
- Give a detailed, well-structured answer in your own words.
- Use bullet points when the answer covers several distinct points.
- Refer to the relevant chunk numbers when it helps the reader locate the source.
- If the context does not contain the answer, reply exactly:
  "I couldn't find this information in the uploaded document."
"""

USER_PROMPT_TEMPLATE = """Context:
-------------------------
{context}
-------------------------

Question: {question}

Answer:"""

# Errors raised when Ollama is unreachable or rejects the request.
#
# The non-streaming path wraps transport failures in ConnectionError, but the
# streaming path lets httpx errors escape as-is, so httpx.HTTPError (the base
# of ConnectError, ReadTimeout and friends) has to be handled explicitly.
_CONNECTION_ERRORS = (
    ollama.ResponseError,
    ollama.RequestError,
    httpx.HTTPError,
    ConnectionError,
    OSError,
)


class LLMService:
    """Thin wrapper around the Ollama chat API.

    The host is configurable so the app works both locally and from inside a
    container, where Ollama runs on the host rather than on localhost.
    """

    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        self.model = model or settings.ollama_model
        self.host = host if host is not None else settings.ollama_host
        self.client = ollama.Client(host=self.host, timeout=settings.llm_timeout_seconds)

    def _build_messages(self, context: str, question: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(context=context, question=question),
            },
        ]

    @property
    def _options(self) -> dict[str, float | int]:
        return {
            "num_predict": settings.llm_max_tokens,
            "temperature": settings.llm_temperature,
        }

    def is_available(self) -> bool:
        """Check that the Ollama server is reachable and the model is pulled."""
        try:
            models = self.client.list().get("models", [])
        except _CONNECTION_ERRORS as exc:
            logger.warning("Ollama is not reachable: %s", exc)
            return False

        # Ollama reports names as "llama3.2:latest"; match on the base name.
        available = {str(m.get("model", "")).split(":")[0] for m in models}
        return self.model.split(":")[0] in available

    def generate(self, context: str, question: str) -> str:
        """Generate a complete answer from the retrieved context.

        Raises:
            LLMUnavailableError: If Ollama cannot be reached or fails.
        """
        try:
            response = self.client.chat(
                model=self.model,
                messages=self._build_messages(context, question),
                options=self._options,
            )
        except _CONNECTION_ERRORS as exc:
            raise LLMUnavailableError(self._unavailable_message(exc)) from exc

        return response["message"]["content"]

    def generate_stream(self, context: str, question: str) -> Iterator[str]:
        """Yield the answer token by token as Ollama produces it.

        Raises:
            LLMUnavailableError: If Ollama cannot be reached or fails mid-stream.
        """
        try:
            stream = self.client.chat(
                model=self.model,
                messages=self._build_messages(context, question),
                options=self._options,
                stream=True,
            )
            for part in stream:
                token = part.get("message", {}).get("content", "")
                if token:
                    yield token
        except _CONNECTION_ERRORS as exc:
            raise LLMUnavailableError(self._unavailable_message(exc)) from exc

    def _unavailable_message(self, exc: Exception) -> str:
        host = self.host or "http://localhost:11434"
        return (
            f"Could not reach the Ollama server at {host} ({exc}). "
            f"Start it with 'ollama serve' and make sure the model is available "
            f"with 'ollama pull {self.model}'."
        )
