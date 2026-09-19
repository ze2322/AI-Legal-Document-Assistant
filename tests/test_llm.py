"""Tests for LLMService.

Ollama is never contacted: the client is replaced with a mock so the suite runs
without a local model server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import ollama
import pytest

from services.exceptions import LLMUnavailableError
from services.llm import LLMService


@pytest.fixture
def client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def llm(client: MagicMock) -> LLMService:
    with patch("services.llm.ollama.Client", return_value=client):
        return LLMService(model="test-model", host="http://localhost:11434")


def test_generate_returns_the_message_content(llm: LLMService, client: MagicMock) -> None:
    client.chat.return_value = {"message": {"content": "Partners must retain records."}}

    assert llm.generate("some context", "a question") == "Partners must retain records."


def test_generate_sends_context_and_question(llm: LLMService, client: MagicMock) -> None:
    client.chat.return_value = {"message": {"content": "ok"}}

    llm.generate("RETRIEVED CONTEXT", "THE QUESTION")

    messages = client.chat.call_args.kwargs["messages"]
    user_message = messages[-1]["content"]

    assert messages[0]["role"] == "system"
    assert "RETRIEVED CONTEXT" in user_message
    assert "THE QUESTION" in user_message


def test_generate_applies_configured_options(llm: LLMService, client: MagicMock) -> None:
    from utils.config import settings

    client.chat.return_value = {"message": {"content": "ok"}}
    llm.generate("context", "question")

    options = client.chat.call_args.kwargs["options"]

    assert options["temperature"] == settings.llm_temperature
    assert options["num_predict"] == settings.llm_max_tokens


def test_generate_stream_yields_tokens(llm: LLMService, client: MagicMock) -> None:
    client.chat.return_value = iter(
        [
            {"message": {"content": "Partners "}},
            {"message": {"content": "must "}},
            {"message": {"content": "retain records."}},
        ]
    )

    assert "".join(llm.generate_stream("context", "question")) == "Partners must retain records."


def test_generate_stream_skips_empty_tokens(llm: LLMService, client: MagicMock) -> None:
    client.chat.return_value = iter(
        [{"message": {"content": ""}}, {"message": {}}, {"message": {"content": "text"}}]
    )

    assert list(llm.generate_stream("context", "question")) == ["text"]


def test_connection_error_becomes_a_readable_error(llm: LLMService, client: MagicMock) -> None:
    client.chat.side_effect = ConnectionError("connection refused")

    with pytest.raises(LLMUnavailableError, match="ollama serve"):
        llm.generate("context", "question")


def test_response_error_becomes_a_readable_error(llm: LLMService, client: MagicMock) -> None:
    client.chat.side_effect = ollama.ResponseError("model not found")

    with pytest.raises(LLMUnavailableError, match="ollama pull test-model"):
        llm.generate("context", "question")


def test_stream_failure_becomes_a_readable_error(llm: LLMService, client: MagicMock) -> None:
    client.chat.side_effect = ConnectionError("connection refused")

    with pytest.raises(LLMUnavailableError):
        list(llm.generate_stream("context", "question"))


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("connection refused"),
        httpx.ReadTimeout("timed out"),
        ollama.RequestError("bad request"),
    ],
)
def test_httpx_transport_errors_are_wrapped(
    llm: LLMService, client: MagicMock, error: Exception
) -> None:
    """The streaming path lets httpx errors escape instead of wrapping them in
    ConnectionError, so they must be handled explicitly."""
    client.chat.side_effect = error

    with pytest.raises(LLMUnavailableError, match="ollama serve"):
        list(llm.generate_stream("context", "question"))

    with pytest.raises(LLMUnavailableError, match="ollama serve"):
        llm.generate("context", "question")


def test_is_available_matches_tagged_model_names(llm: LLMService, client: MagicMock) -> None:
    """Ollama reports 'test-model:latest' for a model pulled as 'test-model'."""
    client.list.return_value = {"models": [{"model": "test-model:latest"}]}

    assert llm.is_available() is True


def test_is_available_false_when_model_missing(llm: LLMService, client: MagicMock) -> None:
    client.list.return_value = {"models": [{"model": "some-other-model:latest"}]}

    assert llm.is_available() is False


def test_is_available_false_when_server_is_down(llm: LLMService, client: MagicMock) -> None:
    client.list.side_effect = ConnectionError("connection refused")

    assert llm.is_available() is False


def test_host_is_passed_to_the_client() -> None:
    """The host must be configurable so the app works from inside a container."""
    with patch("services.llm.ollama.Client") as client_factory:
        LLMService(model="test-model", host="http://host.docker.internal:11434")

    assert client_factory.call_args.kwargs["host"] == "http://host.docker.internal:11434"
