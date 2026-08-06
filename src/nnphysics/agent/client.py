"""A thin wrapper over the Anthropic SDK.

Thin on purpose. Everything the diagnosis needs from a language model is one call that
takes a system prompt, a user prompt and a tool schema, and returns the arguments the
model passed to that tool. Forcing the answer through a tool is what makes it structured:
the schema is validated by the API rather than by a regular expression over prose, and a
reply that does not fit it never reaches this package.

Three things are deliberate.

The model identifier is a required setting with no default in the source. A default would
be a model identifier hardcoded in a repository that outlives it, and the one thing
certain about a model identifier is that it changes.

No key is read from a file, written to a record or printed. The SDK resolves the
credential from the environment, this module only checks that one is there so that the
failure is a sentence rather than an authentication error from three layers down, and
nothing that gets serialised ever holds it.

Token counts are recorded and the cost is derived from configured prices. Tokens are what
the API measured; a price is a fact about a pricing page on a particular day, so it is
configuration with its own cached date rather than a constant in the code.
"""

from __future__ import annotations

import time as _time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import anthropic
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nnphysics.core.errors import ConfigurationError, NNPhysicsError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = [
    "API_KEY_VARIABLES",
    "AgentConfig",
    "AgentError",
    "AnthropicClient",
    "Client",
    "RecordedClient",
    "Reply",
    "ToolSchema",
    "Usage",
    "load_agent_config",
]

API_KEY_VARIABLES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
"""Environment variables the SDK resolves a credential from. Checked by name only: the
value is never read into a variable this module keeps, never logged and never recorded."""

_TOKENS_PER_MILLION = 1.0e6

_SERVER_ERROR = 500
"""At or above this a status code is the service's problem and is worth retrying."""

_RETRYABLE_CODES = frozenset({408, 409, 429})
"""Status codes below a server error that are still worth retrying: a timeout, a conflict
and a rate limit. Matched on the code rather than on the exception class, because the class
depends on which SDK version raised it and the code does not."""


class AgentError(NNPhysicsError):
    """The model could not be reached, or answered in a shape that cannot be used."""


class AgentConfig(BaseModel):
    """How the diagnosis agent calls the model.

    Not part of `RunConfig`. A run identifier is a hash of the configuration that produced
    the weights, and which model was later asked to explain them changes nothing about
    them. Folding this in would move the identifier of every run that already exists.

    Attributes:
        model: Model identifier. Required, because a default here is a hardcoded model
            identifier by another name.
        max_tokens: Largest reply the model may produce, thinking included.
        temperature: Sampling temperature, or `None` to send none. `None` by default:
            recent models reject the parameter outright, so sending one unasked would
            fail every request against them.
        timeout_seconds: How long one request may take.
        max_attempts: Attempts a retryable failure is given, the first one included.
        backoff_seconds: Delay before the second attempt, doubling after each failure.
        input_price: Dollars per million input tokens.
        output_price: Dollars per million output tokens.
        prices_cached: When the prices above were last checked, as an ISO 8601 date. A
            price with no date beside it is a number nobody can audit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(min_length=1)
    max_tokens: int = Field(default=16000, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    timeout_seconds: float = Field(default=120.0, gt=0.0)
    max_attempts: int = Field(default=4, ge=1)
    backoff_seconds: float = Field(default=1.0, gt=0.0)
    input_price: float = Field(default=0.0, ge=0.0)
    output_price: float = Field(default=0.0, ge=0.0)
    prices_cached: str = ""


@dataclass(frozen=True, slots=True)
class Usage:
    """What one call cost, in tokens.

    Attributes:
        input_tokens: Tokens the prompt occupied.
        output_tokens: Tokens the reply occupied, thinking included.
    """

    input_tokens: int
    output_tokens: int

    def cost(self, config: AgentConfig) -> float:
        """Dollars this call cost at the configured prices.

        Args:
            config: The settings the call was made under.

        Returns:
            The cost. Zero when no prices are configured, which reads as unpriced rather
            than free: the token counts are beside it and say what was actually spent.
        """
        return (
            self.input_tokens * config.input_price + self.output_tokens * config.output_price
        ) / _TOKENS_PER_MILLION

    def __add__(self, other: Usage) -> Usage:
        """Add two usages, so a suite can total what it spent."""
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """The tool a reply is forced through.

    Attributes:
        name: Tool name.
        description: What the tool is for, which the model reads.
        input_schema: JSON Schema the arguments must satisfy. Strict validation needs
            `additionalProperties: false` and a `required` list, and the constructor of
            the schema is responsible for both.
    """

    name: str
    description: str
    input_schema: Mapping[str, Any]

    def as_payload(self) -> dict[str, Any]:
        """The tool as the API expects it, with strict validation switched on."""
        return {
            "name": self.name,
            "description": self.description,
            "strict": True,
            "input_schema": dict(self.input_schema),
        }


@dataclass(frozen=True, slots=True)
class Reply:
    """One answer.

    Attributes:
        arguments: The arguments the model passed to the tool.
        usage: What the call cost in tokens.
        model: Model that answered, as the API reported it.
        attempts: Requests made, the successful one included.
    """

    arguments: Mapping[str, Any]
    usage: Usage
    model: str
    attempts: int


class Client(Protocol):
    """Anything that can answer one structured question."""

    @property
    def model(self) -> str:
        """Model identifier this client asks for."""

    def call(self, *, system: str, prompt: str, tool: ToolSchema) -> Reply:
        """Ask once and return the arguments the model passed to the tool.

        Args:
            system: System prompt.
            prompt: User prompt.
            tool: The tool the answer is forced through.

        Returns:
            The reply.

        Raises:
            AgentError: If the model could not be reached, or answered without calling
                the tool.
        """


class AnthropicClient:
    """Calls the real API.

    Args:
        config: The settings to call under.
        env: Environment the credential is resolved from. Checked for a variable the SDK
            knows about; the value is never read.
        sleep: How to wait between attempts. Injected so a test can retry instantly.

    Raises:
        ConfigurationError: If no credential is present in the environment.
    """

    def __init__(
        self,
        config: AgentConfig,
        env: Mapping[str, str],
        *,
        sleep: Callable[[float], None] = _time.sleep,
    ) -> None:
        if not any(env.get(name) for name in API_KEY_VARIABLES):
            raise ConfigurationError(
                f"no Anthropic credential in the environment: set one of "
                f"{', '.join(API_KEY_VARIABLES)}. Nothing is read from the repository."
            )
        self._config = config
        self._sleep = sleep
        # The SDK retries on its own by default. Turned off, because the backoff and the
        # attempt count are configuration here and two retry policies stacked on each
        # other would make the recorded number of attempts a fiction.
        self._client = anthropic.Anthropic(timeout=config.timeout_seconds, max_retries=0)

    @property
    def model(self) -> str:
        """Model identifier this client asks for."""
        return self._config.model

    def call(self, *, system: str, prompt: str, tool: ToolSchema) -> Reply:
        """Ask once, retrying a failure that is the service's rather than the request's.

        Args:
            system: System prompt.
            prompt: User prompt.
            tool: The tool the answer is forced through.

        Returns:
            The reply.

        Raises:
            AgentError: If every attempt failed, or the model answered without calling
                the tool, or it declined, or it ran out of tokens before finishing.
        """
        request: dict[str, Any] = {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [tool.as_payload()],
            "tool_choice": {"type": "tool", "name": tool.name},
        }
        if self._config.temperature is not None:
            request["temperature"] = self._config.temperature

        delay = self._config.backoff_seconds
        last: Exception | None = None
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                message = self._send(request)
            except Exception as error:
                if not self._retryable(error):
                    raise AgentError(f"the model could not be reached: {error}") from error
                last = error
                if attempt == self._config.max_attempts:
                    break
                self._sleep(delay)
                delay *= 2.0
                continue
            return self._reply(message, tool, attempt)
        raise AgentError(
            f"the model could not be reached after {self._config.max_attempts} attempts: {last}"
        )

    def _send(self, request: Mapping[str, Any]) -> Any:  # noqa: ANN401 - an SDK message
        """Make one request.

        The only place the SDK is called from, which is what lets a test replace it with a
        recorded response and exercise the retry policy and the parsing without a network.

        Args:
            request: The request body.

        Returns:
            Whatever the SDK returned.
        """
        return self._client.messages.create(**request)

    def _retryable(self, error: Exception) -> bool:
        """Whether a failure is the service's problem rather than the request's."""
        if isinstance(error, anthropic.APIConnectionError):
            return True
        return isinstance(error, anthropic.APIStatusError) and (
            error.status_code >= _SERVER_ERROR or error.status_code in _RETRYABLE_CODES
        )

    def _reply(self, message: Any, tool: ToolSchema, attempts: int) -> Reply:  # noqa: ANN401
        """Pull the tool arguments out of a message, or say why there are none."""
        if message.stop_reason == "refusal":
            raise AgentError("the model declined to answer this diagnosis request")
        if message.stop_reason == "max_tokens":
            raise AgentError(
                f"the model ran out of its {self._config.max_tokens} token budget before "
                f"it finished the diagnosis"
            )
        for block in message.content:
            if getattr(block, "type", "") == "tool_use" and block.name == tool.name:
                return Reply(
                    arguments=dict(block.input),
                    usage=Usage(
                        input_tokens=int(message.usage.input_tokens),
                        output_tokens=int(message.usage.output_tokens),
                    ),
                    model=str(message.model),
                    attempts=attempts,
                )
        raise AgentError(
            f"the model answered without calling {tool.name!r}, so there is no structured "
            f"diagnosis to read"
        )


class RecordedClient:
    """Replays recorded replies, so a test can exercise everything but the network.

    Replies are returned in order, one per call. Running out is an error rather than a
    repeat of the last one: a test that made more calls than it recorded is a test whose
    fixture no longer describes it.

    Args:
        replies: The replies, in the order they will be returned.
        model: Model identifier to report.
    """

    def __init__(self, replies: Sequence[Reply], *, model: str = "recorded") -> None:
        self._replies = list(replies)
        self._model = model
        self._calls: list[tuple[str, str, str]] = []

    @property
    def model(self) -> str:
        """Model identifier this client reports."""
        return self._model

    @property
    def calls(self) -> tuple[tuple[str, str, str], ...]:
        """Every call made, as system prompt, user prompt and tool name."""
        return tuple(self._calls)

    def call(self, *, system: str, prompt: str, tool: ToolSchema) -> Reply:
        """Return the next recorded reply.

        Args:
            system: System prompt, recorded so a test can assert on it.
            prompt: User prompt, recorded for the same reason.
            tool: The tool the answer would have been forced through.

        Returns:
            The next reply.

        Raises:
            AgentError: If every recorded reply has been used.
        """
        self._calls.append((system, prompt, tool.name))
        if not self._replies:
            raise AgentError(f"this recorded client has no reply left for call {len(self._calls)}")
        return self._replies.pop(0)


def load_agent_config(path: Path, env: Mapping[str, str] | None = None) -> AgentConfig:
    """Read the agent settings from a YAML file.

    Args:
        path: Path to the YAML file.
        env: Environment to take `NNP_AGENT__KEY` overrides from, or `None` for none. The
            same convention the run configuration uses, so one habit covers both.

    Returns:
        The validated, frozen settings.

    Raises:
        ConfigurationError: If the file is missing, is not a YAML mapping, or fails
            validation.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(f"cannot read agent configuration {path}: {error}") from error
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigurationError(f"{path} is not valid YAML: {error}") from error
    if not isinstance(loaded, dict):
        raise ConfigurationError(
            f"{path} must contain a mapping at the top level, got {type(loaded).__name__}"
        )
    raw = dict(loaded)
    for variable, value in sorted((env or {}).items()):
        prefix = "NNP_AGENT__"
        if variable.startswith(prefix) and len(variable) > len(prefix):
            raw[variable[len(prefix) :].lower()] = yaml.safe_load(value)
    try:
        return AgentConfig.model_validate(raw)
    except ValidationError as error:
        raise ConfigurationError(f"invalid agent configuration in {path}: {error}") from error
