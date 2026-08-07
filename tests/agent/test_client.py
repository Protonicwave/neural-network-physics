from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import anthropic
import httpx
import pytest

from nnphysics.agent.client import (
    API_KEY_VARIABLES,
    AgentConfig,
    AgentError,
    AnthropicClient,
    RecordedClient,
    ToolSchema,
    Usage,
    load_agent_config,
)
from nnphysics.core.errors import ConfigurationError

TOOL = ToolSchema(
    name="report_diagnosis",
    description="Report a diagnosis.",
    input_schema={
        "type": "object",
        "properties": {"cause": {"type": "string"}},
        "required": ["cause"],
        "additionalProperties": False,
    },
)


class _Stub(AnthropicClient):
    """An `AnthropicClient` whose one call to the SDK is replaced by a recording.

    Subclassed rather than monkeypatched so that everything above the single method that
    touches the network is the real code: the request that gets built, the retry policy,
    and the parsing of the reply.
    """

    def __init__(
        self,
        config: AgentConfig,
        env: Mapping[str, str],
        outcomes: list[Any],
    ) -> None:
        super().__init__(config, env, sleep=self._record_sleep)
        self._outcomes = outcomes
        self.requests: list[Mapping[str, Any]] = []
        self.slept: list[float] = []

    def _record_sleep(self, seconds: float) -> None:
        self.slept.append(seconds)

    def _send(self, request: Mapping[str, Any]) -> Any:
        self.requests.append(dict(request))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _status_error(status: int) -> anthropic.APIStatusError:
    """An SDK status error of a given code, built the way the SDK builds one."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request, json={"error": {"message": "no"}})
    return anthropic.APIStatusError("failed", response=response, body=None)


@pytest.fixture
def env() -> dict[str, str]:
    return {"ANTHROPIC_API_KEY": "not-a-real-key"}


@pytest.fixture
def config() -> AgentConfig:
    return AgentConfig(model="claude-recorded-1", max_tokens=1024, backoff_seconds=0.5)


@pytest.fixture(autouse=True)
def _sdk_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SDK reads the environment itself when a client is constructed.

    Set to an obvious non key. Nothing here reaches the network, and a test that needed a
    real one would be an integration test.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")


class TestAgentConfig:
    def test_the_model_has_no_default(self) -> None:
        """A default model identifier is a hardcoded model identifier by another name."""
        with pytest.raises(Exception, match="model"):
            AgentConfig()  # type: ignore[call-arg]

    def test_temperature_is_unset_by_default(self, config: AgentConfig) -> None:
        """Recent models reject the parameter, so sending one unasked fails every call."""
        assert config.temperature is None

    def test_unknown_keys_are_refused(self) -> None:
        with pytest.raises(Exception, match="extra"):
            AgentConfig(model="m", nonsense=1)  # type: ignore[call-arg]


class TestUsage:
    def test_cost_is_derived_from_the_configured_prices(self) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=200_000)
        config = AgentConfig(model="m", input_price=5.0, output_price=25.0)

        assert usage.cost(config) == pytest.approx(5.0 + 5.0)

    def test_no_prices_means_no_cost_rather_than_no_tokens(self) -> None:
        """The token counts are the measurement. A price only turns them into money."""
        usage = Usage(input_tokens=10, output_tokens=10)

        assert usage.cost(AgentConfig(model="m")) == 0.0
        assert usage.input_tokens == 10

    def test_usages_add(self) -> None:
        total = Usage(1, 2) + Usage(10, 20)

        assert (total.input_tokens, total.output_tokens) == (11, 22)


class TestToolSchema:
    def test_the_payload_asks_for_strict_validation(self) -> None:
        """Strict is what makes the API reject a reply of the wrong shape, not this code."""
        payload = TOOL.as_payload()

        assert payload["strict"] is True
        assert payload["input_schema"]["additionalProperties"] is False


class TestCredential:
    def test_a_missing_credential_is_a_sentence_rather_than_a_stack_trace(self) -> None:
        with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
            AnthropicClient(AgentConfig(model="m"), {})

    def test_the_error_names_every_variable_the_sdk_would_look_at(self) -> None:
        with pytest.raises(ConfigurationError) as caught:
            AnthropicClient(AgentConfig(model="m"), {})

        for name in API_KEY_VARIABLES:
            assert name in str(caught.value)

    def test_no_credential_reaches_the_error(self) -> None:
        """A message that quoted the key would put it in whatever caught the failure."""
        with pytest.raises(ConfigurationError) as caught:
            AnthropicClient(AgentConfig(model="m"), {"UNRELATED": "secret-value"})

        assert "secret-value" not in str(caught.value)


class TestCall:
    def test_a_recorded_reply_is_parsed_into_arguments_and_usage(
        self, config: AgentConfig, env: dict[str, str], recorded_message: Callable[..., Any]
    ) -> None:
        client = _Stub(config, env, [recorded_message()])

        reply = client.call(system="s", prompt="p", tool=TOOL)

        assert reply.arguments["candidates"][0]["cause"] == "rollout_curriculum"
        assert (reply.usage.input_tokens, reply.usage.output_tokens) == (2431, 318)
        assert reply.model == "claude-recorded-1"
        assert reply.attempts == 1

    def test_the_request_forces_the_tool(
        self, config: AgentConfig, env: dict[str, str], recorded_message: Callable[..., Any]
    ) -> None:
        client = _Stub(config, env, [recorded_message()])

        client.call(system="s", prompt="p", tool=TOOL)

        assert client.requests[0]["tool_choice"] == {"type": "tool", "name": TOOL.name}
        assert client.requests[0]["model"] == "claude-recorded-1"

    def test_no_temperature_is_sent_when_none_is_configured(
        self, config: AgentConfig, env: dict[str, str], recorded_message: Callable[..., Any]
    ) -> None:
        client = _Stub(config, env, [recorded_message()])

        client.call(system="s", prompt="p", tool=TOOL)

        assert "temperature" not in client.requests[0]

    def test_a_configured_temperature_is_sent(
        self, env: dict[str, str], recorded_message: Callable[..., Any]
    ) -> None:
        config = AgentConfig(model="m", temperature=0.2)
        client = _Stub(config, env, [recorded_message()])

        client.call(system="s", prompt="p", tool=TOOL)

        assert client.requests[0]["temperature"] == 0.2

    def test_a_rate_limit_is_retried_with_a_doubling_backoff(
        self, config: AgentConfig, env: dict[str, str], recorded_message: Callable[..., Any]
    ) -> None:
        client = _Stub(config, env, [_status_error(429), _status_error(429), recorded_message()])

        reply = client.call(system="s", prompt="p", tool=TOOL)

        assert reply.attempts == 3
        assert client.slept == [0.5, 1.0]

    def test_a_server_error_is_retried_and_a_bad_request_is_not(
        self, config: AgentConfig, env: dict[str, str], recorded_message: Callable[..., Any]
    ) -> None:
        """The distinction is the whole point of a retry policy: 500 is theirs, 400 ours."""
        retried = _Stub(config, env, [_status_error(503), recorded_message()])
        assert retried.call(system="s", prompt="p", tool=TOOL).attempts == 2

        refused = _Stub(config, env, [_status_error(400)])
        with pytest.raises(AgentError, match="could not be reached"):
            refused.call(system="s", prompt="p", tool=TOOL)
        assert refused.slept == []

    def test_running_out_of_attempts_says_how_many_were_made(
        self, env: dict[str, str], recorded_message: Callable[..., Any]
    ) -> None:
        config = AgentConfig(model="m", max_attempts=2, backoff_seconds=0.1)
        client = _Stub(config, env, [_status_error(429), _status_error(429), recorded_message()])

        with pytest.raises(AgentError, match="after 2 attempts"):
            client.call(system="s", prompt="p", tool=TOOL)

    def test_a_refusal_is_reported_rather_than_parsed(
        self, config: AgentConfig, env: dict[str, str], recorded_message: Callable[..., Any]
    ) -> None:
        message = recorded_message()
        message.stop_reason = "refusal"
        client = _Stub(config, env, [message])

        with pytest.raises(AgentError, match="declined"):
            client.call(system="s", prompt="p", tool=TOOL)

    def test_running_out_of_tokens_is_reported_rather_than_parsed(
        self, config: AgentConfig, env: dict[str, str], recorded_message: Callable[..., Any]
    ) -> None:
        """A truncated reply can still carry a tool block, and it is not a diagnosis."""
        message = recorded_message()
        message.stop_reason = "max_tokens"
        client = _Stub(config, env, [message])

        with pytest.raises(AgentError, match="token budget"):
            client.call(system="s", prompt="p", tool=TOOL)

    def test_an_answer_without_the_tool_is_reported(
        self, config: AgentConfig, env: dict[str, str], recorded_message: Callable[..., Any]
    ) -> None:
        message = recorded_message()
        message.content = [block for block in message.content if block.type != "tool_use"]
        client = _Stub(config, env, [message])

        with pytest.raises(AgentError, match="without calling"):
            client.call(system="s", prompt="p", tool=TOOL)

    def test_a_thinking_block_before_the_tool_is_stepped_over(
        self, config: AgentConfig, env: dict[str, str], recorded_message: Callable[..., Any]
    ) -> None:
        """The recording begins with a thinking block, which the first block is not."""
        message = recorded_message()
        assert message.content[0].type != "tool_use"

        client = _Stub(config, env, [message])

        assert client.call(system="s", prompt="p", tool=TOOL).arguments


class TestRecordedClient:
    def test_replies_come_back_in_order(self, recorded_reply: Callable[..., Any]) -> None:
        first, second = recorded_reply(), recorded_reply()
        client = RecordedClient([first, second])

        assert client.call(system="s", prompt="p", tool=TOOL) is first
        assert client.call(system="s", prompt="p", tool=TOOL) is second

    def test_running_out_is_an_error_rather_than_a_repeat(
        self, recorded_reply: Callable[..., Any]
    ) -> None:
        """A test that made more calls than it recorded is a test with a stale fixture."""
        client = RecordedClient([recorded_reply()])
        client.call(system="s", prompt="p", tool=TOOL)

        with pytest.raises(AgentError, match="no reply left"):
            client.call(system="s", prompt="p", tool=TOOL)

    def test_the_calls_are_kept_so_a_test_can_assert_on_the_prompt(
        self, recorded_reply: Callable[..., Any]
    ) -> None:
        client = RecordedClient([recorded_reply()])

        client.call(system="the system prompt", prompt="the user prompt", tool=TOOL)

        assert client.calls == (("the system prompt", "the user prompt", TOOL.name),)


class TestLoadAgentConfig:
    def test_a_file_is_read_and_validated(self, tmp_path: Path) -> None:
        path = tmp_path / "agent.yaml"
        path.write_text("model: claude-recorded-1\nmax_tokens: 512\n", encoding="utf-8")

        config = load_agent_config(path)

        assert (config.model, config.max_tokens) == ("claude-recorded-1", 512)

    def test_the_environment_overrides_a_key(self, tmp_path: Path) -> None:
        path = tmp_path / "agent.yaml"
        path.write_text("model: from-the-file\n", encoding="utf-8")

        config = load_agent_config(path, {"NNP_AGENT__MODEL": "from-the-environment"})

        assert config.model == "from-the-environment"

    def test_a_missing_file_is_a_configuration_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="cannot read"):
            load_agent_config(tmp_path / "absent.yaml")

    def test_a_file_that_is_not_a_mapping_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "agent.yaml"
        path.write_text("- a list\n", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="mapping"):
            load_agent_config(path)

    def test_the_shipped_configuration_loads(self) -> None:
        """The file the command line defaults to has to be one the loader accepts."""
        config = load_agent_config(Path("configs/agent.yaml"))

        assert config.model
        assert config.prices_cached, "a price with no date beside it cannot be audited"
