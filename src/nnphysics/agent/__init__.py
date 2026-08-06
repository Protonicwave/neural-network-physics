"""Reading two run reports and saying what changed, and measuring whether it is right.

Building an agent is common. Measuring whether it is right is not, so the deliverable of
this layer is the scored fault injection rather than the agent: a set of faults whose true
cause was written down before anything was asked, and a table saying how often the agent
named it against how often a diagnoser that reads no prose did.

The layer diagram in `docs/plan/00-architecture.md` puts this package and `reporting` at
the same indentation, which leaves the direction between them unsaid. It is resolved here
in the only way that works: an agent whose job is to read reports has to be outside the
layer that writes them, so this package imports `reporting` and `reporting` never imports
this one. `tests/agent/test_layering.py` enforces the rest of the rule, that nothing here
reaches out into the command line layer.
"""

from __future__ import annotations

from nnphysics.agent.causes import CAUSE_DESCRIPTIONS, Cause, cause_catalogue, describe_cause
from nnphysics.agent.client import (
    AgentConfig,
    AgentError,
    AnthropicClient,
    Client,
    RecordedClient,
    Reply,
    ToolSchema,
    Usage,
    load_agent_config,
)
from nnphysics.agent.context import DiagnosisContext, build_context
from nnphysics.agent.diagnose import (
    DIAGNOSIS_TOOL,
    Candidate,
    Diagnosis,
    DiagnosisCost,
    diagnose,
    rule_based_diagnosis,
)
from nnphysics.agent.faults import FAULTS, Fault, Injection, fault, fault_names
from nnphysics.agent.scoring import (
    FaultOutcome,
    ScoreCard,
    SuiteReport,
    render_report,
    score_card,
    score_fault,
)

__all__ = [
    "CAUSE_DESCRIPTIONS",
    "DIAGNOSIS_TOOL",
    "FAULTS",
    "AgentConfig",
    "AgentError",
    "AnthropicClient",
    "Candidate",
    "Cause",
    "Client",
    "Diagnosis",
    "DiagnosisContext",
    "DiagnosisCost",
    "Fault",
    "FaultOutcome",
    "Injection",
    "RecordedClient",
    "Reply",
    "ScoreCard",
    "SuiteReport",
    "ToolSchema",
    "Usage",
    "build_context",
    "cause_catalogue",
    "describe_cause",
    "diagnose",
    "fault",
    "fault_names",
    "load_agent_config",
    "render_report",
    "rule_based_diagnosis",
    "score_card",
    "score_fault",
]
