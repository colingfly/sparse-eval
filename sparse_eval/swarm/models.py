"""
models.py -- Data model for swarm (multi-agent) evaluation.

Defines the core types for representing agent swarms, interaction traces,
test cases, and evaluation results for multi-agent orchestration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SwarmTopologyType(Enum):
    """How agents in the swarm are connected."""
    ORCHESTRATOR = "orchestrator"    # Hub-and-spoke: one orchestrator + N specialists
    COLLABORATIVE = "collaborative"  # Peer-to-peer: agents communicate freely


class SwarmMessageRole(Enum):
    """Role of a message in the interaction trace."""
    USER = "user"                    # External input
    ORCHESTRATOR = "orchestrator"    # Routing/delegation message
    AGENT = "agent"                  # Specialist/peer response
    HANDOFF = "handoff"              # Explicit handoff between agents
    FINAL = "final"                  # Final consolidated answer


@dataclass
class SwarmAgent:
    """Definition of a single agent in the swarm."""
    id: str
    role: str
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    # Target config for orchestrated mode (how to call this agent)
    type: str = "callable"  # callable | http | command
    url: Optional[str] = None
    method: str = "POST"
    headers: dict = field(default_factory=dict)
    payload_template: str = '{"message": "{{message}}"}'
    response_path: str = "response"
    command: Optional[str] = None  # for CLI agents

    def __repr__(self):
        return f"SwarmAgent({self.id!r}, role={self.role!r})"


@dataclass
class SwarmTopology:
    """Defines how agents are connected in the swarm."""
    type: SwarmTopologyType
    agents: list[SwarmAgent] = field(default_factory=list)
    orchestrator_id: Optional[str] = None  # Only for ORCHESTRATOR topology
    connections: dict[str, list[str]] = field(default_factory=dict)

    def get_agent(self, agent_id: str) -> Optional[SwarmAgent]:
        """Look up an agent by ID."""
        for a in self.agents:
            if a.id == agent_id:
                return a
        return None

    @property
    def agent_ids(self) -> list[str]:
        return [a.id for a in self.agents]

    def __repr__(self):
        return f"SwarmTopology({self.type.value}, {len(self.agents)} agents)"


@dataclass
class SwarmMessage:
    """A single message in the interaction trace."""
    from_agent: str = ""        # agent_id or "user"
    to_agent: str = ""          # agent_id or "user"
    role: SwarmMessageRole = SwarmMessageRole.AGENT
    content: str = ""
    timestamp_ms: int = 0
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        preview = self.content[:40] + "..." if len(self.content) > 40 else self.content
        return f"SwarmMessage({self.from_agent}->{self.to_agent}, {self.role.value}, {preview!r})"


@dataclass
class SwarmTrace:
    """Complete interaction trace for one test scenario."""
    messages: list[SwarmMessage] = field(default_factory=list)
    final_answer: str = ""
    total_time_ms: int = 0

    @property
    def agents_involved(self) -> list[str]:
        """Unique agent IDs that participated (excluding 'user')."""
        seen = []
        for m in self.messages:
            for aid in (m.from_agent, m.to_agent):
                if aid and aid != "user" and aid not in seen:
                    seen.append(aid)
        return seen

    @property
    def routing_decisions(self) -> list[dict]:
        """Extract routing decisions (handoff messages)."""
        return [
            {
                "from": m.from_agent,
                "to": m.to_agent,
                "content": m.content,
                "metadata": m.metadata,
            }
            for m in self.messages
            if m.role == SwarmMessageRole.HANDOFF
        ]

    @property
    def actual_route(self) -> list[str]:
        """Ordered list of agent IDs in the routing path."""
        route = []
        for m in self.messages:
            if m.from_agent and m.from_agent != "user" and (not route or route[-1] != m.from_agent):
                route.append(m.from_agent)
        return route

    @property
    def num_handoffs(self) -> int:
        return sum(1 for m in self.messages if m.role == SwarmMessageRole.HANDOFF)

    @property
    def num_turns(self) -> int:
        return len(self.messages)

    def messages_by_agent(self, agent_id: str) -> list[SwarmMessage]:
        return [m for m in self.messages if m.from_agent == agent_id]

    def to_transcript(self) -> str:
        """Render trace as human-readable transcript."""
        lines = []
        for m in self.messages:
            role_tag = f"[{m.role.value}]" if m.role != SwarmMessageRole.AGENT else ""
            lines.append(f"{m.from_agent} -> {m.to_agent} {role_tag}: {m.content}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "messages": [
                {
                    "from": m.from_agent,
                    "to": m.to_agent,
                    "role": m.role.value,
                    "content": m.content,
                    "timestamp_ms": m.timestamp_ms,
                    "metadata": m.metadata,
                }
                for m in self.messages
            ],
            "final_answer": self.final_answer,
            "total_time_ms": self.total_time_ms,
            "agents_involved": self.agents_involved,
            "num_handoffs": self.num_handoffs,
            "num_turns": self.num_turns,
        }


@dataclass
class SwarmTestCase:
    """A single swarm evaluation scenario."""
    question: str
    expected_answer: Optional[str] = None
    expected_route: Optional[list[str]] = None    # Expected agent routing path
    expected_agents: Optional[list[str]] = None   # Which agents should participate
    max_turns: Optional[int] = None               # Convergence limit
    context: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    trace: Optional[SwarmTrace] = None            # Pre-recorded trace (trace mode)
    trace_file: Optional[str] = None              # Path to external trace file

    def __repr__(self):
        q = self.question[:50] + "..." if len(self.question) > 50 else self.question
        return f"SwarmTestCase({q!r})"


@dataclass
class SwarmResult:
    """Result for a single swarm test case."""
    test_case: SwarmTestCase
    trace: SwarmTrace
    outcome_score: float = 0.0         # Final answer quality (0-100)
    process_score: float = 0.0         # Overall process quality (0-100)
    routing_score: float = 0.0         # Routing correctness (0-100)
    handoff_score: float = 0.0         # Handoff quality (0-100)
    context_score: float = 0.0         # Context preservation (0-100)
    convergence_score: float = 0.0     # Convergence efficiency (0-100)
    per_agent_scores: dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    status: str = "pending"            # pass | warn | fail
    reasoning: str = ""

    @classmethod
    def classify(cls, score: float) -> str:
        if score >= 80:
            return "pass"
        if score >= 60:
            return "warn"
        return "fail"

    def to_dict(self) -> dict:
        return {
            "question": self.test_case.question,
            "expected_answer": self.test_case.expected_answer or "",
            "final_answer": self.trace.final_answer,
            "overall_score": self.overall_score,
            "status": self.status,
            "outcome_score": self.outcome_score,
            "process_score": self.process_score,
            "process_breakdown": {
                "routing": self.routing_score,
                "handoff": self.handoff_score,
                "context_preservation": self.context_score,
                "convergence": self.convergence_score,
            },
            "per_agent_scores": self.per_agent_scores,
            "reasoning": self.reasoning,
            "trace": self.trace.to_dict(),
            "tags": self.test_case.tags,
        }


@dataclass
class SwarmSummary:
    """Summary of a complete swarm eval run."""
    suite_name: str
    topology_type: str = ""
    total: int = 0
    passed: int = 0
    warned: int = 0
    failed: int = 0
    overall_score: float = 0.0
    outcome_avg: float = 0.0
    process_avg: float = 0.0
    results: list[SwarmResult] = field(default_factory=list)
    per_agent_summary: dict[str, dict] = field(default_factory=dict)
    reliability_score: Optional[float] = None
    reliability_grade: str = ""
    duration_seconds: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total else 0.0

    def failures(self) -> list[SwarmResult]:
        return [r for r in self.results if r.status == "fail"]

    def warnings(self) -> list[SwarmResult]:
        return [r for r in self.results if r.status == "warn"]

    def by_tag(self, tag: str) -> list[SwarmResult]:
        return [r for r in self.results if tag in r.test_case.tags]

    def to_dict(self) -> dict:
        d = {
            "suite_name": self.suite_name,
            "topology_type": self.topology_type,
            "total": self.total,
            "passed": self.passed,
            "warned": self.warned,
            "failed": self.failed,
            "overall_score": self.overall_score,
            "outcome_avg": self.outcome_avg,
            "process_avg": self.process_avg,
            "pass_rate": round(self.pass_rate, 1),
            "per_agent_summary": self.per_agent_summary,
            "duration_seconds": round(self.duration_seconds, 1),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "results": [r.to_dict() for r in self.results],
        }
        if self.reliability_score is not None:
            d["reliability_score"] = self.reliability_score
            d["reliability_grade"] = self.reliability_grade
        return d
