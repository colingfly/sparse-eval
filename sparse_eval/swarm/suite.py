"""
suite.py -- YAML suite loader for swarm evaluation.

Loads swarm test suites from YAML files. Each suite defines:
- A topology (agents, connections, type)
- Judge criteria with weights
- Custom rules
- Test cases with questions, expected answers, and traces
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sparse_eval.swarm.models import (
    SwarmAgent,
    SwarmTopology,
    SwarmTopologyType,
    SwarmTestCase,
)
from sparse_eval.swarm.trace import parse_trace, load_trace_file
from sparse_eval.suite import Criterion, DEFAULT_CRITERIA


DEFAULT_SWARM_CRITERIA = [
    Criterion(key="outcome", label="Final Answer Quality", description="Quality of the swarm's final answer", weight=35),
    Criterion(key="routing", label="Routing Correctness", description="Was the query sent to the right specialist?", weight=20),
    Criterion(key="handoff", label="Handoff Quality", description="Was context preserved during handoffs?", weight=15),
    Criterion(key="context_preservation", label="Context Preservation", description="Did agents maintain conversation context?", weight=15),
    Criterion(key="convergence", label="Convergence", description="Did the swarm reach an answer efficiently?", weight=15),
]


class SwarmSuite:
    """
    A collection of swarm test cases loaded from YAML.

    YAML format:
    ```yaml
    name: Support Swarm Eval
    mode: swarm-trace
    model: claude-sonnet-4-5-20250929

    topology:
      type: orchestrator
      orchestrator: router
      agents:
        - id: router
          role: Router
        - id: billing
          role: Billing Specialist
          capabilities: [billing, refunds]

    criteria:
      - key: outcome
        weight: 35
      - key: routing
        weight: 20

    tests:
      - question: "What is my refund status?"
        expected_answer: "Your refund is processing..."
        expected_route: [router, billing]
        trace:
          - from: user
            to: router
            content: "What is my refund status?"
          - from: router
            to: billing
            role: handoff
            content: "Customer asking about refund"
          - from: billing
            to: user
            role: final
            content: "Your refund is processing..."
    ```
    """

    def __init__(
        self,
        name: str = "Untitled Swarm Suite",
        description: str = "",
        model: str = "claude-sonnet-4-5-20250929",
        mode: str = "swarm-trace",
        topology: Optional[SwarmTopology] = None,
        criteria: Optional[list[Criterion]] = None,
        custom_rules: Optional[list[str]] = None,
        tests: Optional[list[SwarmTestCase]] = None,
        reliability_config: Optional[dict] = None,
        concurrency: int = 1,
    ):
        self.name = name
        self.description = description
        self.model = model
        self.mode = mode
        self.topology = topology or SwarmTopology(type=SwarmTopologyType.ORCHESTRATOR)
        self.criteria = criteria or list(DEFAULT_SWARM_CRITERIA)
        self.custom_rules = custom_rules or []
        self.tests = tests or []
        self.reliability_config = reliability_config
        self.concurrency = concurrency

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SwarmSuite":
        """Load a swarm test suite from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Suite file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty suite file: {path}")

        mode = data.get("mode", "swarm-trace")
        if mode not in ("swarm-trace", "swarm"):
            raise ValueError(
                f"Invalid mode '{mode}' for swarm suite. Use 'swarm-trace' or 'swarm'."
            )

        # Parse topology
        topology = cls._parse_topology(data.get("topology", {}))

        # Parse criteria
        criteria = None
        if "criteria" in data:
            criteria = [
                Criterion(
                    key=c["key"],
                    label=c.get("label", c["key"].replace("_", " ").title()),
                    description=c.get("description", ""),
                    weight=c.get("weight", 50),
                )
                for c in data["criteria"]
            ]

        # Parse test cases with traces
        tests = []
        suite_dir = path.parent
        for t in data.get("tests", []):
            trace = None
            trace_file = t.get("trace_file")

            if "trace" in t:
                trace = parse_trace(t["trace"])
            elif trace_file:
                trace = load_trace_file(suite_dir / trace_file)

            tests.append(SwarmTestCase(
                question=t["question"],
                expected_answer=t.get("expected_answer"),
                expected_route=t.get("expected_route"),
                expected_agents=t.get("expected_agents"),
                max_turns=t.get("max_turns"),
                context=t.get("context"),
                tags=t.get("tags", []),
                metadata=t.get("metadata", {}),
                trace=trace,
                trace_file=trace_file,
            ))

        return cls(
            name=data.get("name", path.stem),
            description=data.get("description", ""),
            model=data.get("model", "claude-sonnet-4-5-20250929"),
            mode=mode,
            topology=topology,
            criteria=criteria,
            custom_rules=data.get("custom_rules", []),
            tests=tests,
            reliability_config=data.get("reliability"),
            concurrency=data.get("concurrency", 1),
        )

    @staticmethod
    def _parse_topology(data: dict) -> SwarmTopology:
        """Parse topology section from YAML."""
        topo_type_str = data.get("type", "orchestrator")
        try:
            topo_type = SwarmTopologyType(topo_type_str)
        except ValueError:
            raise ValueError(
                f"Invalid topology type '{topo_type_str}'. Use 'orchestrator' or 'collaborative'."
            )

        agents = []
        for a in data.get("agents", []):
            agents.append(SwarmAgent(
                id=a["id"],
                role=a.get("role", a["id"]),
                description=a.get("description", ""),
                capabilities=a.get("capabilities", []),
                type=a.get("type", "callable"),
                url=a.get("url"),
                method=a.get("method", "POST"),
                headers=a.get("headers", {}),
                payload_template=a.get("payload_template", '{"message": "{{message}}"}'),
                response_path=a.get("response_path", "response"),
                command=a.get("command"),
            ))

        connections = data.get("connections", {})

        return SwarmTopology(
            type=topo_type,
            agents=agents,
            orchestrator_id=data.get("orchestrator"),
            connections=connections,
        )

    def criteria_dicts(self) -> list[dict]:
        """Return criteria as list of dicts for judge consumption."""
        return [c.to_dict() for c in self.criteria]

    def filter_by_tags(self, tags: list[str]) -> list[SwarmTestCase]:
        """Return test cases matching any of the given tags."""
        tag_set = set(tags)
        return [t for t in self.tests if tag_set & set(t.tags)]

    def get_reliability_config(self):
        """Parse reliability config dict into SwarmReliabilityConfig, or None."""
        if not self.reliability_config:
            return None
        from sparse_eval.swarm.reliability import SwarmReliabilityConfig
        return SwarmReliabilityConfig(
            outcome_weight=self.reliability_config.get("outcome_weight", 0.35),
            process_weight=self.reliability_config.get("process_weight", 0.35),
            structural_weight=self.reliability_config.get("structural_weight", 0.15),
            performance_weight=self.reliability_config.get("performance_weight", 0.15),
        )

    def __len__(self):
        return len(self.tests)

    def __repr__(self):
        return f"SwarmSuite({self.name!r}, {len(self.tests)} tests, {self.topology})"
