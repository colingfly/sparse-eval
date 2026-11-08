"""Tests for SwarmRunner (orchestrated execution)."""

import json
import pytest
from unittest.mock import patch, MagicMock

from sparse_eval.swarm.models import (
    SwarmAgent,
    SwarmTopology,
    SwarmTopologyType,
    SwarmMessage,
    SwarmMessageRole,
    SwarmTrace,
    SwarmTestCase,
)
from sparse_eval.swarm.runner import SwarmRunner, _parse_orchestrator_response
from sparse_eval.swarm.engine import SwarmReliabilityRunner
from sparse_eval.swarm.suite import SwarmSuite
from sparse_eval.suite import Criterion
from sparse_eval.judge import JudgeResult, CriteriaScore


# ---- Helpers ----

def _make_orchestrator_topology():
    return SwarmTopology(
        type=SwarmTopologyType.ORCHESTRATOR,
        agents=[
            SwarmAgent(id="router", role="Router"),
            SwarmAgent(id="billing", role="Billing", capabilities=["billing", "refunds"]),
            SwarmAgent(id="tech", role="Tech Support", capabilities=["passwords", "accounts"]),
        ],
        orchestrator_id="router",
    )


def _make_collaborative_topology():
    return SwarmTopology(
        type=SwarmTopologyType.COLLABORATIVE,
        agents=[
            SwarmAgent(id="analyst", role="Analyst"),
            SwarmAgent(id="reviewer", role="Reviewer"),
        ],
    )


def _make_router_callable():
    """Mock router that routes billing to billing, tech to tech."""
    def router(message: str) -> str:
        msg_lower = message.lower()
        # Synthesis step: check this first since synthesis prompts contain agent names
        if "synthesize" in msg_lower or "response from" in msg_lower or "original question" in msg_lower:
            return json.dumps({
                "route_to": None,
                "context": None,
                "direct_answer": "Based on our specialist: " + message.split(":")[-1].strip()[:100],
            })
        # Check the "Customer question:" line specifically for routing
        customer_q = ""
        if "customer question:" in msg_lower:
            customer_q = msg_lower.split("customer question:")[-1].strip()
        else:
            customer_q = msg_lower
        if "refund" in customer_q or "payment" in customer_q:
            return json.dumps({
                "route_to": "billing",
                "context": "Customer has a billing question",
                "direct_answer": None,
            })
        if "password" in customer_q or "log" in customer_q or "login" in customer_q or "can't log" in customer_q:
            return json.dumps({
                "route_to": "tech",
                "context": "Customer has a technical issue",
                "direct_answer": None,
            })
        return json.dumps({
            "route_to": None,
            "context": None,
            "direct_answer": "I can help you with that directly.",
        })
    return router


def _make_billing_callable():
    def billing(message: str) -> str:
        return "Your refund for order #12345 is being processed. Expected: 5-7 business days."
    return billing


def _make_tech_callable():
    def tech(message: str) -> str:
        return "Reset your password at Settings > Security > Reset Password."
    return tech


# ---- Test _parse_orchestrator_response ----

class TestParseOrchestratorResponse:
    def test_valid_json_routing(self):
        raw = json.dumps({"route_to": "billing", "context": "billing query", "direct_answer": None})
        result = _parse_orchestrator_response(raw)
        assert result["route_to"] == "billing"
        assert result["context"] == "billing query"
        assert result["direct_answer"] is None

    def test_valid_json_direct_answer(self):
        raw = json.dumps({"route_to": None, "context": None, "direct_answer": "Here is your answer"})
        result = _parse_orchestrator_response(raw)
        assert result["route_to"] is None
        assert result["direct_answer"] == "Here is your answer"

    def test_markdown_fenced_json(self):
        raw = '```json\n{"route_to": "tech", "context": "login issue", "direct_answer": null}\n```'
        result = _parse_orchestrator_response(raw)
        assert result["route_to"] == "tech"

    def test_plain_text_fallback(self):
        raw = "I'll help you with that directly."
        result = _parse_orchestrator_response(raw)
        assert result["route_to"] is None
        assert result["direct_answer"] == raw

    def test_empty_response(self):
        result = _parse_orchestrator_response("")
        assert result["direct_answer"] == ""


# ---- Test SwarmRunner (Orchestrator) ----

class TestSwarmRunnerOrchestrator:
    def test_basic_routing(self):
        """Test that the orchestrator routes to the right specialist."""
        topology = _make_orchestrator_topology()
        runner = SwarmRunner(
            topology=topology,
            agent_callables={
                "router": _make_router_callable(),
                "billing": _make_billing_callable(),
                "tech": _make_tech_callable(),
            },
        )

        tc = SwarmTestCase(
            question="What is the status of my refund for order #12345?",
            expected_route=["router", "billing"],
        )
        trace = runner.execute(tc)

        assert trace.final_answer != ""
        assert trace.num_turns >= 3  # user->router, router->billing handoff, billing->router, router->user final
        assert "billing" in trace.agents_involved
        assert any(m.role == SwarmMessageRole.HANDOFF for m in trace.messages)
        assert any(m.role == SwarmMessageRole.FINAL for m in trace.messages)

    def test_tech_routing(self):
        """Test routing to tech agent."""
        topology = _make_orchestrator_topology()
        runner = SwarmRunner(
            topology=topology,
            agent_callables={
                "router": _make_router_callable(),
                "billing": _make_billing_callable(),
                "tech": _make_tech_callable(),
            },
        )

        tc = SwarmTestCase(question="I can't log into my account")
        trace = runner.execute(tc)

        assert trace.final_answer != ""
        assert "tech" in trace.agents_involved

    def test_direct_answer(self):
        """Test orchestrator answering directly without routing."""
        topology = _make_orchestrator_topology()

        def direct_router(message: str) -> str:
            return json.dumps({
                "route_to": None,
                "context": None,
                "direct_answer": "Hello! How can I help you today?",
            })

        runner = SwarmRunner(
            topology=topology,
            agent_callables={"router": direct_router},
        )

        tc = SwarmTestCase(question="Hello")
        trace = runner.execute(tc)

        assert trace.final_answer == "Hello! How can I help you today?"
        assert any(m.role == SwarmMessageRole.FINAL for m in trace.messages)
        # No handoffs expected
        assert trace.num_handoffs == 0

    def test_no_orchestrator_configured(self):
        """Test graceful handling when no orchestrator is set."""
        topology = SwarmTopology(type=SwarmTopologyType.ORCHESTRATOR, agents=[])
        runner = SwarmRunner(topology=topology)

        tc = SwarmTestCase(question="Hello")
        trace = runner.execute(tc)

        assert "(No orchestrator configured)" in trace.final_answer

    def test_unknown_route_target(self):
        """Test routing to an agent that doesn't exist."""
        topology = _make_orchestrator_topology()

        def bad_router(message: str) -> str:
            return json.dumps({"route_to": "nonexistent", "context": "oops", "direct_answer": None})

        runner = SwarmRunner(
            topology=topology,
            agent_callables={"router": bad_router},
        )

        tc = SwarmTestCase(question="Help me")
        trace = runner.execute(tc)

        assert "nonexistent" in trace.final_answer or trace.final_answer != ""

    def test_trace_captures_timestamps(self):
        """Test that trace messages have timestamps."""
        topology = _make_orchestrator_topology()
        runner = SwarmRunner(
            topology=topology,
            agent_callables={
                "router": _make_router_callable(),
                "billing": _make_billing_callable(),
            },
        )

        tc = SwarmTestCase(question="Refund status?")
        trace = runner.execute(tc)

        assert trace.total_time_ms >= 0
        # At least the first message should have timestamp 0 or close to it
        assert trace.messages[0].timestamp_ms >= 0


# ---- Test SwarmRunner (Collaborative) ----

class TestSwarmRunnerCollaborative:
    def test_basic_collaboration(self):
        """Test that all agents participate in collaborative mode."""
        topology = _make_collaborative_topology()

        def analyst(message: str) -> str:
            return "From analysis perspective: the data shows positive trends."

        def reviewer(message: str) -> str:
            return "From review perspective: the methodology is sound."

        runner = SwarmRunner(
            topology=topology,
            agent_callables={"analyst": analyst, "reviewer": reviewer},
            max_rounds=2,
        )

        tc = SwarmTestCase(question="What does the data tell us?")
        trace = runner.execute(tc)

        assert trace.final_answer != ""
        assert "analyst" in trace.agents_involved
        assert "reviewer" in trace.agents_involved
        assert any(m.role == SwarmMessageRole.FINAL for m in trace.messages)

    def test_max_rounds_respected(self):
        """Test that collaborative execution respects max_turns."""
        topology = _make_collaborative_topology()
        call_count = {"analyst": 0, "reviewer": 0}

        def counting_agent(agent_id):
            def agent(message: str) -> str:
                call_count[agent_id] += 1
                return f"Response {call_count[agent_id]} from {agent_id}"
            return agent

        runner = SwarmRunner(
            topology=topology,
            agent_callables={
                "analyst": counting_agent("analyst"),
                "reviewer": counting_agent("reviewer"),
            },
            max_rounds=2,
        )

        tc = SwarmTestCase(question="Analyze this")
        trace = runner.execute(tc)

        # Initial round + 1 refinement round + 1 synthesis = ~3 calls per agent
        # But exact count depends on synthesis step
        assert call_count["analyst"] >= 2
        assert call_count["reviewer"] >= 2

    def test_single_agent_collaborative(self):
        """Test collaborative with a single agent."""
        topology = SwarmTopology(
            type=SwarmTopologyType.COLLABORATIVE,
            agents=[SwarmAgent(id="solo", role="Solo Agent")],
        )

        runner = SwarmRunner(
            topology=topology,
            agent_callables={"solo": lambda msg: "I'm the only one here."},
            max_rounds=1,
        )

        tc = SwarmTestCase(question="Hello")
        trace = runner.execute(tc)

        assert trace.final_answer == "I'm the only one here."


# ---- Test full pipeline: SwarmRunner + SwarmReliabilityRunner ----

class TestOrchestratedPipeline:
    @patch("sparse_eval.swarm.judge.Judge")
    def test_end_to_end_orchestrated(self, MockJudge):
        """Test the full orchestrated pipeline: execute -> judge -> summary."""
        mock_instance = MockJudge.return_value
        mock_instance.score.return_value = JudgeResult(
            criteria_scores=[
                CriteriaScore(key="accuracy", score=85, reasoning="Good"),
                CriteriaScore(key="completeness", score=80, reasoning="OK"),
                CriteriaScore(key="hallucination", score=90, reasoning="Clean"),
            ],
            overall_score=85.0,
            overall_reasoning="Solid answer",
            status="pass",
        )

        topology = _make_orchestrator_topology()
        suite = SwarmSuite(
            name="Orchestrated Test",
            mode="swarm",
            topology=topology,
            tests=[
                SwarmTestCase(
                    question="What is the status of my refund?",
                    expected_answer="Your refund is being processed.",
                    expected_route=["router", "billing"],
                    max_turns=6,
                ),
                SwarmTestCase(
                    question="I can't log in",
                    expected_answer="Reset your password.",
                    expected_route=["router", "tech"],
                    max_turns=6,
                ),
            ],
        )

        runner = SwarmReliabilityRunner(verbose=False)
        summary = runner.run(
            suite,
            agent_callables={
                "router": _make_router_callable(),
                "billing": _make_billing_callable(),
                "tech": _make_tech_callable(),
            },
        )

        assert summary.total == 2
        assert summary.suite_name == "Orchestrated Test"
        assert summary.topology_type == "orchestrator"
        assert all(r.trace.num_turns >= 3 for r in summary.results)
        assert summary.reliability_score is not None
        assert summary.reliability_grade in ("A", "B", "C", "D", "F")

    @patch("sparse_eval.swarm.judge.Judge")
    def test_orchestrated_with_concurrency(self, MockJudge):
        """Test orchestrated mode with concurrent execution."""
        mock_instance = MockJudge.return_value
        mock_instance.score.return_value = JudgeResult(
            criteria_scores=[CriteriaScore(key="accuracy", score=80)],
            overall_score=80.0,
            status="pass",
        )

        topology = _make_orchestrator_topology()
        suite = SwarmSuite(
            name="Concurrent Test",
            mode="swarm",
            topology=topology,
            tests=[
                SwarmTestCase(question="Refund?", expected_answer="Processing.", max_turns=6),
                SwarmTestCase(question="Login help?", expected_answer="Reset password.", max_turns=6),
            ],
        )

        runner = SwarmReliabilityRunner(verbose=False, concurrency=2)
        summary = runner.run(
            suite,
            agent_callables={
                "router": _make_router_callable(),
                "billing": _make_billing_callable(),
                "tech": _make_tech_callable(),
            },
        )

        assert summary.total == 2

    @patch("sparse_eval.swarm.judge.Judge")
    def test_mixed_mode_trace_and_orchestrated(self, MockJudge):
        """Test suite where some tests have pre-recorded traces and others don't."""
        mock_instance = MockJudge.return_value
        mock_instance.score.return_value = JudgeResult(
            criteria_scores=[CriteriaScore(key="accuracy", score=80)],
            overall_score=80.0,
            status="pass",
        )

        topology = _make_orchestrator_topology()

        # Pre-recorded trace for test 1
        pre_trace = SwarmTrace(
            messages=[
                SwarmMessage(from_agent="user", to_agent="router", role=SwarmMessageRole.USER, content="Q1"),
                SwarmMessage(from_agent="router", to_agent="user", role=SwarmMessageRole.FINAL, content="A1"),
            ],
            final_answer="A1",
        )

        suite = SwarmSuite(
            name="Mixed Mode Test",
            mode="swarm",
            topology=topology,
            tests=[
                SwarmTestCase(question="Pre-recorded", expected_answer="A1", trace=pre_trace, max_turns=4),
                SwarmTestCase(question="Refund status?", expected_answer="Processing.", max_turns=6),
            ],
        )

        runner = SwarmReliabilityRunner(verbose=False)
        summary = runner.run(
            suite,
            agent_callables={
                "router": _make_router_callable(),
                "billing": _make_billing_callable(),
                "tech": _make_tech_callable(),
            },
        )

        assert summary.total == 2
        # First used pre-recorded trace, second was executed
        assert summary.results[0].trace.final_answer == "A1"
        assert summary.results[1].trace.num_turns >= 3


class TestSwarmSuiteOrchestrated:
    def test_load_orchestrated_yaml(self):
        """Test loading the orchestrated example YAML."""
        from pathlib import Path

        suite_path = Path(__file__).parent.parent / "examples" / "swarm_orchestrated.yaml"
        if not suite_path.exists():
            pytest.skip("Example suite not found")

        suite = SwarmSuite.from_yaml(suite_path)
        assert suite.name == "Customer Support Swarm (Orchestrated)"
        assert suite.mode == "swarm"
        assert len(suite.tests) == 3
        assert suite.topology.orchestrator_id == "router"
        assert len(suite.topology.agents) == 3

        # In orchestrated mode, tests don't have pre-recorded traces
        for tc in suite.tests:
            assert tc.trace is None

        # Check agent configs
        billing = suite.topology.get_agent("billing")
        assert billing.type == "callable"
        assert billing.capabilities == ["billing", "payments", "refunds"]
