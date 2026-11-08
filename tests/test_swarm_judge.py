"""Tests for swarm judge and criteria scoring."""

import pytest
from sparse_eval.swarm.models import (
    SwarmMessage,
    SwarmMessageRole,
    SwarmTrace,
    SwarmTestCase,
    SwarmResult,
)
from sparse_eval.swarm.criteria import (
    SwarmEvalContext,
    RoutingCriterion,
    HandoffCriterion,
    ContextPreservationCriterion,
    ConvergenceCriterion,
    CollaborationCriterion,
)


def _make_trace(messages, final_answer="Final answer"):
    return SwarmTrace(
        messages=[
            SwarmMessage(
                from_agent=m[0],
                to_agent=m[1],
                role=m[2] if len(m) > 2 else SwarmMessageRole.AGENT,
                content=m[3] if len(m) > 3 else "",
                metadata=m[4] if len(m) > 4 else {},
            )
            for m in messages
        ],
        final_answer=final_answer,
    )


# ---- RoutingCriterion ----

class TestRoutingCriterion:
    criterion = RoutingCriterion()

    def test_perfect_route_match(self):
        tc = SwarmTestCase(question="Q", expected_route=["router", "billing"])
        trace = _make_trace([
            ("user", "router", SwarmMessageRole.USER, "Q"),
            ("router", "billing", SwarmMessageRole.HANDOFF, "route"),
            ("billing", "router", SwarmMessageRole.AGENT, "answer"),
            ("router", "user", SwarmMessageRole.FINAL, "Final answer"),
        ])
        ctx = SwarmEvalContext(test_case=tc, trace=trace)
        score = self.criterion.score(ctx)
        assert score == 100.0

    def test_wrong_route(self):
        tc = SwarmTestCase(question="Q", expected_route=["router", "billing"])
        trace = _make_trace([
            ("user", "router", SwarmMessageRole.USER, "Q"),
            ("router", "tech", SwarmMessageRole.HANDOFF, "route"),
            ("tech", "router", SwarmMessageRole.AGENT, "answer"),
            ("router", "user", SwarmMessageRole.FINAL, "Final answer"),
        ])
        ctx = SwarmEvalContext(test_case=tc, trace=trace)
        score = self.criterion.score(ctx)
        assert score < 100.0

    def test_no_expected_route(self):
        tc = SwarmTestCase(question="Q")
        trace = _make_trace([("router", "user", SwarmMessageRole.FINAL, "answer")])
        ctx = SwarmEvalContext(test_case=tc, trace=trace)
        score = self.criterion.score(ctx)
        assert score == 100.0  # No expectations = pass

    def test_expected_agents_scoring(self):
        tc = SwarmTestCase(question="Q", expected_agents=["billing"])
        trace = _make_trace([
            ("user", "router", SwarmMessageRole.USER, "Q"),
            ("router", "billing", SwarmMessageRole.HANDOFF, "route"),
            ("billing", "router", SwarmMessageRole.AGENT, "answer"),
        ])
        ctx = SwarmEvalContext(test_case=tc, trace=trace)
        score = self.criterion.score(ctx)
        assert score == 100.0


# ---- HandoffCriterion ----

class TestHandoffCriterion:
    criterion = HandoffCriterion()

    def test_good_handoff(self):
        tc = SwarmTestCase(question="Q")
        trace = _make_trace([
            ("router", "billing", SwarmMessageRole.HANDOFF,
             "Customer asking about refund for order #12345",
             {"routing_reason": "refund query"}),
        ], final_answer="answer")
        ctx = SwarmEvalContext(test_case=tc, trace=trace)
        score = self.criterion.score(ctx)
        assert score == 100.0

    def test_empty_handoff(self):
        tc = SwarmTestCase(question="Q")
        trace = _make_trace([
            ("router", "billing", SwarmMessageRole.HANDOFF, ""),
        ], final_answer="answer")
        ctx = SwarmEvalContext(test_case=tc, trace=trace)
        score = self.criterion.score(ctx)
        assert score == 0.0

    def test_minimal_handoff(self):
        tc = SwarmTestCase(question="Q")
        trace = _make_trace([
            ("router", "billing", SwarmMessageRole.HANDOFF, "refund"),
        ], final_answer="answer")
        ctx = SwarmEvalContext(test_case=tc, trace=trace)
        score = self.criterion.score(ctx)
        assert score == 30.0  # Below MIN_HANDOFF_CONTENT_LENGTH

    def test_no_handoffs_simple(self):
        tc = SwarmTestCase(question="Q")
        trace = _make_trace([
            ("user", "router", SwarmMessageRole.USER, "Q"),
            ("router", "user", SwarmMessageRole.FINAL, "answer"),
        ], final_answer="answer")
        ctx = SwarmEvalContext(test_case=tc, trace=trace)
        score = self.criterion.score(ctx)
        assert score == 100.0  # Simple 2-turn, no handoffs needed


# ---- ContextPreservationCriterion ----

class TestContextPreservation:
    criterion = ContextPreservationCriterion()

    def test_good_context(self):
        tc = SwarmTestCase(question="What is the status of my refund for order #12345?")
        trace = _make_trace([
            ("user", "router", SwarmMessageRole.USER, "What is the status of my refund for order #12345?"),
            ("router", "billing", SwarmMessageRole.HANDOFF, "Customer asking about refund status for order #12345"),
            ("billing", "router", SwarmMessageRole.AGENT, "Refund processing"),
            ("router", "user", SwarmMessageRole.FINAL, "Your refund for order #12345 is processing"),
        ], final_answer="Your refund for order #12345 is processing")
        ctx = SwarmEvalContext(test_case=tc, trace=trace)
        score = self.criterion.score(ctx)
        assert score >= 70.0  # Should preserve key terms

    def test_lost_context(self):
        tc = SwarmTestCase(question="What is the status of my refund for order #12345?")
        trace = _make_trace([
            ("user", "router", SwarmMessageRole.USER, "What is the status of my refund for order #12345?"),
            ("router", "billing", SwarmMessageRole.HANDOFF, "question"),
            ("billing", "router", SwarmMessageRole.AGENT, "done"),
        ], final_answer="Everything is fine.")
        ctx = SwarmEvalContext(test_case=tc, trace=trace)
        score = self.criterion.score(ctx)
        assert score < 50.0  # Context was lost

    def test_single_turn(self):
        tc = SwarmTestCase(question="Hello")
        trace = _make_trace([
            ("user", "router", SwarmMessageRole.USER, "Hello"),
        ], final_answer="Hi")
        ctx = SwarmEvalContext(test_case=tc, trace=trace)
        score = self.criterion.score(ctx)
        assert score == 100.0


# ---- ConvergenceCriterion ----

class TestConvergence:
    criterion = ConvergenceCriterion()

    def test_within_max_turns(self):
        tc = SwarmTestCase(question="Q", max_turns=4)
        trace = _make_trace([
            ("user", "router", SwarmMessageRole.USER, "Q"),
            ("router", "billing", SwarmMessageRole.HANDOFF, "route"),
            ("billing", "router", SwarmMessageRole.AGENT, "answer"),
        ], final_answer="answer")
        ctx = SwarmEvalContext(test_case=tc, trace=trace)
        score = self.criterion.score(ctx)
        assert score == 100.0

    def test_exceeds_max_turns(self):
        tc = SwarmTestCase(question="Q", max_turns=2)
        msgs = [("a", "b", SwarmMessageRole.AGENT, f"msg{i}") for i in range(5)]
        trace = _make_trace(msgs, final_answer="answer")
        ctx = SwarmEvalContext(test_case=tc, trace=trace)
        score = self.criterion.score(ctx)
        assert score < 100.0  # Penalized for excess turns

    def test_no_final_answer(self):
        tc = SwarmTestCase(question="Q")
        trace = SwarmTrace(messages=[], final_answer="")
        ctx = SwarmEvalContext(test_case=tc, trace=trace)
        score = self.criterion.score(ctx)
        assert score == 0.0

    def test_circular_routing(self):
        tc = SwarmTestCase(question="Q", max_turns=10)
        trace = _make_trace([
            ("router", "billing", SwarmMessageRole.HANDOFF, "go"),
            ("billing", "router", SwarmMessageRole.AGENT, "back"),
            ("router", "router", SwarmMessageRole.AGENT, "self"),  # circular
        ], final_answer="answer")
        ctx = SwarmEvalContext(test_case=tc, trace=trace)
        score = self.criterion.score(ctx)
        assert score < 100.0  # Penalized for circular routing


# ---- CollaborationCriterion ----

class TestCollaboration:
    criterion = CollaborationCriterion()

    def test_non_collaborative_topology(self):
        ctx = SwarmEvalContext(topology_type="orchestrator")
        score = self.criterion.score(ctx)
        assert score == 100.0  # Not applicable

    def test_balanced_collaboration(self):
        tc = SwarmTestCase(question="Q")
        trace = _make_trace([
            ("agent_a", "agent_b", SwarmMessageRole.AGENT, "My analysis"),
            ("agent_b", "agent_a", SwarmMessageRole.AGENT, "Building on that"),
            ("agent_a", "user", SwarmMessageRole.FINAL, "Combined answer"),
        ], final_answer="Combined answer")
        ctx = SwarmEvalContext(test_case=tc, trace=trace, topology_type="collaborative")
        score = self.criterion.score(ctx)
        assert score >= 60.0  # Two agents, balanced

    def test_single_agent_collaboration(self):
        tc = SwarmTestCase(question="Q")
        trace = _make_trace([
            ("agent_a", "user", SwarmMessageRole.AGENT, "Only me"),
        ], final_answer="Only me")
        ctx = SwarmEvalContext(test_case=tc, trace=trace, topology_type="collaborative")
        score = self.criterion.score(ctx)
        assert score <= 40.0  # Poor collaboration


# ---- Trace parsing ----

class TestTraceParsing:
    def test_parse_inline_trace(self):
        from sparse_eval.swarm.trace import parse_trace

        data = [
            {"from": "user", "to": "router", "content": "Hello"},
            {"from": "router", "to": "user", "role": "final", "content": "Hi there!"},
        ]
        trace = parse_trace(data)
        assert trace.final_answer == "Hi there!"
        assert trace.num_turns == 2
        assert trace.messages[0].role == SwarmMessageRole.USER
        assert trace.messages[1].role == SwarmMessageRole.FINAL

    def test_parse_trace_fallback_final(self):
        from sparse_eval.swarm.trace import parse_trace

        data = [
            {"from": "user", "to": "router", "content": "Hello"},
            {"from": "router", "to": "user", "content": "Hi there!"},
        ]
        trace = parse_trace(data)
        assert trace.final_answer == "Hi there!"  # Last message to user


# ---- SwarmSuite ----

class TestSwarmSuite:
    def test_load_example_yaml(self):
        from sparse_eval.swarm.suite import SwarmSuite
        from pathlib import Path

        suite_path = Path(__file__).parent.parent / "examples" / "swarm_trace.yaml"
        if not suite_path.exists():
            pytest.skip("Example suite not found")

        suite = SwarmSuite.from_yaml(suite_path)
        assert suite.name == "Customer Support Swarm Eval"
        assert suite.mode == "swarm-trace"
        assert len(suite.tests) == 4
        assert suite.topology.type.value == "orchestrator"
        assert len(suite.topology.agents) == 3

        # Check traces were parsed
        for tc in suite.tests:
            assert tc.trace is not None
            assert tc.trace.final_answer != ""

    def test_filter_by_tags(self):
        from sparse_eval.swarm.suite import SwarmSuite
        from pathlib import Path

        suite_path = Path(__file__).parent.parent / "examples" / "swarm_trace.yaml"
        if not suite_path.exists():
            pytest.skip("Example suite not found")

        suite = SwarmSuite.from_yaml(suite_path)
        billing_tests = suite.filter_by_tags(["billing"])
        assert len(billing_tests) >= 2


# ---- SwarmReliability ----

class TestSwarmReliability:
    def test_basic_computation(self):
        from sparse_eval.swarm.reliability import compute_swarm_reliability

        score, grade = compute_swarm_reliability(
            outcome_score=90,
            process_score=85,
        )
        assert score >= 85
        assert grade in ("A", "B")

    def test_all_four_pillars(self):
        from sparse_eval.swarm.reliability import compute_swarm_reliability

        score, grade = compute_swarm_reliability(
            outcome_score=90,
            process_score=85,
            schema_score=100,
            latency_score=95,
        )
        assert score >= 88
        assert grade == "A"

    def test_failing_score(self):
        from sparse_eval.swarm.reliability import compute_swarm_reliability

        score, grade = compute_swarm_reliability(
            outcome_score=20,
            process_score=15,
        )
        assert score < 40
        assert grade == "F"

    def test_config_weights(self):
        from sparse_eval.swarm.reliability import (
            compute_swarm_reliability,
            SwarmReliabilityConfig,
        )

        # Process-heavy config
        config = SwarmReliabilityConfig(
            outcome_weight=0.20,
            process_weight=0.50,
            structural_weight=0.15,
            performance_weight=0.15,
        )
        score, _ = compute_swarm_reliability(
            outcome_score=50,
            process_score=90,
            config=config,
        )
        # With process weighted higher, score should be closer to 90
        assert score > 70

    def test_config_validation(self):
        from sparse_eval.swarm.reliability import SwarmReliabilityConfig

        with pytest.raises(ValueError):
            SwarmReliabilityConfig(
                outcome_weight=0.5,
                process_weight=0.5,
                structural_weight=0.5,
                performance_weight=0.5,
            )
