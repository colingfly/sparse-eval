"""Tests for swarm engine (SwarmReliabilityRunner)."""

import pytest
from unittest.mock import patch, MagicMock
from sparse_eval.swarm.models import (
    SwarmMessage,
    SwarmMessageRole,
    SwarmTrace,
    SwarmTestCase,
    SwarmResult,
)
from sparse_eval.swarm.suite import SwarmSuite
from sparse_eval.swarm.engine import SwarmReliabilityRunner
from sparse_eval.judge import JudgeResult, CriteriaScore


def _make_suite_with_traces():
    """Create a minimal SwarmSuite with inline traces for testing."""
    from sparse_eval.swarm.models import SwarmAgent, SwarmTopology, SwarmTopologyType
    from sparse_eval.suite import Criterion

    topology = SwarmTopology(
        type=SwarmTopologyType.ORCHESTRATOR,
        agents=[
            SwarmAgent(id="router", role="Router"),
            SwarmAgent(id="billing", role="Billing"),
        ],
        orchestrator_id="router",
    )

    trace1 = SwarmTrace(
        messages=[
            SwarmMessage(from_agent="user", to_agent="router", role=SwarmMessageRole.USER, content="Refund?"),
            SwarmMessage(from_agent="router", to_agent="billing", role=SwarmMessageRole.HANDOFF, content="Customer asking about refund"),
            SwarmMessage(from_agent="billing", to_agent="router", role=SwarmMessageRole.AGENT, content="Refund is processing"),
            SwarmMessage(from_agent="router", to_agent="user", role=SwarmMessageRole.FINAL, content="Your refund is processing."),
        ],
        final_answer="Your refund is processing.",
    )

    trace2 = SwarmTrace(
        messages=[
            SwarmMessage(from_agent="user", to_agent="router", role=SwarmMessageRole.USER, content="Login issue"),
            SwarmMessage(from_agent="router", to_agent="billing", role=SwarmMessageRole.HANDOFF, content="login"),
            SwarmMessage(from_agent="billing", to_agent="router", role=SwarmMessageRole.AGENT, content="Not my area"),
        ],
        final_answer="Not my area",
    )

    tests = [
        SwarmTestCase(
            question="What is my refund status?",
            expected_answer="Your refund is processing.",
            expected_route=["router", "billing"],
            max_turns=4,
            tags=["billing"],
            trace=trace1,
        ),
        SwarmTestCase(
            question="I can't log in",
            expected_answer="Reset your password at Settings > Security",
            expected_route=["router", "billing"],  # Wrong route in trace — should score low
            max_turns=4,
            tags=["tech"],
            trace=trace2,
        ),
    ]

    criteria = [
        Criterion(key="outcome", label="Outcome", weight=35),
        Criterion(key="routing", label="Routing", weight=20),
        Criterion(key="handoff", label="Handoff", weight=15),
        Criterion(key="context_preservation", label="Context", weight=15),
        Criterion(key="convergence", label="Convergence", weight=15),
    ]

    return SwarmSuite(
        name="Test Swarm",
        topology=topology,
        criteria=criteria,
        tests=tests,
    )


def _mock_judge_score(**kwargs):
    """Create a mock JudgeResult for outcome evaluation."""
    return JudgeResult(
        criteria_scores=[
            CriteriaScore(key="accuracy", score=80, reasoning="Good"),
            CriteriaScore(key="completeness", score=75, reasoning="OK"),
            CriteriaScore(key="hallucination", score=90, reasoning="Clean"),
        ],
        overall_score=81.0,
        overall_reasoning="Decent answer",
        status="pass",
    )


class TestSwarmReliabilityRunner:

    @patch("sparse_eval.swarm.judge.Judge")
    def test_run_sequential(self, MockJudge):
        """Test basic sequential swarm eval run."""
        mock_instance = MockJudge.return_value
        mock_instance.score.return_value = _mock_judge_score()

        suite = _make_suite_with_traces()
        runner = SwarmReliabilityRunner(verbose=False)
        summary = runner.run(suite)

        assert summary.suite_name == "Test Swarm"
        assert summary.total == 2
        assert len(summary.results) == 2
        assert summary.overall_score > 0
        assert summary.reliability_score is not None
        assert summary.reliability_grade in ("A", "B", "C", "D", "F")

    @patch("sparse_eval.swarm.judge.Judge")
    def test_run_with_tags(self, MockJudge):
        """Test tag filtering."""
        mock_instance = MockJudge.return_value
        mock_instance.score.return_value = _mock_judge_score()

        suite = _make_suite_with_traces()
        runner = SwarmReliabilityRunner(verbose=False)
        summary = runner.run(suite, tags=["billing"])

        assert summary.total == 1
        assert summary.results[0].test_case.question == "What is my refund status?"

    @patch("sparse_eval.swarm.judge.Judge")
    def test_run_empty_suite(self, MockJudge):
        """Test with no matching tests."""
        suite = _make_suite_with_traces()
        runner = SwarmReliabilityRunner(verbose=False)
        summary = runner.run(suite, tags=["nonexistent"])

        assert summary.total == 0
        assert summary.results == []

    @patch("sparse_eval.swarm.judge.Judge")
    def test_per_agent_summary(self, MockJudge):
        """Test that per-agent summary is computed."""
        mock_instance = MockJudge.return_value
        mock_instance.score.return_value = _mock_judge_score()

        suite = _make_suite_with_traces()
        runner = SwarmReliabilityRunner(verbose=False)
        summary = runner.run(suite)

        assert "router" in summary.per_agent_summary
        assert "billing" in summary.per_agent_summary
        assert summary.per_agent_summary["router"]["role"] == "Router"

    @patch("sparse_eval.swarm.judge.Judge")
    def test_on_result_callback(self, MockJudge):
        """Test that on_result callback is called."""
        mock_instance = MockJudge.return_value
        mock_instance.score.return_value = _mock_judge_score()

        results_received = []
        def on_result(result, index, total):
            results_received.append((result.overall_score, index, total))

        suite = _make_suite_with_traces()
        runner = SwarmReliabilityRunner(verbose=False, on_result=on_result)
        runner.run(suite)

        assert len(results_received) == 2
        assert results_received[0][1] == 1  # index
        assert results_received[0][2] == 2  # total

    @patch("sparse_eval.swarm.judge.Judge")
    def test_summary_to_dict(self, MockJudge):
        """Test that summary serializes to dict."""
        mock_instance = MockJudge.return_value
        mock_instance.score.return_value = _mock_judge_score()

        suite = _make_suite_with_traces()
        runner = SwarmReliabilityRunner(verbose=False)
        summary = runner.run(suite)

        d = summary.to_dict()
        assert "suite_name" in d
        assert "topology_type" in d
        assert "results" in d
        assert "per_agent_summary" in d
        assert "reliability_score" in d

    @patch("sparse_eval.swarm.judge.Judge")
    def test_run_async(self, MockJudge):
        """Test async concurrent run."""
        mock_instance = MockJudge.return_value
        mock_instance.score.return_value = _mock_judge_score()

        suite = _make_suite_with_traces()
        runner = SwarmReliabilityRunner(verbose=False, concurrency=2)
        summary = runner.run(suite)

        assert summary.total == 2
        assert len(summary.results) == 2


class TestSwarmSuiteFromYaml:

    def test_load_example(self):
        """Test loading the example swarm_trace.yaml."""
        from pathlib import Path

        suite_path = Path(__file__).parent.parent / "examples" / "swarm_trace.yaml"
        if not suite_path.exists():
            pytest.skip("Example suite not found")

        suite = SwarmSuite.from_yaml(suite_path)
        assert suite.name == "Customer Support Swarm Eval"
        assert len(suite.tests) == 4
        assert all(tc.trace is not None for tc in suite.tests)

    def test_invalid_mode(self, tmp_path):
        """Test that invalid mode raises error."""
        yaml_content = "name: test\nmode: invalid\ntests: []\n"
        p = tmp_path / "bad.yaml"
        p.write_text(yaml_content)

        with pytest.raises(ValueError, match="Invalid mode"):
            SwarmSuite.from_yaml(p)
