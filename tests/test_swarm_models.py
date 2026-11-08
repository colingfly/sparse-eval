"""Tests for swarm data model."""

import pytest
from sparse_eval.swarm.models import (
    SwarmTopologyType,
    SwarmMessageRole,
    SwarmAgent,
    SwarmTopology,
    SwarmMessage,
    SwarmTrace,
    SwarmTestCase,
    SwarmResult,
    SwarmSummary,
)


# ---- SwarmAgent ----

def test_swarm_agent_basic():
    agent = SwarmAgent(id="billing", role="Billing Specialist")
    assert agent.id == "billing"
    assert agent.role == "Billing Specialist"
    assert agent.capabilities == []


def test_swarm_agent_with_capabilities():
    agent = SwarmAgent(id="tech", role="Tech", capabilities=["passwords", "accounts"])
    assert "passwords" in agent.capabilities


# ---- SwarmTopology ----

def test_topology_orchestrator():
    agents = [
        SwarmAgent(id="router", role="Router"),
        SwarmAgent(id="billing", role="Billing"),
    ]
    topo = SwarmTopology(
        type=SwarmTopologyType.ORCHESTRATOR,
        agents=agents,
        orchestrator_id="router",
    )
    assert topo.type == SwarmTopologyType.ORCHESTRATOR
    assert topo.get_agent("router").role == "Router"
    assert topo.get_agent("missing") is None
    assert topo.agent_ids == ["router", "billing"]


def test_topology_collaborative():
    topo = SwarmTopology(type=SwarmTopologyType.COLLABORATIVE)
    assert topo.type == SwarmTopologyType.COLLABORATIVE


# ---- SwarmMessage ----

def test_message_basic():
    msg = SwarmMessage(
        from_agent="router",
        to_agent="billing",
        role=SwarmMessageRole.HANDOFF,
        content="Customer asking about refund",
    )
    assert msg.from_agent == "router"
    assert msg.role == SwarmMessageRole.HANDOFF


# ---- SwarmTrace ----

def test_trace_properties():
    messages = [
        SwarmMessage(from_agent="user", to_agent="router", role=SwarmMessageRole.USER, content="Question"),
        SwarmMessage(from_agent="router", to_agent="billing", role=SwarmMessageRole.HANDOFF, content="Route to billing"),
        SwarmMessage(from_agent="billing", to_agent="router", role=SwarmMessageRole.AGENT, content="Answer from billing"),
        SwarmMessage(from_agent="router", to_agent="user", role=SwarmMessageRole.FINAL, content="Final answer"),
    ]
    trace = SwarmTrace(messages=messages, final_answer="Final answer")

    assert trace.num_turns == 4
    assert trace.num_handoffs == 1
    assert "router" in trace.agents_involved
    assert "billing" in trace.agents_involved
    assert "user" not in trace.agents_involved
    assert trace.actual_route == ["router", "billing", "router"]
    assert len(trace.routing_decisions) == 1
    assert trace.messages_by_agent("billing") == [messages[2]]


def test_trace_to_dict():
    trace = SwarmTrace(
        messages=[SwarmMessage(from_agent="a", to_agent="b", content="hello")],
        final_answer="hello",
        total_time_ms=100,
    )
    d = trace.to_dict()
    assert d["final_answer"] == "hello"
    assert d["total_time_ms"] == 100
    assert len(d["messages"]) == 1


def test_trace_to_transcript():
    messages = [
        SwarmMessage(from_agent="user", to_agent="router", role=SwarmMessageRole.USER, content="Hi"),
        SwarmMessage(from_agent="router", to_agent="user", role=SwarmMessageRole.FINAL, content="Hello!"),
    ]
    trace = SwarmTrace(messages=messages, final_answer="Hello!")
    transcript = trace.to_transcript()
    assert "user -> router [user]: Hi" in transcript
    assert "router -> user [final]: Hello!" in transcript


# ---- SwarmTestCase ----

def test_test_case_basic():
    tc = SwarmTestCase(
        question="What is my refund status?",
        expected_answer="Processing",
        expected_route=["router", "billing"],
        tags=["billing"],
    )
    assert tc.question == "What is my refund status?"
    assert tc.expected_route == ["router", "billing"]


# ---- SwarmResult ----

def test_result_classify():
    assert SwarmResult.classify(90) == "pass"
    assert SwarmResult.classify(70) == "warn"
    assert SwarmResult.classify(50) == "fail"


def test_result_to_dict():
    tc = SwarmTestCase(question="Q", tags=["t1"])
    trace = SwarmTrace(messages=[], final_answer="A")
    result = SwarmResult(
        test_case=tc,
        trace=trace,
        outcome_score=80,
        process_score=70,
        routing_score=90,
        overall_score=75,
        status="warn",
    )
    d = result.to_dict()
    assert d["question"] == "Q"
    assert d["overall_score"] == 75
    assert d["process_breakdown"]["routing"] == 90


# ---- SwarmSummary ----

def test_summary_pass_rate():
    summary = SwarmSummary(suite_name="test", total=4, passed=3, warned=1)
    assert summary.pass_rate == 75.0


def test_summary_empty():
    summary = SwarmSummary(suite_name="test")
    assert summary.pass_rate == 0.0


def test_summary_to_dict():
    summary = SwarmSummary(
        suite_name="test",
        total=2,
        passed=1,
        warned=0,
        failed=1,
        overall_score=65.0,
        reliability_score=68.0,
        reliability_grade="C",
    )
    d = summary.to_dict()
    assert d["suite_name"] == "test"
    assert d["reliability_score"] == 68.0
    assert d["reliability_grade"] == "C"


def test_summary_by_tag():
    tc1 = SwarmTestCase(question="Q1", tags=["billing"])
    tc2 = SwarmTestCase(question="Q2", tags=["tech"])
    trace = SwarmTrace(messages=[], final_answer="A")
    r1 = SwarmResult(test_case=tc1, trace=trace)
    r2 = SwarmResult(test_case=tc2, trace=trace)
    summary = SwarmSummary(suite_name="test", results=[r1, r2])
    assert len(summary.by_tag("billing")) == 1
    assert len(summary.by_tag("other")) == 0
