"""
criteria.py -- Swarm-specific evaluation criteria.

Provides criteria plugins for evaluating multi-agent orchestration quality:
routing correctness, handoff quality, context preservation, and convergence.
"""

from cane_eval.criteria import CriterionPlugin, CriterionType
from cane_eval.swarm.models import (
    SwarmTrace,
    SwarmTestCase,
    SwarmMessageRole,
)
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SwarmEvalContext:
    """All data available to a swarm criterion for scoring."""
    test_case: Optional[SwarmTestCase] = None
    trace: Optional[SwarmTrace] = None
    topology_type: str = "orchestrator"
    metadata: dict = field(default_factory=dict)


class RoutingCriterion(CriterionPlugin):
    """
    Evaluates whether queries were routed to the correct agent(s).

    Deterministic component: compares actual route against expected_route.
    Score is based on overlap between expected and actual routing paths.
    """
    name = "routing"
    criterion_type = CriterionType.DETERMINISTIC
    weight = 1.0
    description = "Was the query routed to the correct specialist agent?"

    def score(self, ctx) -> float:
        if not isinstance(ctx, SwarmEvalContext):
            return 50.0

        tc = ctx.test_case
        trace = ctx.trace
        if not tc or not trace:
            return 50.0

        # If no expected route defined, skip (neutral score)
        if not tc.expected_route:
            if tc.expected_agents:
                return self._score_agent_involvement(tc.expected_agents, trace)
            return 100.0  # No expectations = pass

        actual_route = trace.actual_route
        expected = tc.expected_route

        if not actual_route:
            return 0.0

        # Score based on sequence matching
        # Full match = 100, partial match = proportional, no overlap = 0
        matches = 0
        for i, expected_id in enumerate(expected):
            if i < len(actual_route) and actual_route[i] == expected_id:
                matches += 1
            elif expected_id in actual_route:
                matches += 0.5  # Present but wrong position

        return round(min(100.0, (matches / len(expected)) * 100), 1)

    @staticmethod
    def _score_agent_involvement(expected_agents: list[str], trace: SwarmTrace) -> float:
        """Score based on whether expected agents were involved."""
        involved = set(trace.agents_involved)
        expected = set(expected_agents)
        if not expected:
            return 100.0
        overlap = expected & involved
        return round((len(overlap) / len(expected)) * 100, 1)


class HandoffCriterion(CriterionPlugin):
    """
    Evaluates the quality of handoffs between agents.

    Checks whether handoff messages contain sufficient context
    for the receiving agent to continue the conversation.
    """
    name = "handoff"
    criterion_type = CriterionType.DETERMINISTIC
    weight = 1.0
    description = "Was sufficient context passed during agent handoffs?"

    # Minimum content length for a handoff to be considered contextual
    MIN_HANDOFF_CONTENT_LENGTH = 10

    def score(self, ctx) -> float:
        if not isinstance(ctx, SwarmEvalContext):
            return 50.0

        trace = ctx.trace
        if not trace:
            return 50.0

        handoffs = [m for m in trace.messages if m.role == SwarmMessageRole.HANDOFF]

        if not handoffs:
            # No handoffs — could be fine for simple queries
            if trace.num_turns <= 2:
                return 100.0
            # Multi-turn without explicit handoffs might indicate poor structure
            return 70.0

        # Score each handoff
        scores = []
        for h in handoffs:
            s = 100.0
            # Penalize empty or minimal handoff content
            if not h.content.strip():
                s = 0.0
            elif len(h.content.strip()) < self.MIN_HANDOFF_CONTENT_LENGTH:
                s = 30.0
            # Bonus if handoff includes metadata (routing reason, context)
            elif h.metadata.get("routing_reason") or h.metadata.get("context"):
                s = 100.0
            else:
                # Content exists but no structured metadata — decent
                s = 80.0
            scores.append(s)

        return round(sum(scores) / len(scores), 1)


class ContextPreservationCriterion(CriterionPlugin):
    """
    Evaluates whether agents maintained conversation context
    across the interaction.

    Checks that the original question's key terms appear in
    subsequent agent responses, indicating context was preserved.
    """
    name = "context_preservation"
    criterion_type = CriterionType.DETERMINISTIC
    weight = 1.0
    description = "Did agents maintain conversation context throughout the interaction?"

    def score(self, ctx) -> float:
        if not isinstance(ctx, SwarmEvalContext):
            return 50.0

        tc = ctx.test_case
        trace = ctx.trace
        if not tc or not trace:
            return 50.0

        if trace.num_turns <= 1:
            return 100.0  # Trivial case

        # Extract key terms from the question (words > 3 chars, lowered)
        question_terms = set(
            w.lower().strip("?.,!\"'")
            for w in tc.question.split()
            if len(w) > 3
        )
        # Remove common stop words
        stop_words = {"what", "that", "this", "with", "from", "your", "have", "been", "does", "will", "about", "there"}
        question_terms -= stop_words

        if not question_terms:
            return 100.0

        # Check if the final answer references key terms from the question
        final_lower = trace.final_answer.lower()
        terms_in_final = sum(1 for t in question_terms if t in final_lower)
        final_coverage = terms_in_final / len(question_terms) if question_terms else 1.0

        # Check if handoff messages preserve context
        handoff_msgs = [m for m in trace.messages if m.role == SwarmMessageRole.HANDOFF]
        if handoff_msgs:
            handoff_coverage_scores = []
            for h in handoff_msgs:
                h_lower = h.content.lower()
                terms_in_handoff = sum(1 for t in question_terms if t in h_lower)
                handoff_coverage_scores.append(terms_in_handoff / len(question_terms))
            handoff_coverage = sum(handoff_coverage_scores) / len(handoff_coverage_scores)
        else:
            handoff_coverage = 1.0  # No handoffs to evaluate

        # Weighted: 60% final answer relevance, 40% handoff context
        score = (final_coverage * 60 + handoff_coverage * 40)
        return round(min(100.0, score), 1)


class ConvergenceCriterion(CriterionPlugin):
    """
    Evaluates whether the swarm reached an answer efficiently.

    Penalizes: exceeding max_turns, circular routing, no final answer.
    """
    name = "convergence"
    criterion_type = CriterionType.DETERMINISTIC
    weight = 1.0
    description = "Did the swarm reach an answer efficiently without unnecessary turns?"

    def score(self, ctx) -> float:
        if not isinstance(ctx, SwarmEvalContext):
            return 50.0

        tc = ctx.test_case
        trace = ctx.trace
        if not tc or not trace:
            return 50.0

        score = 100.0

        # No final answer = critical failure
        if not trace.final_answer.strip():
            return 0.0

        # Penalize exceeding max_turns
        if tc.max_turns and trace.num_turns > tc.max_turns:
            excess = trace.num_turns - tc.max_turns
            score -= excess * 15  # -15 per excess turn

        # Detect circular routing: self-sends or consecutive same-sender
        circular_count = 0
        non_user_msgs = [m for m in trace.messages if m.from_agent != "user"]
        for i, m in enumerate(non_user_msgs):
            # Self-send (agent routes to itself)
            if m.from_agent == m.to_agent:
                circular_count += 1
            # Consecutive same sender
            elif i > 0 and non_user_msgs[i].from_agent == non_user_msgs[i - 1].from_agent:
                circular_count += 1
        if circular_count > 0:
            score -= circular_count * 20  # -20 per circular routing

        # Penalize very long traces (more than 2x expected)
        if tc.max_turns:
            if trace.num_turns > tc.max_turns * 2:
                score -= 30

        return round(max(0.0, min(100.0, score)), 1)


class CollaborationCriterion(CriterionPlugin):
    """
    Evaluates collaboration quality for peer-based topologies.

    Checks that multiple agents contributed and that contributions
    were distinct (not redundant).
    """
    name = "collaboration"
    criterion_type = CriterionType.DETERMINISTIC
    weight = 1.0
    description = "Did agents collaborate effectively and build on each other's contributions?"

    def score(self, ctx) -> float:
        if not isinstance(ctx, SwarmEvalContext):
            return 50.0

        # Only meaningful for collaborative topology
        if ctx.topology_type != "collaborative":
            return 100.0

        trace = ctx.trace
        if not trace:
            return 50.0

        agents = trace.agents_involved
        if len(agents) <= 1:
            return 30.0  # Only one agent participated — poor collaboration

        # Check contribution balance (all agents should contribute)
        agent_msg_counts = {}
        for m in trace.messages:
            if m.from_agent != "user":
                agent_msg_counts[m.from_agent] = agent_msg_counts.get(m.from_agent, 0) + 1

        if not agent_msg_counts:
            return 0.0

        counts = list(agent_msg_counts.values())
        max_msgs = max(counts)
        min_msgs = min(counts)

        # Balanced participation: ratio of min to max
        balance = min_msgs / max_msgs if max_msgs > 0 else 0
        balance_score = balance * 100

        # Multi-agent involvement
        involvement_score = min(100.0, (len(agents) / 2) * 100)

        return round((balance_score * 0.5 + involvement_score * 0.5), 1)
