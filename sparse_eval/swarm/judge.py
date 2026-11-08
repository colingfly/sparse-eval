"""
judge.py -- Swarm evaluation judge.

Evaluates multi-agent interactions on both outcome quality (reusing the
existing Judge for final answer scoring) and process quality (routing,
handoff, context preservation, convergence) using swarm-specific criteria.
"""

from typing import Optional

from sparse_eval.judge import Judge
from sparse_eval.swarm.models import (
    SwarmTestCase,
    SwarmTrace,
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


# Process criteria instances (stateless, reusable)
_ROUTING = RoutingCriterion()
_HANDOFF = HandoffCriterion()
_CONTEXT = ContextPreservationCriterion()
_CONVERGENCE = ConvergenceCriterion()
_COLLABORATION = CollaborationCriterion()


SWARM_JUDGE_SYSTEM = """You are an expert evaluator assessing a multi-agent AI system's response quality.
You will score the swarm's final consolidated answer on specific criteria, each from 0 to 100.
Consider not only the final answer but also how well the agents worked together.

Scoring guidelines:
- 90-100: Excellent. Accurate, complete, well-coordinated.
- 70-89: Good. Mostly correct, minor gaps.
- 50-69: Partial. Some correct info but significant gaps.
- 30-49: Poor. Major inaccuracies or coordination failures.
- 0-29: Failing. Wrong, hallucinated, or complete breakdown.

Respond ONLY with valid JSON, no markdown, no backticks."""


class SwarmJudge:
    """
    Evaluates a swarm trace on both outcome and process quality.

    Outcome scoring reuses the existing Judge for final answer evaluation.
    Process scoring uses deterministic swarm-specific criteria plugins.

    Usage:
        judge = SwarmJudge(api_key="sk-ant-...")
        result = judge.evaluate(test_case, trace, criteria, topology_type="orchestrator")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929",
        provider: str = "anthropic",
        base_url: Optional[str] = None,
    ):
        self._outcome_judge = Judge(
            api_key=api_key,
            model=model,
            provider=provider,
            base_url=base_url,
        )

    def evaluate(
        self,
        test_case: SwarmTestCase,
        trace: SwarmTrace,
        criteria: list[dict],
        custom_rules: Optional[list[str]] = None,
        topology_type: str = "orchestrator",
    ) -> SwarmResult:
        """
        Full evaluation of a swarm interaction.

        Args:
            test_case: The test scenario with expectations
            trace: The interaction trace to evaluate
            criteria: List of criteria dicts (key, weight, label, description)
            custom_rules: Optional evaluation rules
            topology_type: "orchestrator" or "collaborative"

        Returns:
            SwarmResult with outcome, process, and per-criterion scores
        """
        # Build criteria weight map
        weights = {}
        total_weight = 0
        for c in criteria:
            w = c.get("weight", 50)
            weights[c["key"]] = w
            total_weight += w

        # 1. Outcome score: use existing judge on final answer
        outcome_score = self._evaluate_outcome(test_case, trace, criteria, custom_rules)

        # 2. Process scores: deterministic criteria
        ctx = SwarmEvalContext(
            test_case=test_case,
            trace=trace,
            topology_type=topology_type,
        )

        routing_score = _ROUTING.score(ctx) if "routing" in weights else None
        handoff_score = _HANDOFF.score(ctx) if "handoff" in weights else None
        context_score = _CONTEXT.score(ctx) if "context_preservation" in weights else None
        convergence_score = _CONVERGENCE.score(ctx) if "convergence" in weights else None
        collaboration_score = _COLLABORATION.score(ctx) if "collaboration" in weights else None

        # 3. Compute weighted process score from available process criteria
        process_scores = {}
        if routing_score is not None:
            process_scores["routing"] = routing_score
        if handoff_score is not None:
            process_scores["handoff"] = handoff_score
        if context_score is not None:
            process_scores["context_preservation"] = context_score
        if convergence_score is not None:
            process_scores["convergence"] = convergence_score
        if collaboration_score is not None:
            process_scores["collaboration"] = collaboration_score

        process_score = self._weighted_avg(process_scores, weights)

        # 4. Compute overall weighted score across all criteria
        all_scores = {"outcome": outcome_score}
        all_scores.update(process_scores)
        overall_score = self._weighted_avg(all_scores, weights)

        # 5. Per-agent scores
        per_agent = self._compute_per_agent_scores(trace)

        # 6. Build result
        status = SwarmResult.classify(overall_score)

        reasoning_parts = [f"Outcome: {outcome_score:.0f}"]
        if process_scores:
            reasoning_parts.append(f"Process: {process_score:.0f}")
            for k, v in process_scores.items():
                reasoning_parts.append(f"  {k}: {v:.0f}")
        reasoning = " | ".join(reasoning_parts[:3])

        return SwarmResult(
            test_case=test_case,
            trace=trace,
            outcome_score=round(outcome_score, 1),
            process_score=round(process_score, 1),
            routing_score=round(routing_score or 0, 1),
            handoff_score=round(handoff_score or 0, 1),
            context_score=round(context_score or 0, 1),
            convergence_score=round(convergence_score or 0, 1),
            per_agent_scores=per_agent,
            overall_score=round(overall_score, 1),
            status=status,
            reasoning=reasoning,
        )

    def _evaluate_outcome(
        self,
        test_case: SwarmTestCase,
        trace: SwarmTrace,
        criteria: list[dict],
        custom_rules: Optional[list[str]] = None,
    ) -> float:
        """Score the final answer using the existing Judge."""
        if not test_case.expected_answer:
            # No expected answer — can't judge outcome meaningfully
            return 50.0

        # Use only outcome-related criteria for the judge
        outcome_criteria = [
            {"key": "accuracy", "label": "Accuracy", "description": "Factual correctness", "weight": 40, "is_enabled": True},
            {"key": "completeness", "label": "Completeness", "description": "Covers all key points", "weight": 30, "is_enabled": True},
            {"key": "hallucination", "label": "Hallucination", "description": "No fabricated info", "weight": 30, "is_enabled": True},
        ]

        # Add trace context to custom rules so judge is aware of multi-agent context
        enhanced_rules = list(custom_rules or [])
        enhanced_rules.append(
            "This is a multi-agent system. Evaluate the FINAL consolidated answer, "
            "not intermediate agent responses."
        )

        try:
            judge_result = self._outcome_judge.score(
                question=test_case.question,
                agent_answer=trace.final_answer,
                expected_answer=test_case.expected_answer,
                criteria=outcome_criteria,
                custom_rules=enhanced_rules,
                context=test_case.context,
            )
            return judge_result.overall_score
        except Exception:
            return 50.0  # Fallback on judge failure

    @staticmethod
    def _weighted_avg(scores: dict[str, float], weights: dict[str, int]) -> float:
        """Compute weighted average of scores using criteria weights."""
        if not scores:
            return 0.0

        total_weight = 0
        weighted_sum = 0.0
        for key, score in scores.items():
            w = weights.get(key, 50)
            weighted_sum += score * w
            total_weight += w

        return round(weighted_sum / total_weight, 1) if total_weight > 0 else 0.0

    @staticmethod
    def _compute_per_agent_scores(trace: SwarmTrace) -> dict[str, float]:
        """
        Compute a simple per-agent participation score.

        Based on: whether the agent contributed, message count relative to total.
        """
        agents = trace.agents_involved
        if not agents:
            return {}

        total_agent_msgs = sum(
            1 for m in trace.messages if m.from_agent != "user"
        )
        if total_agent_msgs == 0:
            return {a: 0.0 for a in agents}

        per_agent = {}
        for agent_id in agents:
            msgs = trace.messages_by_agent(agent_id)
            # Score: participated at all (base 50) + contribution proportion (up to 50)
            contribution = len(msgs) / total_agent_msgs
            per_agent[agent_id] = round(50 + contribution * 50, 1)

        return per_agent
