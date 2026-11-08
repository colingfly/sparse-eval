"""
engine.py -- Swarm reliability runner.

Orchestrates running a swarm test suite, evaluating traces,
and producing a swarm summary with reliability scoring.

Supports two modes:
- swarm-trace: Evaluates pre-recorded interaction traces
- swarm: Executes agents via SwarmRunner, captures traces, then evaluates
"""

import asyncio
import time
from datetime import datetime
from typing import Optional, Callable

from sparse_eval.swarm.models import (
    SwarmTestCase,
    SwarmTrace,
    SwarmResult,
    SwarmSummary,
)
from sparse_eval.swarm.suite import SwarmSuite
from sparse_eval.swarm.judge import SwarmJudge
from sparse_eval.swarm.reliability import SwarmReliabilityConfig, compute_swarm_reliability


class SwarmReliabilityRunner:
    """
    Run a swarm test suite and collect evaluation results.

    For trace-based evaluation (mode: swarm-trace), traces are loaded from
    the suite and evaluated. For orchestrated evaluation (mode: swarm),
    SwarmRunner executes agents and captures traces before evaluation.

    Usage:
        suite = SwarmSuite.from_yaml("swarm_tests.yaml")
        runner = SwarmReliabilityRunner(api_key="sk-ant-...")
        summary = runner.run(suite)

        # With callable agents (orchestrated mode):
        summary = runner.run(suite, agent_callables={"router": my_router, "billing": my_billing})
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        verbose: bool = True,
        on_result: Optional[Callable[[SwarmResult, int, int], None]] = None,
        provider: str = "anthropic",
        base_url: Optional[str] = None,
        concurrency: int = 1,
        reliability_config: Optional[SwarmReliabilityConfig] = None,
        agent_callables: Optional[dict[str, Callable[[str], str]]] = None,
    ):
        self.api_key = api_key
        self.model_override = model
        self.verbose = verbose
        self.on_result = on_result
        self.provider = provider
        self.base_url = base_url
        self.concurrency = concurrency
        self.reliability_config = reliability_config
        self.agent_callables = agent_callables or {}

    def run(
        self,
        suite: SwarmSuite,
        tags: Optional[list[str]] = None,
        agent_callables: Optional[dict[str, Callable[[str], str]]] = None,
    ) -> SwarmSummary:
        """
        Execute swarm eval run.

        Args:
            suite: Swarm test suite to run
            tags: Optional tag filter
            agent_callables: Optional dict of agent_id -> callable for orchestrated mode.
                             Merges with self.agent_callables (run-level takes priority).

        Returns:
            SwarmSummary with all results
        """
        # Merge callables: instance-level + run-level
        callables = dict(self.agent_callables)
        if agent_callables:
            callables.update(agent_callables)

        if self.concurrency > 1:
            try:
                return asyncio.run(self.run_async(suite, tags=tags, agent_callables=callables))
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    raise RuntimeError(
                        "Cannot use concurrency > 1 inside an existing event loop. "
                        "Use `await runner.run_async(suite)` instead."
                    ) from e
                raise

        return self._run_sequential(suite, tags=tags, agent_callables=callables)

    def _run_sequential(
        self,
        suite: SwarmSuite,
        tags: Optional[list[str]] = None,
        agent_callables: Optional[dict[str, Callable[[str], str]]] = None,
    ) -> SwarmSummary:
        """Execute swarm eval run sequentially."""
        judge = SwarmJudge(
            api_key=self.api_key,
            model=self.model_override or suite.model,
            provider=self.provider,
            base_url=self.base_url,
        )

        tests = suite.filter_by_tags(tags) if tags else suite.tests
        if not tests:
            return SwarmSummary(suite_name=suite.name)

        criteria = suite.criteria_dicts()
        custom_rules = suite.custom_rules
        topology_type = suite.topology.type.value

        # Create SwarmRunner for orchestrated mode
        swarm_runner = None
        if suite.mode == "swarm":
            from sparse_eval.swarm.runner import SwarmRunner
            swarm_runner = SwarmRunner(
                topology=suite.topology,
                agent_callables=agent_callables or {},
                verbose=self.verbose,
            )

        start_time = time.time()
        started_at = datetime.utcnow().isoformat()

        results = []

        for i, tc in enumerate(tests):
            if self.verbose:
                q_preview = tc.question[:60] + "..." if len(tc.question) > 60 else tc.question
                print(f"  [{i+1}/{len(tests)}] {q_preview}")

            # Get trace: pre-recorded or execute via SwarmRunner
            trace = tc.trace
            if trace is None and swarm_runner is not None:
                try:
                    trace = swarm_runner.execute(tc)
                except Exception as e:
                    if self.verbose:
                        print(f"    Execution error: {e}")
                    trace = SwarmTrace(messages=[], final_answer=f"Execution failed: {e}")
            if trace is None:
                if self.verbose:
                    print(f"    [SKIP] No trace available for this test case")
                continue

            # Evaluate
            try:
                result = judge.evaluate(
                    test_case=tc,
                    trace=trace,
                    criteria=criteria,
                    custom_rules=custom_rules,
                    topology_type=topology_type,
                )
            except Exception as e:
                if self.verbose:
                    print(f"    Judge error: {e}")
                result = SwarmResult(
                    test_case=tc,
                    trace=trace,
                    overall_score=50,
                    status="warn",
                    reasoning=f"Evaluation failed: {str(e)}",
                )

            results.append(result)

            if self.verbose:
                status_icon = {"pass": "P", "warn": "W", "fail": "F"}.get(result.status, "?")
                print(f"    Score: {result.overall_score} ({status_icon}) "
                      f"[outcome={result.outcome_score}, process={result.process_score}]")

            if self.on_result:
                self.on_result(result, i + 1, len(tests))

        return self._build_summary(suite, results, start_time, started_at)

    async def run_async(
        self,
        suite: SwarmSuite,
        tags: Optional[list[str]] = None,
        agent_callables: Optional[dict[str, Callable[[str], str]]] = None,
    ) -> SwarmSummary:
        """Execute swarm eval with async concurrency."""
        judge = SwarmJudge(
            api_key=self.api_key,
            model=self.model_override or suite.model,
            provider=self.provider,
            base_url=self.base_url,
        )

        tests = suite.filter_by_tags(tags) if tags else suite.tests
        if not tests:
            return SwarmSummary(suite_name=suite.name)

        criteria = suite.criteria_dicts()
        custom_rules = suite.custom_rules
        topology_type = suite.topology.type.value

        # Create SwarmRunner for orchestrated mode
        swarm_runner = None
        if suite.mode == "swarm":
            from sparse_eval.swarm.runner import SwarmRunner
            swarm_runner = SwarmRunner(
                topology=suite.topology,
                agent_callables=agent_callables or {},
                verbose=self.verbose,
            )

        start_time = time.time()
        started_at = datetime.utcnow().isoformat()

        semaphore = asyncio.Semaphore(self.concurrency)

        async def _run_single(i: int, tc: SwarmTestCase) -> Optional[SwarmResult]:
            async with semaphore:
                if self.verbose:
                    q_preview = tc.question[:60] + "..." if len(tc.question) > 60 else tc.question
                    print(f"  [{i+1}/{len(tests)}] {q_preview}")

                trace = tc.trace
                if trace is None and swarm_runner is not None:
                    try:
                        trace = await asyncio.to_thread(swarm_runner.execute, tc)
                    except Exception as e:
                        if self.verbose:
                            print(f"    Execution error: {e}")
                        trace = SwarmTrace(messages=[], final_answer=f"Execution failed: {e}")
                if trace is None:
                    if self.verbose:
                        print(f"    [SKIP] No trace available")
                    return None

                try:
                    result = await asyncio.to_thread(
                        judge.evaluate,
                        test_case=tc,
                        trace=trace,
                        criteria=criteria,
                        custom_rules=custom_rules,
                        topology_type=topology_type,
                    )
                except Exception as e:
                    if self.verbose:
                        print(f"    Judge error: {e}")
                    result = SwarmResult(
                        test_case=tc,
                        trace=trace,
                        overall_score=50,
                        status="warn",
                        reasoning=f"Evaluation failed: {str(e)}",
                    )

                if self.verbose:
                    status_icon = {"pass": "P", "warn": "W", "fail": "F"}.get(result.status, "?")
                    print(f"    Score: {result.overall_score} ({status_icon})")

                if self.on_result:
                    self.on_result(result, i + 1, len(tests))

                return result

        tasks = [_run_single(i, tc) for i, tc in enumerate(tests)]
        raw_results = await asyncio.gather(*tasks)
        results = [r for r in raw_results if r is not None]

        return self._build_summary(suite, results, start_time, started_at)

    def _build_summary(
        self,
        suite: SwarmSuite,
        results: list[SwarmResult],
        start_time: float,
        started_at: str,
    ) -> SwarmSummary:
        """Build a SwarmSummary from results."""
        duration = time.time() - start_time

        if not results:
            return SwarmSummary(
                suite_name=suite.name,
                topology_type=suite.topology.type.value,
                duration_seconds=duration,
                started_at=started_at,
                completed_at=datetime.utcnow().isoformat(),
            )

        overall = round(sum(r.overall_score for r in results) / len(results), 1)
        outcome_avg = round(sum(r.outcome_score for r in results) / len(results), 1)
        process_avg = round(sum(r.process_score for r in results) / len(results), 1)

        # Per-agent summary: aggregate scores across all results
        agent_scores: dict[str, list[float]] = {}
        for r in results:
            for agent_id, score in r.per_agent_scores.items():
                agent_scores.setdefault(agent_id, []).append(score)

        per_agent_summary = {}
        for agent_id, scores in agent_scores.items():
            agent = suite.topology.get_agent(agent_id)
            per_agent_summary[agent_id] = {
                "role": agent.role if agent else agent_id,
                "avg_score": round(sum(scores) / len(scores), 1),
                "appearances": len(scores),
            }

        # Reliability score
        config = self.reliability_config or suite.get_reliability_config()
        reliability_score, reliability_grade = compute_swarm_reliability(
            outcome_score=outcome_avg,
            process_score=process_avg,
            config=config,
        )

        summary = SwarmSummary(
            suite_name=suite.name,
            topology_type=suite.topology.type.value,
            total=len(results),
            passed=sum(1 for r in results if r.status == "pass"),
            warned=sum(1 for r in results if r.status == "warn"),
            failed=sum(1 for r in results if r.status == "fail"),
            overall_score=overall,
            outcome_avg=outcome_avg,
            process_avg=process_avg,
            results=results,
            per_agent_summary=per_agent_summary,
            reliability_score=reliability_score,
            reliability_grade=reliability_grade,
            duration_seconds=duration,
            started_at=started_at,
            completed_at=datetime.utcnow().isoformat(),
        )

        if self.verbose:
            print(f"\n  Done: {summary.overall_score} "
                  f"({summary.passed}P/{summary.warned}W/{summary.failed}F) "
                  f"in {duration:.1f}s")

        return summary
