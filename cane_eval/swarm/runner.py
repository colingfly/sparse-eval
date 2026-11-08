"""
runner.py -- Swarm orchestrated execution.

Executes multi-agent workflows by calling agents according to the
topology, capturing the full interaction trace for evaluation.

Supports two topologies:
- Orchestrator: Hub-and-spoke. Orchestrator routes to specialists.
- Collaborative: Peer-to-peer. Agents exchange messages round-robin.
"""

import json
import time
import subprocess
import urllib.request
from typing import Optional, Callable

from cane_eval.swarm.models import (
    SwarmAgent,
    SwarmTopology,
    SwarmTopologyType,
    SwarmMessage,
    SwarmMessageRole,
    SwarmTrace,
    SwarmTestCase,
)


# ---- Agent callers ----

def _extract_by_path(data, path: str) -> str:
    """Extract a value from nested dict using dot notation."""
    if not path:
        return str(data) if data else ""
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part, "")
        else:
            return str(current)
    return str(current) if current else ""


def _call_agent_http(agent: SwarmAgent, message: str) -> dict:
    """Call an agent via HTTP."""
    start = time.time()
    try:
        payload_str = agent.payload_template.replace("{{message}}", message)
        payload_bytes = payload_str.encode("utf-8")

        headers = dict(agent.headers)
        headers.setdefault("Content-Type", "application/json")

        req = urllib.request.Request(
            agent.url,
            data=payload_bytes if agent.method.upper() == "POST" else None,
            headers=headers,
            method=agent.method.upper(),
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")

        try:
            resp_data = json.loads(body)
            answer = _extract_by_path(resp_data, agent.response_path)
        except json.JSONDecodeError:
            answer = body.strip()

        elapsed = int((time.time() - start) * 1000)
        return {"answer": answer or "(empty response)", "time_ms": elapsed}

    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        return {"answer": f"Error calling agent: {e}", "time_ms": elapsed}


def _call_agent_command(agent: SwarmAgent, message: str) -> dict:
    """Call an agent via CLI command."""
    start = time.time()
    try:
        cmd = agent.command.replace("{{message}}", message)
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=60,
        )
        answer = result.stdout.strip()
        if not answer:
            answer = result.stderr.strip() or "(no output)"

        elapsed = int((time.time() - start) * 1000)
        return {"answer": answer, "time_ms": elapsed}

    except subprocess.TimeoutExpired:
        elapsed = int((time.time() - start) * 1000)
        return {"answer": "Error: command timed out", "time_ms": elapsed}
    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        return {"answer": f"Error: {e}", "time_ms": elapsed}


# ---- Orchestrator response parsing ----

def _parse_orchestrator_response(raw: str) -> dict:
    """
    Parse an orchestrator's response to extract routing decisions.

    Expected format (JSON):
        {"route_to": "agent_id", "context": "...", "direct_answer": null}

    If the orchestrator returns a direct_answer, no routing occurs.
    If it returns route_to, the question is forwarded to that agent.

    Falls back to treating the entire response as a direct answer if
    JSON parsing fails.
    """
    cleaned = raw.strip()
    # Strip markdown fences
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return {
                "route_to": parsed.get("route_to"),
                "context": parsed.get("context", ""),
                "direct_answer": parsed.get("direct_answer"),
            }
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: treat as direct answer
    return {"route_to": None, "context": "", "direct_answer": raw}


# ---- SwarmRunner ----

class SwarmRunner:
    """
    Executes multi-agent workflows and captures interaction traces.

    For ORCHESTRATOR topology:
      1. Send question to orchestrator
      2. Orchestrator returns routing decision (JSON with route_to)
      3. Forward to specialist with context
      4. Specialist responds
      5. Orchestrator synthesizes final answer
      6. Full trace captured

    For COLLABORATIVE topology:
      1. Send question to all agents
      2. Agents produce initial responses
      3. Agents exchange responses round-robin
      4. After max_rounds, combine into final answer
      5. Full trace captured

    Usage:
        runner = SwarmRunner(topology, agent_callables={"router": my_router, "billing": my_billing})
        trace = runner.execute(test_case)
    """

    def __init__(
        self,
        topology: SwarmTopology,
        agent_callables: Optional[dict[str, Callable[[str], str]]] = None,
        max_rounds: int = 3,
        verbose: bool = False,
    ):
        """
        Args:
            topology: Swarm topology definition with agent configs
            agent_callables: Dict of agent_id -> callable(message) -> response.
                             Overrides HTTP/command configs for agents with callables.
            max_rounds: Max rounds for collaborative topology (default: 3)
            verbose: Print execution progress
        """
        self.topology = topology
        self.agent_callables = agent_callables or {}
        self.max_rounds = max_rounds
        self.verbose = verbose

    def execute(self, test_case: SwarmTestCase) -> SwarmTrace:
        """Run one test case through the swarm, return the captured trace."""
        if self.topology.type == SwarmTopologyType.ORCHESTRATOR:
            return self._execute_orchestrator(test_case)
        return self._execute_collaborative(test_case)

    def _call_agent(self, agent: SwarmAgent, message: str) -> dict:
        """
        Call a single agent. Uses callable if provided, else HTTP/command.

        Returns:
            {"answer": str, "time_ms": int}
        """
        # Check for callable override
        if agent.id in self.agent_callables:
            start = time.time()
            try:
                answer = self.agent_callables[agent.id](message)
                elapsed = int((time.time() - start) * 1000)
                return {"answer": str(answer), "time_ms": elapsed}
            except Exception as e:
                elapsed = int((time.time() - start) * 1000)
                return {"answer": f"Agent error: {e}", "time_ms": elapsed}

        # HTTP
        if agent.type == "http" and agent.url:
            return _call_agent_http(agent, message)

        # Command
        if agent.type == "command" and agent.command:
            return _call_agent_command(agent, message)

        return {"answer": f"(No target configured for agent '{agent.id}')", "time_ms": 0}

    def _execute_orchestrator(self, tc: SwarmTestCase) -> SwarmTrace:
        """Hub-and-spoke execution with orchestrator routing."""
        messages = []
        total_start = time.time()

        orchestrator_id = self.topology.orchestrator_id
        if not orchestrator_id:
            # Default: first agent in topology
            orchestrator_id = self.topology.agents[0].id if self.topology.agents else None
        if not orchestrator_id:
            return SwarmTrace(messages=[], final_answer="(No orchestrator configured)")

        orchestrator = self.topology.get_agent(orchestrator_id)
        if not orchestrator:
            return SwarmTrace(messages=[], final_answer=f"(Orchestrator '{orchestrator_id}' not found)")

        # Step 1: User sends question to orchestrator
        messages.append(SwarmMessage(
            from_agent="user",
            to_agent=orchestrator_id,
            role=SwarmMessageRole.USER,
            content=tc.question,
            timestamp_ms=int((time.time() - total_start) * 1000),
        ))

        if self.verbose:
            print(f"      user -> {orchestrator_id}: {tc.question[:60]}")

        # Step 2: Orchestrator processes and routes
        # Build a prompt that tells the orchestrator about available agents
        agent_descriptions = []
        for a in self.topology.agents:
            if a.id != orchestrator_id:
                caps = f" (capabilities: {', '.join(a.capabilities)})" if a.capabilities else ""
                agent_descriptions.append(f"- {a.id}: {a.role}{caps}")

        orchestrator_prompt = (
            f"You are an orchestrator routing customer queries to specialist agents.\n\n"
            f"Available agents:\n" + "\n".join(agent_descriptions) + "\n\n"
            f"Customer question: {tc.question}\n\n"
            f"Respond with JSON: {{\"route_to\": \"agent_id\", \"context\": \"brief context for the specialist\", \"direct_answer\": null}}\n"
            f"If you can answer directly without a specialist, set route_to to null and provide direct_answer."
        )

        orch_result = self._call_agent(orchestrator, orchestrator_prompt)
        routing = _parse_orchestrator_response(orch_result["answer"])

        # Step 3: Route to specialist or return direct answer
        max_hops = len(self.topology.agents)  # Prevent infinite routing loops
        hop_count = 0
        final_answer = ""

        while hop_count < max_hops:
            hop_count += 1

            if routing.get("direct_answer"):
                # Orchestrator answered directly
                final_answer = routing["direct_answer"]
                messages.append(SwarmMessage(
                    from_agent=orchestrator_id,
                    to_agent="user",
                    role=SwarmMessageRole.FINAL,
                    content=final_answer,
                    timestamp_ms=int((time.time() - total_start) * 1000),
                ))
                if self.verbose:
                    print(f"      {orchestrator_id} -> user [final]: {final_answer[:60]}")
                break

            target_id = routing.get("route_to")
            if not target_id:
                # No routing, no direct answer — treat raw response as answer
                final_answer = orch_result["answer"]
                messages.append(SwarmMessage(
                    from_agent=orchestrator_id,
                    to_agent="user",
                    role=SwarmMessageRole.FINAL,
                    content=final_answer,
                    timestamp_ms=int((time.time() - total_start) * 1000),
                ))
                break

            target_agent = self.topology.get_agent(target_id)
            if not target_agent:
                final_answer = f"(Orchestrator routed to unknown agent '{target_id}')"
                messages.append(SwarmMessage(
                    from_agent=orchestrator_id,
                    to_agent="user",
                    role=SwarmMessageRole.FINAL,
                    content=final_answer,
                    timestamp_ms=int((time.time() - total_start) * 1000),
                ))
                break

            # Record handoff
            handoff_content = routing.get("context") or f"Routing to {target_id}"
            messages.append(SwarmMessage(
                from_agent=orchestrator_id,
                to_agent=target_id,
                role=SwarmMessageRole.HANDOFF,
                content=handoff_content,
                timestamp_ms=int((time.time() - total_start) * 1000),
                metadata={"routing_reason": routing.get("context", "")},
            ))
            if self.verbose:
                print(f"      {orchestrator_id} -> {target_id} [handoff]: {handoff_content[:60]}")

            # Step 4: Call specialist
            specialist_prompt = (
                f"Context from orchestrator: {handoff_content}\n\n"
                f"Customer question: {tc.question}\n\n"
                f"Please provide a helpful response."
            )
            specialist_result = self._call_agent(target_agent, specialist_prompt)

            messages.append(SwarmMessage(
                from_agent=target_id,
                to_agent=orchestrator_id,
                role=SwarmMessageRole.AGENT,
                content=specialist_result["answer"],
                timestamp_ms=int((time.time() - total_start) * 1000),
            ))
            if self.verbose:
                print(f"      {target_id} -> {orchestrator_id}: {specialist_result['answer'][:60]}")

            # Step 5: Orchestrator synthesizes final answer
            synthesis_prompt = (
                f"Original question: {tc.question}\n\n"
                f"Response from {target_id} ({target_agent.role}): {specialist_result['answer']}\n\n"
                f"Synthesize a final answer for the customer. "
                f"Respond with JSON: {{\"route_to\": null, \"context\": null, \"direct_answer\": \"your final answer\"}}\n"
                f"Or route to another agent if needed: {{\"route_to\": \"agent_id\", \"context\": \"...\", \"direct_answer\": null}}"
            )
            synth_result = self._call_agent(orchestrator, synthesis_prompt)
            routing = _parse_orchestrator_response(synth_result["answer"])
            orch_result = synth_result

        if not final_answer:
            final_answer = "(Orchestrator did not produce a final answer)"
            messages.append(SwarmMessage(
                from_agent=orchestrator_id,
                to_agent="user",
                role=SwarmMessageRole.FINAL,
                content=final_answer,
                timestamp_ms=int((time.time() - total_start) * 1000),
            ))

        total_time = int((time.time() - total_start) * 1000)

        return SwarmTrace(
            messages=messages,
            final_answer=final_answer,
            total_time_ms=total_time,
        )

    def _execute_collaborative(self, tc: SwarmTestCase) -> SwarmTrace:
        """Peer-to-peer execution with convergence."""
        messages = []
        total_start = time.time()
        agents = self.topology.agents

        if not agents:
            return SwarmTrace(messages=[], final_answer="(No agents configured)")

        max_rounds = tc.max_turns or self.max_rounds

        # Step 1: Broadcast question to all agents
        initial_responses = {}
        for agent in agents:
            messages.append(SwarmMessage(
                from_agent="user",
                to_agent=agent.id,
                role=SwarmMessageRole.USER,
                content=tc.question,
                timestamp_ms=int((time.time() - total_start) * 1000),
            ))

            result = self._call_agent(agent, tc.question)
            initial_responses[agent.id] = result["answer"]

            messages.append(SwarmMessage(
                from_agent=agent.id,
                to_agent="user",
                role=SwarmMessageRole.AGENT,
                content=result["answer"],
                timestamp_ms=int((time.time() - total_start) * 1000),
            ))
            if self.verbose:
                print(f"      {agent.id}: {result['answer'][:60]}")

        # Step 2: Exchange rounds — each agent sees others' responses and refines
        current_responses = dict(initial_responses)
        for round_num in range(1, max_rounds):
            new_responses = {}
            for agent in agents:
                # Build context from other agents' responses
                others = "\n".join(
                    f"- {aid} ({self.topology.get_agent(aid).role if self.topology.get_agent(aid) else aid}): {resp}"
                    for aid, resp in current_responses.items()
                    if aid != agent.id
                )
                refine_prompt = (
                    f"Original question: {tc.question}\n\n"
                    f"Other agents' responses:\n{others}\n\n"
                    f"Your previous response: {current_responses.get(agent.id, '')}\n\n"
                    f"Refine your answer considering the other perspectives. "
                    f"If you agree with the consensus, you may keep your answer similar."
                )

                result = self._call_agent(agent, refine_prompt)
                new_responses[agent.id] = result["answer"]

                # Record as agent message to the group
                messages.append(SwarmMessage(
                    from_agent=agent.id,
                    to_agent="all",
                    role=SwarmMessageRole.AGENT,
                    content=result["answer"],
                    timestamp_ms=int((time.time() - total_start) * 1000),
                    metadata={"round": round_num},
                ))
                if self.verbose:
                    print(f"      {agent.id} (round {round_num}): {result['answer'][:60]}")

            current_responses = new_responses

        # Step 3: Combine final responses
        # Use the first agent to synthesize, or just pick the longest response
        all_final = "\n\n".join(
            f"{aid}: {resp}" for aid, resp in current_responses.items()
        )

        if len(agents) > 1:
            # Ask first agent to synthesize
            synth_agent = agents[0]
            synth_prompt = (
                f"Original question: {tc.question}\n\n"
                f"All agents' final responses:\n{all_final}\n\n"
                f"Synthesize a single, consolidated final answer combining the best points."
            )
            synth_result = self._call_agent(synth_agent, synth_prompt)
            final_answer = synth_result["answer"]
        else:
            final_answer = current_responses.get(agents[0].id, "")

        messages.append(SwarmMessage(
            from_agent=agents[0].id if agents else "system",
            to_agent="user",
            role=SwarmMessageRole.FINAL,
            content=final_answer,
            timestamp_ms=int((time.time() - total_start) * 1000),
        ))

        total_time = int((time.time() - total_start) * 1000)

        return SwarmTrace(
            messages=messages,
            final_answer=final_answer,
            total_time_ms=total_time,
        )
