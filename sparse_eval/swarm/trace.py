"""
trace.py -- Trace loading and parsing for swarm evaluation.

Loads interaction traces from JSON/YAML files or inline YAML definitions.
Traces represent the complete message exchange in a multi-agent interaction.
"""

import json
from pathlib import Path
from typing import Optional

from sparse_eval.swarm.models import SwarmTrace, SwarmMessage, SwarmMessageRole


# Map string role names to enum values
_ROLE_MAP = {
    "user": SwarmMessageRole.USER,
    "orchestrator": SwarmMessageRole.ORCHESTRATOR,
    "agent": SwarmMessageRole.AGENT,
    "handoff": SwarmMessageRole.HANDOFF,
    "final": SwarmMessageRole.FINAL,
}


def _parse_message(msg: dict) -> SwarmMessage:
    """Parse a single message dict into a SwarmMessage."""
    role_str = msg.get("role", "")
    if not role_str:
        # Auto-detect role from sender
        from_agent = msg.get("from", msg.get("from_agent", ""))
        if from_agent == "user":
            role_str = "user"
        else:
            role_str = "agent"
    role = _ROLE_MAP.get(role_str, SwarmMessageRole.AGENT)

    return SwarmMessage(
        from_agent=msg.get("from", msg.get("from_agent", "")),
        to_agent=msg.get("to", msg.get("to_agent", "")),
        role=role,
        content=msg.get("content", ""),
        timestamp_ms=msg.get("timestamp_ms", 0),
        metadata=msg.get("metadata", {}),
    )


def parse_trace(data: list[dict]) -> SwarmTrace:
    """
    Parse a list of message dicts into a SwarmTrace.

    The final answer is extracted from the last message with role 'final',
    or the last message in the trace if no 'final' message exists.

    Args:
        data: List of message dicts with from, to, role, content fields.

    Returns:
        SwarmTrace with parsed messages and extracted final answer.
    """
    messages = [_parse_message(m) for m in data]

    # Extract final answer
    final_answer = ""
    for m in reversed(messages):
        if m.role == SwarmMessageRole.FINAL:
            final_answer = m.content
            break
    if not final_answer and messages:
        # Fallback: last message to "user"
        for m in reversed(messages):
            if m.to_agent == "user":
                final_answer = m.content
                break
        if not final_answer:
            final_answer = messages[-1].content

    # Total time from timestamps
    total_time_ms = 0
    timestamps = [m.timestamp_ms for m in messages if m.timestamp_ms > 0]
    if len(timestamps) >= 2:
        total_time_ms = max(timestamps) - min(timestamps)

    return SwarmTrace(
        messages=messages,
        final_answer=final_answer,
        total_time_ms=total_time_ms,
    )


def load_trace_file(path: str | Path) -> SwarmTrace:
    """
    Load a trace from a JSON or YAML file.

    JSON format: list of message objects
    YAML format: list of message objects

    Args:
        path: Path to the trace file (.json or .yaml/.yml)

    Returns:
        SwarmTrace

    Raises:
        FileNotFoundError: If the trace file doesn't exist
        ValueError: If the trace format is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    content = path.read_text(encoding="utf-8")

    if path.suffix == ".json":
        data = json.loads(content)
    elif path.suffix in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(content)
    else:
        raise ValueError(f"Unsupported trace file format: {path.suffix} (use .json or .yaml)")

    # Handle both list-of-messages and wrapped format
    if isinstance(data, dict):
        if "messages" in data:
            messages_data = data["messages"]
            trace = parse_trace(messages_data)
            # Override final_answer if explicitly provided
            if "final_answer" in data:
                trace.final_answer = data["final_answer"]
            if "total_time_ms" in data:
                trace.total_time_ms = data["total_time_ms"]
            return trace
        raise ValueError(f"Trace file must contain a 'messages' key or be a list of messages")
    elif isinstance(data, list):
        return parse_trace(data)
    else:
        raise ValueError(f"Invalid trace format: expected list or dict, got {type(data).__name__}")
