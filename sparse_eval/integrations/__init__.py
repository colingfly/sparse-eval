"""
integrations -- One-liner eval helpers for popular frameworks.

Wrap your existing agent in a single function call:

    from sparse_eval.integrations import evaluate_langchain
    results = evaluate_langchain(chain, suite="qa.yaml")

    from sparse_eval.integrations import evaluate_openai
    results = evaluate_openai("http://localhost:8000/v1/chat/completions", suite="qa.yaml")

    from sparse_eval.integrations import evaluate_phlote
    results = evaluate_phlote(music_agent, suite="music_eval.yaml")

Each integration adapts the framework's agent interface into sparse-eval's
callable format, runs the eval, and optionally pushes results to Sparse Intelligence Cloud.
"""

from sparse_eval.integrations.langchain import evaluate_langchain
from sparse_eval.integrations.llamaindex import evaluate_llamaindex
from sparse_eval.integrations.openai_compat import evaluate_openai
from sparse_eval.integrations.fastapi_agent import evaluate_fastapi
from sparse_eval.integrations.phlote import evaluate_phlote

__all__ = [
    "evaluate_langchain",
    "evaluate_llamaindex",
    "evaluate_openai",
    "evaluate_fastapi",
    "evaluate_phlote",
]
