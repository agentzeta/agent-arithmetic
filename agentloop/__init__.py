"""agentloop — the whole stack, built, so the claim that nothing is magic can be checked.

    core.py     model, chat, tools, the loop, and token accounting
    context.py  skills, memory, persona — string concatenation, priced

The deterministic ScriptedModel keeps every lab offline and every assertion
reproducible. Swapping in a real model changes one function.
"""

from __future__ import annotations

from .context import ContextBudget, Memory, Skill, SkillSet, prefix_cost
from .core import (
    Accounting,
    Agent,
    Message,
    Response,
    RunResult,
    ScriptedModel,
    Tool,
    ToolCall,
    ToolResult,
    Turn,
    count_tokens,
    quadratic_projection,
)

__all__ = [
    "Accounting", "Agent", "ContextBudget", "Memory", "Message", "Response",
    "RunResult", "ScriptedModel", "Skill", "SkillSet", "Tool", "ToolCall",
    "ToolResult", "Turn", "count_tokens", "prefix_cost", "quadratic_projection",
]
__version__ = "0.1.0"
