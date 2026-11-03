"""A working agent loop in one file, with a deterministic model.

The claim that nothing here is magic is correct, and the way to settle it is
to build the whole thing and look at it. This module is the loop:

    model       a function from messages to a response
    chat        the model called again with the history appended
    tools       Python functions the model asks you to run
    agent       that loop, until the model stops asking
    skills      markdown loaded into the prompt when needed
    memory      text saved to disk and pasted into the next prompt

The model here is a scripted stub, so every lab runs offline and every
assertion is deterministic. Swapping in a real model changes one function.

What the loop makes visible, and what people building on top of it usually
discover in a billing statement instead:

**Context cost grows quadratically in turn count.** Each turn re-sends the
whole history. A 20-turn conversation does not cost 20 units, it costs about
210 — the sum 1+2+...+20. `Accounting.sent_tokens` computes it exactly, and
`quadratic_projection` shows where it goes.

**A tool result never leaves the context.** One oversized tool result is paid
for on that turn and on every turn after it.

**The loop's termination condition is not guaranteed.** "Until the model stops
asking for tools" is a hope. The step budget is what actually terminates it,
and the budget silently sets your observed success rate, so it belongs in the
report rather than in a constant at the top of a file.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Sequence

__all__ = [
    "Message",
    "ToolCall",
    "ToolResult",
    "Tool",
    "Response",
    "Model",
    "ScriptedModel",
    "Turn",
    "Accounting",
    "RunResult",
    "Agent",
    "count_tokens",
    "quadratic_projection",
]

Role = Literal["system", "user", "assistant", "tool"]


# --------------------------------------------------------------------------
# the pieces
# --------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """A deliberately crude token count: whitespace-delimited words.

    Crude on purpose. The quadratic growth in §Accounting is a property of
    the loop, not of the tokeniser, and a real tokeniser would obscure that
    with a constant factor while teaching nothing extra.
    """
    return len(text.split())


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    @property
    def tokens(self) -> int:
        return count_tokens(self.content)


@dataclass(frozen=True)
class ToolCall:
    """What the model asks you to run. It cannot run anything itself."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in sorted(self.args.items()))
        return f"{self.tool}({inner})"


@dataclass(frozen=True)
class ToolResult:
    call: ToolCall
    content: str
    ok: bool = True

    @property
    def tokens(self) -> int:
        return count_tokens(self.content)


@dataclass(frozen=True)
class Tool:
    """A Python function the model may ask for by name.

    `category` exists so that a permission policy has something to match on.
    It is required: a tool whose actions cannot be categorised cannot be
    governed, and discovering that at policy-writing time is better than
    discovering it in an incident.
    """

    name: str
    category: str
    run: Callable[..., str]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a tool needs a name")
        if not self.category:
            raise ValueError(
                f"tool {self.name!r} must declare a category; an uncategorised "
                f"tool cannot be covered by a policy"
            )


@dataclass(frozen=True)
class Response:
    """What the model returns: some text, and zero or more tool requests."""

    text: str = ""
    calls: tuple[ToolCall, ...] = ()
    thinking: str = ""
    """Charged, not returned to the user, and usually absent from cost reports."""

    @property
    def wants_tools(self) -> bool:
        return bool(self.calls)

    @property
    def tokens(self) -> int:
        return count_tokens(self.text) + count_tokens(self.thinking)


Model = Callable[[Sequence[Message]], Response]


class ScriptedModel:
    """A deterministic stand-in for a model.

    Rules are matched against the most recent message's content. The first
    matching rule fires. An unmatched state returns `default`, which by
    default ends the loop.

    `repeat_guard` reproduces the failure mode that makes step budgets
    necessary: when True, a rule whose tool result does not satisfy it will
    be requested again, forever, exactly as a real model sometimes does.
    """

    def __init__(
        self,
        rules: Sequence[tuple[str, Response]],
        default: Response | None = None,
    ) -> None:
        self.rules = list(rules)
        self.default = default if default is not None else Response(text="done")
        self.calls_made = 0

    def __call__(self, messages: Sequence[Message]) -> Response:
        self.calls_made += 1
        if not messages:
            return self.default
        recent = messages[-1].content
        for needle, response in self.rules:
            if needle in recent:
                return response
        return self.default


# --------------------------------------------------------------------------
# accounting
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Turn:
    index: int
    sent_tokens: int
    """Tokens sent TO the model this turn: the whole history, every time."""
    received_tokens: int
    thinking_tokens: int
    calls: tuple[ToolCall, ...]
    result_tokens: int


@dataclass(frozen=True)
class Accounting:
    turns: tuple[Turn, ...]

    @property
    def sent_tokens(self) -> int:
        """Total tokens sent across the run. This is the quadratic term."""
        return sum(t.sent_tokens for t in self.turns)

    @property
    def received_tokens(self) -> int:
        return sum(t.received_tokens for t in self.turns)

    @property
    def thinking_tokens(self) -> int:
        return sum(t.thinking_tokens for t in self.turns)

    @property
    def final_context_tokens(self) -> int:
        return self.turns[-1].sent_tokens if self.turns else 0

    @property
    def naive_estimate(self) -> int:
        """What you would predict if you thought each turn cost its own content.

        The ratio of `sent_tokens` to this is the factor by which a per-turn
        mental model understates a conversation.
        """
        if not self.turns:
            return 0
        first = self.turns[0].sent_tokens
        return first + sum(
            t.received_tokens + t.result_tokens for t in self.turns[:-1]
        ) + sum(0 for _ in self.turns)

    @property
    def amplification(self) -> float | None:
        """sent_tokens / final_context_tokens.

        A one-turn conversation is 1.0. It rises roughly linearly in turn
        count, which is the quadratic growth seen from a different angle.
        """
        fc = self.final_context_tokens
        return None if fc == 0 else self.sent_tokens / fc

    def summary(self) -> str:
        lines = [
            f"{len(self.turns)} model calls",
            f"  sent to model (cumulative)  {self.sent_tokens:>8,}",
            f"  received from model         {self.received_tokens:>8,}",
            f"  of which thinking           {self.thinking_tokens:>8,}"
            f"   (charged, not shown to the user)",
            f"  final context size          {self.final_context_tokens:>8,}",
        ]
        amp = self.amplification
        if amp is not None:
            lines.append(
                f"  amplification               {amp:>8.2f}x"
                f"   (sent / final context)"
            )
        lines.append("")
        lines.append("  turn   sent  recv  think  tools  result")
        for t in self.turns:
            lines.append(
                f"  {t.index:>4} {t.sent_tokens:>6} {t.received_tokens:>5} "
                f"{t.thinking_tokens:>6} {len(t.calls):>6} {t.result_tokens:>7}"
            )
        return "\n".join(lines)


def quadratic_projection(per_turn_tokens: int, n_turns: int) -> dict[str, int]:
    """Cumulative tokens sent for a conversation that grows by a fixed amount.

    Each turn re-sends everything, so the total is per_turn * n(n+1)/2 rather
    than per_turn * n. At 200 tokens a turn, 50 turns sends 255,000 tokens,
    not 10,000.
    """
    if per_turn_tokens < 0 or n_turns < 0:
        raise ValueError("arguments must be non-negative")
    linear = per_turn_tokens * n_turns
    quadratic = per_turn_tokens * n_turns * (n_turns + 1) // 2
    return {
        "turns": n_turns,
        "if_each_turn_cost_its_own_content": linear,
        "actual_cumulative_sent": quadratic,
        "understatement_factor": (quadratic // linear) if linear else 0,
    }


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RunResult:
    messages: tuple[Message, ...]
    accounting: Accounting
    stopped_because: Literal["model_finished", "step_budget", "tool_error"]
    steps_used: int
    step_budget: int
    final_text: str
    denied: tuple[ToolCall, ...] = ()

    @property
    def hit_budget(self) -> bool:
        return self.stopped_because == "step_budget"

    def summary(self) -> str:
        lines = [
            f"stopped: {self.stopped_because} "
            f"after {self.steps_used}/{self.step_budget} steps",
            f"final: {self.final_text!r}",
        ]
        if self.denied:
            lines.append(
                f"denied {len(self.denied)} call(s): "
                + ", ".join(c.describe() for c in self.denied)
            )
        if self.hit_budget:
            lines.append(
                "  NOTE: the budget terminated this run, not the model. "
                "An observed failure here is a budget artefact and must be "
                "reported as one."
            )
        return "\n".join(lines)


class Agent:
    """The loop. Twenty lines of control flow and nothing else.

    `permit` is an optional gate called with each requested ToolCall. It
    returns True to allow. This is the entire hook a permission system gets:
    one boolean, decided from a description of an action that has not
    happened yet. Everything difficult about agent permissions follows from
    the thinness of that interface, which is why it is worth seeing how thin
    it is.
    """

    def __init__(
        self,
        model: Model,
        tools: Sequence[Tool],
        system: str = "",
        step_budget: int = 8,
        permit: Callable[[ToolCall], bool] | None = None,
    ) -> None:
        if step_budget < 1:
            raise ValueError(f"step_budget must be >= 1, got {step_budget}")
        self.model = model
        self.tools = {t.name: t for t in tools}
        if len(self.tools) != len(tools):
            raise ValueError("duplicate tool names")
        self.system = system
        self.step_budget = step_budget
        self.permit = permit

    def run(self, user_text: str) -> RunResult:
        messages: list[Message] = []
        if self.system:
            messages.append(Message("system", self.system))
        messages.append(Message("user", user_text))

        turns: list[Turn] = []
        denied: list[ToolCall] = []
        stopped: Literal["model_finished", "step_budget", "tool_error"] = (
            "step_budget"
        )
        final_text = ""
        steps = 0

        while steps < self.step_budget:
            steps += 1
            sent = sum(m.tokens for m in messages)
            response = self.model(messages)

            result_tokens = 0
            if response.text or not response.wants_tools:
                messages.append(Message("assistant", response.text))

            if not response.wants_tools:
                final_text = response.text
                stopped = "model_finished"
                turns.append(
                    Turn(steps, sent, response.tokens,
                         count_tokens(response.thinking), (), 0)
                )
                break

            for call in response.calls:
                if self.permit is not None and not self.permit(call):
                    denied.append(call)
                    msg = f"denied: {call.describe()}"
                    messages.append(Message("tool", msg))
                    result_tokens += count_tokens(msg)
                    continue
                tool = self.tools.get(call.tool)
                if tool is None:
                    msg = f"error: no such tool {call.tool!r}"
                    messages.append(Message("tool", msg))
                    result_tokens += count_tokens(msg)
                    continue
                try:
                    out = tool.run(**call.args)
                except Exception as exc:  # noqa: BLE001 - surfaced, not hidden
                    out = f"error: {type(exc).__name__}: {exc}"
                messages.append(Message("tool", out))
                result_tokens += count_tokens(out)

            turns.append(
                Turn(steps, sent, response.tokens,
                     count_tokens(response.thinking), response.calls,
                     result_tokens)
            )

        return RunResult(
            messages=tuple(messages),
            accounting=Accounting(tuple(turns)),
            stopped_because=stopped,
            steps_used=steps,
            step_budget=self.step_budget,
            final_text=final_text,
            denied=tuple(denied),
        )
