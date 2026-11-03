"""Skills, memory and persona — all of which are string concatenation.

Each is genuinely just text going into the prompt:

    persona   a markdown file at the top
    skills    markdown files loaded when a trigger matches
    memory    text written to disk and pasted into the next prompt

Demonstrating that is the easy half. The half worth measuring is what it
costs, because all three land in the *prefix* of the context, and the prefix
is re-sent on every turn of the loop.

**A 400-token skill loaded on turn 2 of a 20-turn conversation is not paid
for once. It is paid for nineteen times.** `ContextBudget` computes that, and
it is the reason "just load the skill, it's only a few hundred tokens" is
wrong by an order of magnitude in a long session.

Memory is worse, because it is cumulative by design: every turn appends, and
every append is re-sent for the rest of the session. Memory that grows
linearly in turns puts another quadratic term into the bill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .core import count_tokens

__all__ = ["Skill", "SkillSet", "Memory", "ContextBudget", "prefix_cost"]


@dataclass(frozen=True)
class Skill:
    """A markdown file loaded into the prompt when a trigger matches."""

    name: str
    body: str
    triggers: tuple[str, ...] = ()

    @property
    def tokens(self) -> int:
        return count_tokens(self.body)

    def matches(self, text: str) -> bool:
        low = text.lower()
        return any(t.lower() in low for t in self.triggers)


class SkillSet:
    """Loads matching skills. The whole mechanism.

    `load_for` returns the concatenated bodies of the skills whose triggers
    match, in declaration order, plus a record of what was loaded and what it
    cost — because a skill that loads silently is a cost that appears
    silently.
    """

    def __init__(self, skills: Sequence[Skill]) -> None:
        names = [s.name for s in skills]
        if len(set(names)) != len(names):
            raise ValueError("duplicate skill names")
        self.skills = list(skills)

    def load_for(self, text: str) -> tuple[str, tuple[str, ...], int]:
        hits = [s for s in self.skills if s.matches(text)]
        body = "\n\n".join(s.body for s in hits)
        return body, tuple(s.name for s in hits), sum(s.tokens for s in hits)

    def always_on_cost(self) -> int:
        """What loading every skill unconditionally would cost per turn.

        Worth computing before deciding that trigger matching is not worth
        the complexity.
        """
        return sum(s.tokens for s in self.skills)


class Memory:
    """Text saved to disk and pasted into the next prompt. Nothing else.

    Kept deliberately literal — append-only lines in a file — so that the
    cost structure is visible rather than hidden behind a store.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._lines: list[str] = []
        if self.path is not None and self.path.exists():
            self._lines = [
                ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln
            ]

    def remember(self, line: str) -> None:
        self._lines.append(line)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def render(self) -> str:
        return "\n".join(self._lines)

    @property
    def tokens(self) -> int:
        return count_tokens(self.render())

    @property
    def n_entries(self) -> int:
        return len(self._lines)


# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextBudget:
    n_turns: int
    persona_tokens: int
    skill_tokens: int
    skill_loaded_at_turn: int
    memory_tokens_per_turn: int
    conversation_tokens_per_turn: int
    components: dict[str, int] = field(default_factory=dict)

    @property
    def total_sent(self) -> int:
        return sum(self.components.values())

    def summary(self) -> str:
        total = self.total_sent
        lines = [
            f"cumulative tokens sent over {self.n_turns} turns: {total:,}",
            "",
            f"  {'component':<34}{'tokens':>10}{'share':>8}",
        ]
        for name, val in self.components.items():
            share = f"{val / total:.1%}" if total else "n/a"
            lines.append(f"  {name:<34}{val:>10,}{share:>8}")
        if self.skill_tokens:
            paid = self.n_turns - self.skill_loaded_at_turn + 1
            lines.append("")
            lines.append(
                f"  the {self.skill_tokens:,}-token skill loaded at turn "
                f"{self.skill_loaded_at_turn} is sent {paid} times, "
                f"costing {self.skill_tokens * paid:,} — not {self.skill_tokens:,}."
            )
        return "\n".join(lines)


def prefix_cost(
    n_turns: int,
    persona_tokens: int = 0,
    skill_tokens: int = 0,
    skill_loaded_at_turn: int = 1,
    memory_tokens_per_turn: int = 0,
    conversation_tokens_per_turn: int = 0,
) -> ContextBudget:
    """Cumulative tokens sent, decomposed by what put them there.

    The persona is a fixed cost paid every turn, so it is linear in turns.
    The conversation and the memory both grow each turn and are re-sent each
    turn, so each contributes a quadratic term. A skill is linear from the
    turn it loads.

    The point of the decomposition is that the three behave differently, and
    a single "context size" number cannot tell you which one to cut.
    """
    if n_turns < 1:
        raise ValueError(f"n_turns must be >= 1, got {n_turns}")
    if not 1 <= skill_loaded_at_turn <= n_turns:
        raise ValueError(
            f"skill_loaded_at_turn must be in [1, {n_turns}], "
            f"got {skill_loaded_at_turn}"
        )
    for name, val in (
        ("persona_tokens", persona_tokens),
        ("skill_tokens", skill_tokens),
        ("memory_tokens_per_turn", memory_tokens_per_turn),
        ("conversation_tokens_per_turn", conversation_tokens_per_turn),
    ):
        if val < 0:
            raise ValueError(f"{name} must be non-negative, got {val}")

    n = n_turns
    triangular = n * (n + 1) // 2
    skill_turns = n - skill_loaded_at_turn + 1

    components = {
        "persona (linear)": persona_tokens * n,
        "skill (linear from load)": skill_tokens * skill_turns,
        "conversation (quadratic)": conversation_tokens_per_turn * triangular,
        "memory (quadratic)": memory_tokens_per_turn * triangular,
    }
    return ContextBudget(
        n_turns=n,
        persona_tokens=persona_tokens,
        skill_tokens=skill_tokens,
        skill_loaded_at_turn=skill_loaded_at_turn,
        memory_tokens_per_turn=memory_tokens_per_turn,
        conversation_tokens_per_turn=conversation_tokens_per_turn,
        components={k: v for k, v in components.items() if v},
    )
