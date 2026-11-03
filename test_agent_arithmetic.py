#!/usr/bin/env python3
"""Checks for agentloop, permcheck, selective and budget.

    python3 -I tests/test_agentstack.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import budget
from agentloop import (
    Agent,
    Memory,
    Message,
    Response,
    ScriptedModel,
    Skill,
    SkillSet,
    Tool,
    ToolCall,
    count_tokens,
    prefix_cost,
    quadratic_projection,
)

PASS = 0
FAIL: list[str] = []


def ok(cond: bool, label: str) -> None:
    global PASS
    if cond:
        PASS += 1
    else:
        FAIL.append(label)


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


# ==========================================================================
# agentloop
# ==========================================================================

def _tools():
    return [
        Tool("search", "read", lambda q: f"results for {q}"),
        Tool("write_file", "write", lambda path, body: f"wrote {path}"),
        Tool("pay", "money", lambda amount: f"paid {amount}"),
    ]


def test_agentloop_core() -> None:
    ok(count_tokens("a b c") == 3, "token count")
    ok(count_tokens("") == 0, "empty token count")
    ok(Message("user", "a b").tokens == 2, "message tokens")

    # a tool must declare a category
    ok(raises(lambda: Tool("x", "", lambda: "y"), ValueError), "tool needs a category")
    ok(raises(lambda: Tool("", "read", lambda: "y"), ValueError), "tool needs a name")

    ok(ToolCall("pay", {"amount": 5}).describe() == "pay(amount=5)", "call describe")
    ok(
        ToolCall("f", {"b": 1, "a": 2}).describe() == "f(a=2, b=1)",
        "describe sorts args for stability",
    )

    # a model that asks for nothing ends the loop immediately
    quiet = ScriptedModel([], default=Response(text="all done"))
    res = Agent(quiet, _tools()).run("hello")
    ok(res.stopped_because == "model_finished", "no tools means immediate finish")
    ok(res.steps_used == 1, "one step used")
    ok(res.final_text == "all done", "final text captured")
    ok(not res.hit_budget, "budget not hit")
    ok(res.denied == (), "nothing denied")

    # one tool call, then done
    one = ScriptedModel(
        [("hello", Response(text="", calls=(ToolCall("search", {"q": "x"}),)))],
        default=Response(text="answered"),
    )
    res = Agent(one, _tools()).run("hello")
    ok(res.stopped_because == "model_finished", "finishes after the tool")
    ok(res.steps_used == 2, "two steps: call then answer")
    ok(any("results for x" in m.content for m in res.messages), "tool result in context")
    ok(len(res.accounting.turns) == 2, "two turns accounted")

    # THE termination finding: the budget stops the loop, not the model
    forever = ScriptedModel(
        [("", Response(text="", calls=(ToolCall("search", {"q": "x"}),)))],
    )
    res = Agent(forever, _tools(), step_budget=4).run("go")
    ok(res.stopped_because == "step_budget", "a looping model is stopped by the budget")
    ok(res.hit_budget, "hit_budget set")
    ok(res.steps_used == 4, "used the whole budget")
    ok("budget artefact" in res.summary(), "summary warns the failure is a budget artefact")

    ok(raises(lambda: Agent(quiet, _tools(), step_budget=0), ValueError), "budget >= 1")
    ok(
        raises(lambda: Agent(quiet, [Tool("a", "r", lambda: ""), Tool("a", "r", lambda: "")]), ValueError),
        "duplicate tool names rejected",
    )

    # an unknown tool is reported into the context, not raised
    bad = ScriptedModel(
        [("go", Response(calls=(ToolCall("nope", {}),)))],
        default=Response(text="recovered"),
    )
    res = Agent(bad, _tools()).run("go")
    ok(any("no such tool" in m.content for m in res.messages), "unknown tool reported")
    ok(res.stopped_because == "model_finished", "loop survives an unknown tool")

    # a raising tool is captured, not propagated
    boom = [Tool("boom", "read", lambda: (_ for _ in ()).throw(RuntimeError("nope")))]
    m = ScriptedModel([("go", Response(calls=(ToolCall("boom", {}),)))],
                      default=Response(text="ok"))
    res = Agent(m, boom).run("go")
    ok(any("RuntimeError" in msg.content for msg in res.messages), "tool error captured")

    # the permission hook is one boolean
    deny_money = lambda c: c.tool != "pay"  # noqa: E731
    m = ScriptedModel(
        [("go", Response(calls=(ToolCall("pay", {"amount": 100}),)))],
        default=Response(text="stopped"),
    )
    res = Agent(m, _tools(), permit=deny_money).run("go")
    ok(len(res.denied) == 1, "one call denied")
    ok(res.denied[0].tool == "pay", "the right call denied")
    ok(any("denied" in msg.content for msg in res.messages), "denial fed back to model")
    ok(not any("paid" in msg.content for msg in res.messages), "tool never ran")
    ok("denied 1 call" in res.summary(), "summary reports denials")


def test_agentloop_accounting() -> None:
    # THE cost finding: each turn re-sends the history, so cost is quadratic
    proj = quadratic_projection(200, 50)
    ok(proj["if_each_turn_cost_its_own_content"] == 10_000, "naive linear estimate")
    ok(proj["actual_cumulative_sent"] == 255_000, "actual is the triangular number")
    ok(proj["understatement_factor"] == 25, "understated 25x at 50 turns")

    ok(quadratic_projection(100, 1)["actual_cumulative_sent"] == 100, "one turn is linear")
    ok(quadratic_projection(10, 4)["actual_cumulative_sent"] == 100, "10*4*5/2 = 100")
    ok(quadratic_projection(5, 0)["actual_cumulative_sent"] == 0, "zero turns")
    ok(raises(lambda: quadratic_projection(-1, 5), ValueError), "negative tokens rejected")

    # amplification rises with turn count in a real run
    loop = ScriptedModel(
        [("", Response(text="thinking out loud", calls=(ToolCall("search", {"q": "x"}),),
                       thinking="a b c"))],
    )
    short = Agent(loop, _tools(), step_budget=2).run("go")
    long = Agent(loop, _tools(), step_budget=8).run("go")
    a_s, a_l = short.accounting.amplification, long.accounting.amplification
    ok(a_s is not None and a_l is not None and a_l > a_s, "amplification grows with turns")
    ok(long.accounting.sent_tokens > long.accounting.final_context_tokens,
       "cumulative sent exceeds the final context")
    ok(long.accounting.thinking_tokens > 0, "thinking tokens counted")
    ok("charged, not shown to the user" in long.accounting.summary(),
       "summary flags thinking as charged")

    empty = Agent(ScriptedModel([], default=Response(text="x")), _tools())
    ok(empty.run("q").accounting.amplification is not None, "amplification defined")


def test_agentloop_context() -> None:
    s = Skill("refund", "how to refund: " + "word " * 20, ("refund", "money back"))
    ok(s.tokens == 23, "skill tokens")
    ok(s.matches("I want a REFUND please"), "trigger match is case-insensitive")
    ok(not s.matches("hello"), "no spurious match")

    ss = SkillSet([s, Skill("ship", "shipping " * 10, ("shipping",))])
    body, names, toks = ss.load_for("refund please")
    ok(names == ("refund",), "only the matching skill loads")
    ok(toks == s.tokens, "cost of what loaded")
    ok(ss.load_for("nothing relevant")[1] == (), "no skills load")
    ok(ss.always_on_cost() == s.tokens + 10, "always-on cost")
    ok(raises(lambda: SkillSet([s, s]), ValueError), "duplicate skill names rejected")

    # memory is text on disk, pasted back
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "mem.txt"
        m = Memory(p)
        m.remember("user prefers metric")
        m.remember("user is in Mumbai")
        ok(m.n_entries == 2, "two entries")
        ok("Mumbai" in m.render(), "render includes entries")
        reloaded = Memory(p)
        ok(reloaded.n_entries == 2, "memory survives reload from disk")
        ok(reloaded.tokens == m.tokens, "token count stable across reload")
    ok(Memory(None).n_entries == 0, "in-memory-only works")

    # THE prefix finding: a skill loaded at turn 2 of 20 is paid for 19 times
    b = prefix_cost(
        n_turns=20,
        persona_tokens=150,
        skill_tokens=400,
        skill_loaded_at_turn=2,
        memory_tokens_per_turn=20,
        conversation_tokens_per_turn=120,
    )
    ok(b.components["persona (linear)"] == 150 * 20, "persona is linear")
    ok(b.components["skill (linear from load)"] == 400 * 19, "skill paid 19 times")
    ok(b.components["conversation (quadratic)"] == 120 * 210, "conversation triangular")
    ok(b.components["memory (quadratic)"] == 20 * 210, "memory triangular")
    ok(b.total_sent == 150 * 20 + 400 * 19 + 140 * 210, "total decomposes exactly")
    ok("is sent 19 times" in b.summary(), "summary states the multiplier")
    ok("not 400" in b.summary(), "summary contrasts with the naive figure")

    # the quadratic terms dominate at length
    short_b = prefix_cost(2, persona_tokens=150, conversation_tokens_per_turn=120)
    long_b = prefix_cost(100, persona_tokens=150, conversation_tokens_per_turn=120)
    sh = short_b.components["conversation (quadratic)"] / short_b.total_sent
    lo = long_b.components["conversation (quadratic)"] / long_b.total_sent
    ok(lo > sh, "the conversation's share grows with length")
    ok(lo > 0.95, "at 100 turns the conversation is essentially all of it")

    ok(raises(lambda: prefix_cost(0), ValueError), "n_turns >= 1")
    ok(raises(lambda: prefix_cost(5, skill_loaded_at_turn=9), ValueError),
       "skill load turn must be inside the run")
    ok(raises(lambda: prefix_cost(5, persona_tokens=-1), ValueError), "negative rejected")


# ==========================================================================
# permcheck
# ==========================================================================

def test_budget() -> None:
    ok(close(budget.effective_success(0.8, 1), 0.8), "r=1 is the raw rate")
    ok(close(budget.effective_success(0.5, 2), 0.75), "two tries at 0.5")
    ok(close(budget.effective_success(0.0, 5), 0.0), "p=0 never succeeds")
    ok(close(budget.effective_success(1.0, 3), 1.0), "p=1 always succeeds")
    ok(close(budget.expected_attempts(1.0, 5), 1.0), "p=1 uses one attempt")
    ok(close(budget.expected_attempts(0.0, 5), 5.0), "p=0 burns the whole budget")
    ok(close(budget.expected_attempts(0.5, 2), 1.5), "0.5 over two tries")

    # THE result: cost per success is c/p, independent of r
    for p in (0.3, 0.5, 0.7, 0.9):
        base = budget.cost_per_success(1.0, p, 1)
        for r in (1, 2, 3, 5, 10, 50):
            got = budget.cost_per_success(1.0, p, r)
            ok(got is not None and base is not None and close(got, base, 1e-12),
               f"cost per success is r-invariant at p={p}, r={r}")
        ok(base is not None and close(base, 1.0 / p), f"cost per success is c/p at p={p}")

    ok(budget.cost_per_success(1.0, 0.0, 3) is None, "p=0 gives undefined, not huge")

    rep = budget.assess(0.002, 0.7, 5)
    ok(close(rep.success, 1 - 0.3 ** 5), "effective success")
    ok(rep.per_success is not None and rep.per_success_no_retry is not None
       and close(rep.per_success, rep.per_success_no_retry),
       "report shows the two cost-per-success figures are equal")
    ok("r cancels" in rep.summary(), "summary states why")

    # retries for a target
    ok(budget.retries_for_target(0.5, 0.9) == 4, "4 tries at p=0.5 clears 0.9")
    ok(budget.retries_for_target(0.9, 0.99) == 2, "2 tries at p=0.9 clears 0.99")
    ok(budget.retries_for_target(1.0, 1.0) == 1, "p=1 reaches certainty")
    ok(budget.retries_for_target(0.9, 1.0) is None, "no finite budget reaches certainty")
    ok(budget.retries_for_target(0.0, 0.5) is None, "p=0 is unreachable")
    ok(raises(lambda: budget.retries_for_target(0.5, 0.0), ValueError), "target 0 rejected")

    # THE monitoring result: retries mask a falling p; cost notices first
    m = budget.masking(0.90, 0.70, 3)
    ok(abs(m.success_drop_pp) < 3.0, "success rate moves under 3 points")
    ok(m.cost_rise > 0.20, "cost rises over 20%")
    ok(m.masking_ratio is not None and m.masking_ratio > 5, "cost signal is much larger")
    ok("would likely miss this" in m.note, "the monitoring conclusion stated")
    ok(close(m.success_before, 0.999), "success before")
    ok(close(m.success_after, 1 - 0.3 ** 3), "success after")

    # a bigger budget masks more
    m1 = budget.masking(0.90, 0.70, 1)
    m5 = budget.masking(0.90, 0.70, 5)
    ok(abs(m5.success_drop_pp) < abs(m1.success_drop_pp),
       "a larger retry budget conceals more of the drop")
    ok(m1.masking_ratio is not None and m5.masking_ratio is not None
       and m5.masking_ratio > m1.masking_ratio, "masking ratio grows with the budget")

    ok("no degradation to mask" in budget.masking(0.7, 0.9, 3).note, "improvement noted")

    # THE correlation result: retries do nothing for items that are just hard
    bimodal = [0.99] * 80 + [0.01] * 20
    actual, naive = budget.correlated_effective_success(bimodal, 5)
    ok(naive > actual, "the independent formula overstates the retry gain")
    ok(naive - actual > 0.1, "and overstates it by a lot on bimodal difficulty")
    ok(actual < 0.85, "the hard 20% stay hard however many retries")

    uniform = [0.8] * 100
    a_u, n_u = budget.correlated_effective_success(uniform, 3)
    ok(close(a_u, n_u), "no gap when every item is equally hard")

    ok(raises(lambda: budget.correlated_effective_success([], 3), ValueError), "empty rejected")
    ok(raises(lambda: budget.correlated_effective_success([1.5], 3), ValueError),
       "out-of-range difficulty rejected")
    ok(raises(lambda: budget.effective_success(0.5, 0), ValueError), "r=0 rejected")
    ok(raises(lambda: budget.effective_success(1.5, 1), ValueError), "p>1 rejected")
    ok(raises(lambda: budget.expected_cost(-1.0, 0.5, 2), ValueError), "negative cost rejected")


# ==========================================================================

def main() -> int:
    for fn in (
        test_agentloop_core,
        test_agentloop_accounting,
        test_agentloop_context,
        test_budget,
    ):
        fn()
    if FAIL:
        print(f"{PASS} passed, {len(FAIL)} FAILED:")
        for f in FAIL:
            print(f"  FAIL: {f}")
        return 1
    print(f"all green ({PASS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
