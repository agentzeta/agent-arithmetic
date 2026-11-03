#!/usr/bin/env python3
"""The agent loop, built and priced.

    python3 -I labs/lab_arithmetic.py

Everything here is next-word prediction with plumbing. The plumbing is where
the money and the risk are, and this lab prices both. Every number is computed
on the spot, not quoted.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import budget
from agentloop import (
    Agent,
    Memory,
    Response,
    ScriptedModel,
    Skill,
    SkillSet,
    Tool,
    ToolCall,
    prefix_cost,
    quadratic_projection,
)


def rule(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


TOOLS = [
    Tool("search", "read", lambda q: f"results for {q}"),
    Tool("read_file", "read", lambda path: f"contents of {path}"),
    Tool("write_file", "write", lambda path, body: f"wrote {path}"),
    Tool("send_email", "comms", lambda to, body: f"sent to {to}"),
    Tool("pay", "money", lambda amount: f"paid {amount}"),
]


# ------------------------------------------------------------------------
rule("0. The loop, running")

print("""Nothing below is a framework. It is a while loop that calls a
function, and the function here is a scripted stub so the output is
deterministic.""")

model = ScriptedModel(
    [
        # Order matters: the first matching rule fires, and "invoice" also
        # occurs inside the search result. Put the later state first or the
        # agent searches forever. (It did, the first time I wrote this.)
        ("results for", Response(text="", thinking="now pay it",
                                 calls=(ToolCall("pay", {"amount": 2400}),))),
        ("invoice", Response(text="", thinking="need to look it up",
                             calls=(ToolCall("search", {"q": "invoice 4471"}),))),
    ],
    default=Response(text="Paid invoice 4471."),
)

res = Agent(model, TOOLS, system="You are a billing assistant.").run(
    "settle invoice 4471"
)
print()
print(res.summary())
print()
print(res.accounting.summary())

print("""
Three things visible in that table and in no dashboard I have seen.

The `sent` column is the whole history, every turn. It only goes up.
The `think` column is charged and never reaches the user.
The run ended because the model stopped asking, which is a hope, not a
guarantee — section 1 shows what happens when it does not.""")


# ------------------------------------------------------------------------
rule("1. The step budget, not the model, is what terminates the loop")

stuck = ScriptedModel([("", Response(calls=(ToolCall("search", {"q": "x"}),)))])
r = Agent(stuck, TOOLS, step_budget=5).run("go")
print(r.summary())

print("""
This matters for eval design. A task that "failed" may have been truncated,
and the budget is a parameter of your harness rather than a property of the
agent. Changing it changes your reported success rate. It belongs in the
report, next to the number it determines.""")


# ------------------------------------------------------------------------
rule("2. Context cost is quadratic in turns")

print("""Each turn re-sends the history, so cumulative tokens sent follow
n(n+1)/2, not n. At 200 tokens per turn:""")
print()
print(f"  {'turns':>6}{'if each turn cost its own':>28}{'actual':>12}{'factor':>9}")
for n in (1, 5, 10, 20, 50, 100):
    p = quadratic_projection(200, n)
    print(f"  {p['turns']:>6}{p['if_each_turn_cost_its_own_content']:>28,}"
          f"{p['actual_cumulative_sent']:>12,}{p['understatement_factor']:>9}")

print("""
A 100-turn session costs 50x what per-turn reasoning predicts. This is the
single biggest gap between what teams budget for agents and what they are
billed, and it needs no model knowledge to compute.""")


# ------------------------------------------------------------------------
rule("3. Skills, memory and persona are string concatenation — priced")

skills = SkillSet([
    Skill("refunds", "REFUND POLICY. " + "detail " * 60, ("refund", "money back")),
    Skill("shipping", "SHIPPING. " + "detail " * 40, ("shipping", "delivery")),
])
body, loaded, toks = skills.load_for("customer wants a refund")
print(f"query mentions a refund -> loaded {loaded}, {toks} tokens")
print(f"loading every skill unconditionally would cost "
      f"{skills.always_on_cost()} tokens per turn")

mem = Memory()
for line in ("user prefers metric units", "user is in Mumbai",
             "user disputes invoice 4471"):
    mem.remember(line)
print(f"memory: {mem.n_entries} entries, {mem.tokens} tokens, "
      f"stored as lines of text and pasted back")

print("""
And what that costs over a session, decomposed by what put it there:""")
print()
print(prefix_cost(
    n_turns=20,
    persona_tokens=150,
    skill_tokens=400,
    skill_loaded_at_turn=2,
    memory_tokens_per_turn=20,
    conversation_tokens_per_turn=120,
).summary())

print("""
The persona is linear. The skill is linear from the turn it loads. The
conversation and the memory are both quadratic, because each grows every turn
AND is re-sent every turn. A single "context size" figure cannot tell you
which of the three to cut, and the answer is almost always the quadratic
ones.""")


# ------------------------------------------------------------------------
rule("4. Cost per successful task is c/p, whatever the retry budget")

print("""With per-attempt cost c and success p, up to r attempts:

    P(success)     = 1 - (1-p)^r
    E[attempts]    = (1 - (1-p)^r) / p
    cost / success = c / p            <- r cancels exactly

So retries buy reliability and not efficiency. Checked across budgets at
p=0.7, c=1:""")
print()
print(f"  {'r':>4}{'P(success)':>14}{'E[attempts]':>14}{'cost/task':>12}{'cost/SUCCESS':>14}")
for r in (1, 2, 3, 5, 10, 50):
    rep = budget.assess(1.0, 0.7, r)
    print(f"  {r:>4}{rep.success:>14.6f}{rep.attempts:>14.4f}"
          f"{rep.cost:>12.4f}{rep.per_success:>14.6f}")

print("""
The last column does not move. If cost per successful task is too high, the
only lever is p — no retry policy touches it. The corollary is more useful
than it sounds: a generous retry budget is not wasteful. It is cost-neutral
per success and strictly better for reliability. What it is, is concealing.""")


# ------------------------------------------------------------------------
rule("5. Retries hide degradation; cost notices before success rate does")

print(budget.masking(0.90, 0.70, 3).summary())

print("""
A 20-point collapse in per-attempt reliability, and the success-rate
dashboard moves 2.6 points — inside the noise at most volumes. Cost moves
25%.

  Cost is the LEADING indicator of reliability degradation.
  Success rate is the LAGGING one.

That inverts the usual dashboard. And a larger retry budget conceals more:""")
print()
print(f"  {'budget':>8}{'success drop (pp)':>20}{'cost rise':>12}{'signal ratio':>14}")
for r in (1, 2, 3, 5, 10):
    m = budget.masking(0.90, 0.70, r)
    ratio = "n/a" if m.masking_ratio is None else f"{m.masking_ratio:.1f}x"
    print(f"  {r:>8}{m.success_drop_pp:>20.2f}{m.cost_rise:>11.1%}{ratio:>14}")

print("""
And retries do nothing for inputs that are simply hard. 80 easy items at
p=0.99, 20 hard ones at p=0.01, five attempts each:""")
print()
actual, naive = budget.correlated_effective_success([0.99] * 80 + [0.01] * 20, 5)
print(f"  independent formula on the mean p : {naive:.4f}")
print(f"  actual, per-item                  : {actual:.4f}")
print(f"  overstatement                     : {naive - actual:.4f}")
print("""
The hard 20% fail again on every retry. Averaging p before applying
1-(1-p)^r assumes away exactly the structure that decides the answer — the
same mistake as treating a clustered sample as independent.""")



# --------------------------------------------------------------------------
rule("What these six share")

print("""Every result here comes from arithmetic on the plumbing, not from
knowing anything about the model. Quadratic context growth, c/p, the masking
ratio -- none of it needs a GPU, and all of it changes what you build.

That is the argument for treating the plumbing as the subject. The model is
the part you cannot control; the plumbing is the part you can, and it is where
the cost, the risk and the false confidence actually live.""")
print()
