# agent-arithmetic

**Your agent's cost and reliability are arithmetic. Most of it is being computed wrong.**

Pure Python standard library. No dependencies, no network, no API keys, no model.
Every number below is computed by the code in this repo.

```bash
git clone https://github.com/<you>/agent-arithmetic && cd agent-arithmetic
python3 -I tests/test_agent_arithmetic.py    # 130 checks
python3 -I labs/lab_arithmetic.py            # the six results, computed live
```

---

## Three results that contradict what most teams believe

### 1. Cost per successful task is `c/p`, whatever your retry budget

With per-attempt cost `c`, per-attempt success `p`, and up to `r` attempts:

```
P(success)     = 1 - (1-p)^r
E[attempts]    = (1 - (1-p)^r) / p
cost / success = c / p                ← r cancels, exactly
```

Checked across budgets at `p = 0.7`:

```
   r    P(success)   E[attempts]   cost/task   cost/SUCCESS
   1      0.700000        1.0000      1.0000       1.428571
   2      0.910000        1.3000      1.3000       1.428571
   3      0.973000        1.3900      1.3900       1.428571
   5      0.997570        1.4251      1.4251       1.428571
  50      1.000000        1.4286      1.4286       1.428571
```

The last column does not move. **Retries buy reliability, not efficiency.** If
cost per successful task is too high, the only lever is `p` — no retry policy
touches it. The corollary surprises people: a generous retry budget is *not*
wasteful. It is cost-neutral per success and strictly better for reliability.

What it is, is concealing.

### 2. Retries hide degradation. Cost notices; success rate doesn't.

```
per-attempt p fell 0.900 -> 0.700 at r=3
  observed success  0.9990 -> 0.9730   (-2.60 pp)
  expected cost     1.110 -> 1.390   (+25.2%)
  cost signal is 9.7x the size of the success signal
```

A **20-point collapse** in per-attempt reliability moves your success dashboard
2.6 points — inside the noise at most volumes — while cost moves 25%. A larger
retry budget conceals more, so the system gets *less* observable exactly as you
make it more reliable:

```
  budget   success drop (pp)   cost rise  signal ratio
       1              -20.00       0.0%          0.0x
       2               -8.00      18.2%          2.3x
       3               -2.60      25.2%          9.7x
       5               -0.24      28.3%        116.8x
      10               -0.00      28.6%      48385.5x
```

> **Cost is the leading indicator of reliability degradation.
> Success rate is the lagging one.**

If you monitor one number, monitor cost per successful task.

### 3. Context cost is quadratic in turns, not linear

Every turn re-sends the whole history, so cumulative tokens follow `n(n+1)/2`:

```
 turns   if each turn cost its own      actual   factor
     1                         200         200        1
    10                       2,000      11,000        5
    20                       4,000      42,000       10
    50                      10,000     255,000       25
   100                      20,000   1,010,000       50
```

A 100-turn session costs **fifty times** what per-turn reasoning predicts. This
also explains a commonly misattributed surprise: a change that lengthens average
conversations by 40% raises spend by about 96%, and the post-mortem goes looking
for a pricing change.

And the decomposition, which is what tells you *what to cut*:

```
cumulative tokens sent over 20 turns: 40,000

  component                             tokens   share
  persona (linear)                       3,000    7.5%
  skill (linear from load)               7,600   19.0%
  conversation (quadratic)              25,200   63.0%
  memory (quadratic)                     4,200   10.5%

  the 400-token skill loaded at turn 2 is sent 20 times,
  costing 7,600 — not 400.
```

Persona is linear. A skill is linear from the turn it loads. Conversation and
memory are both quadratic, because each grows every turn *and* is re-sent every
turn. A single "context size" number cannot tell you which to cut. At 100 turns
the conversation term is over 95% of the bill, at which point prefix
optimisation stops mattering at all.

---

## Also in here: the whole stack, built

The claim that nothing about agents is magic is correct, and the way to settle
it is to build it and look. `agentloop` is the entire thing in ~450 lines with a
deterministic stub model, so every lab runs offline and every assertion is
reproducible. Swapping in a real model changes one function.

```python
Model = Callable[[Sequence[Message]], Response]
Tool  = (name, category, Callable[..., str])
```

```python
while steps < self.step_budget:
    response = self.model(messages)                 # think
    if not response.wants_tools: break              # done
    for call in response.calls:                     # act
        if self.permit and not self.permit(call):
            messages.append(Message("tool", f"denied: {call.describe()}"))
            continue
        messages.append(Message("tool", self.tools[call.tool].run(**call.args)))
```

Three things that loop makes visible:

**The step budget is what actually terminates it.** "Until the model stops
asking" is a hope. A looping model is stopped by the counter, and the run is
recorded as a failure:

```
stopped: step_budget after 5/5 steps
  NOTE: the budget terminated this run, not the model. An observed failure
  here is a budget artefact and must be reported as one.
```

This is an eval-design problem, not a detail. **The budget silently sets the
success rate you then measure**, so it belongs in the report next to the number
it determines.

**`permit` is one boolean**, decided from a description of an action that has
not happened yet. Everything hard about agent permissions follows from the
thinness of that interface.

**Skills, memory and persona are string concatenation** — implemented literally,
so the cost structure is visible rather than hidden behind a store.

---

## Correlated failures defeat retries

`1-(1-p)^r` assumes attempts are independent. Real workloads have per-item
difficulty: some inputs are simply hard, and a retry on a hard input fails
again. With 80 easy items at `p=0.99` and 20 hard at `p=0.01`, five attempts:

```
  independent formula on the mean p : 0.9996
  actual, per-item                  : 0.8098
  overstatement                     : 0.1898
```

Nineteen points of overstatement. Averaging `p` before applying `1-(1-p)^r`
assumes away precisely the structure that decides the answer — the same error
shape as treating a clustered evaluation sample as independent.

---

## The design rule

Every function here follows one rule, and it is the point of the project:

> **A function that cannot honestly answer returns `None` and reports the
> shortfall. It never relaxes a constraint to produce a number.**

- `cost_per_success(c, p=0)` returns `None`. At `p=0` there are no successes,
  so cost per success is undefined, not large. A big number would invite
  somebody to budget for it.
- `retries_for_target(p=0.9, target=1.0)` returns `None`. No finite retry
  budget reaches certainty, and rounding that to a large integer would be a lie
  about what the system can promise.
- `n_for_pass_hat_k(..., assumed_p=below_required)` returns `None`. If the
  system does not meet the target, more data is not the remedy.

A tool that always returns a number is a dashboard. A dashboard is not a
control.

---

## Errors this code caught in its own author

Published rather than hidden, because a measurement tool whose author cannot
name a time it caught them probably is not being used.

1. **The demo agent searched forever.** The scripted model fires the first
   matching rule, and the needle `"invoice"` also appears inside the search
   result `"results for invoice 4471"`. So it re-searched every turn and the
   step budget terminated a run meant to finish in three. The lesson was in the
   section above the bug.
2. **A test passed for the wrong reason.** `recovery_rate or -1` — `0.0` is
   falsy in Python, so a correct recovery rate of zero was silently being
   compared against `-1`. That shape is the single most common bug in eval code.
3. **A hand-written table in the write-up was wrong.** Three of five rows of the
   masking table were transcribed from memory rather than from the lab. Caught
   by re-running every quoted figure before publishing. The process fix:
   **quote tool output, never transcribe it.**
4. **Mutation-tested the headline result.** `cost_per_success` was perturbed to
   depend weakly on `r` to confirm the suite was not passing vacuously. 25
   checks failed, naming `p` and `r` for each. Worth doing to any result you
   intend to lead with.

5. **The tests gave a false green under pytest.** The suites record failures
   and exit non-zero when run as scripts, but never raised — so `pytest`, the
   runner most people reach for, reported every test as passing even with the
   headline result deliberately broken (`4 passed`, exit 0, against `4 FAILED`,
   exit 1, as a script). A check that cannot fail is the exact failure this
   repo warns about. `ok()` now raises under pytest; both runners agree.

---

## Layout

```
agentloop/
  core.py        model, tools, the loop, token accounting, the permit hook
  context.py     skills, memory, persona — concatenation, priced by component
budget/
  retries.py     c/p, expected attempts, masking, correlated difficulty
labs/lab_arithmetic.py        the six results, computed live
tests/test_agent_arithmetic.py    130 checks
```

## Scope

This measures the plumbing. It does **not** evaluate model quality, and
conflating the two weakens both. For judge alignment, thresholds, calibration,
leakage and sampling, see the companion repo.

Two caveats worth stating: the token counter is deliberately crude
(whitespace-delimited words), because the quadratic result is a property of the
loop rather than of the tokeniser. And **prompt caching changes the constant in
result 3, though not its shape** — treat the quadratic figure as the uncached
bound and get your provider's caching semantics before applying it to a bill.

## Licence

MIT.
