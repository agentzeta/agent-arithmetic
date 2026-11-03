"""Retries, cost, and the degradation retries hide.

Retrying is the cheapest reliability mechanism available and it has two
properties that almost nobody computes.

**1. Cost per successful task is `c/p`, whatever the retry budget.**

With per-attempt cost `c`, per-attempt success `p`, and up to `r` attempts:

    P(success)     = 1 - (1-p)^r
    E[attempts]    = (1 - (1-p)^r) / p
    E[cost]        = c * (1 - (1-p)^r) / p
    cost / success = E[cost] / P(success) = c / p          <- r cancels

So retries buy reliability and do not buy efficiency. The cost of getting one
task done is fixed by the per-attempt success rate, and no retry policy moves
it. If cost per successful task is too high, the only lever is `p`. This
falls out in two lines and contradicts the common intuition that a generous
retry budget is wasteful — it is not wasteful, it is cost-neutral per success
and strictly better for reliability. What it is, is *concealing*:

**2. Retries hide a falling `p`, and cost notices first.**

At r=3, a drop in `p` from 0.90 to 0.70 moves observed success from 0.999 to
0.973 — under three points, inside the noise of most monitoring — while
expected cost per task rises about 25%. Monitor success rate alone and you
see almost nothing. Monitor cost and the degradation is unmissable.

> **Cost is the leading indicator of reliability degradation. Success rate is
> the lagging one.** This inverts the usual dashboard, where success rate is
> the headline and cost is a footnote.

**3. Correlated failures defeat retries.** The `1-(1-p)^r` formula assumes
attempts are independent. They are not: some inputs are simply hard, and a
retry on a hard input fails again. `correlated_effective_success` models this
with a per-item difficulty mixture, and the retry gain collapses — the same
design-effect structure that governs clustered evaluation samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "effective_success",
    "expected_attempts",
    "expected_cost",
    "cost_per_success",
    "RetryReport",
    "assess",
    "MaskingReport",
    "masking",
    "correlated_effective_success",
    "retries_for_target",
]


def _check(p: float, r: int) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if r < 1:
        raise ValueError(f"r must be >= 1, got {r}")


# --------------------------------------------------------------------------
# the arithmetic
# --------------------------------------------------------------------------

def effective_success(p: float, r: int) -> float:
    """P(at least one success in up to r independent attempts)."""
    _check(p, r)
    return 1.0 - (1.0 - p) ** r


def expected_attempts(p: float, r: int) -> float:
    """Expected attempts used, stopping on first success, capped at r."""
    _check(p, r)
    if p == 0.0:
        return float(r)
    return (1.0 - (1.0 - p) ** r) / p


def expected_cost(cost_per_attempt: float, p: float, r: int) -> float:
    if cost_per_attempt < 0:
        raise ValueError("cost_per_attempt must be non-negative")
    return cost_per_attempt * expected_attempts(p, r)


def cost_per_success(cost_per_attempt: float, p: float, r: int = 1) -> float | None:
    """`c/p` — independent of r. Returns None when p is 0.

    The None matters: at p=0 there are no successes, so cost per success is
    not large, it is undefined. A function returning a big number here would
    invite someone to budget for it.
    """
    _check(p, r)
    if cost_per_attempt < 0:
        raise ValueError("cost_per_attempt must be non-negative")
    if p == 0.0:
        return None
    return cost_per_attempt / p


def retries_for_target(p: float, target: float) -> int | None:
    """Smallest r with effective_success(p, r) >= target.

    Returns None when p is 0, or when the target is 1.0 and p < 1 — no finite
    retry budget reaches certainty, and rounding that to a large integer
    would be a lie about what the system can promise.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if not 0.0 < target <= 1.0:
        raise ValueError(f"target must be in (0, 1], got {target}")
    if p == 0.0:
        return None
    if target == 1.0:
        return 1 if p == 1.0 else None
    if p == 1.0:
        return 1
    r = 1
    while effective_success(p, r) < target:
        r += 1
        if r > 10_000:
            return None
    return r


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RetryReport:
    p: float
    r: int
    cost_per_attempt: float
    success: float
    attempts: float
    cost: float
    per_success: float | None
    per_success_no_retry: float | None

    def summary(self) -> str:
        lines = [
            f"per-attempt p={self.p:.3f}, budget r={self.r}, "
            f"cost per attempt {self.cost_per_attempt:g}",
            f"  P(success)              {self.success:.4f}",
            f"  E[attempts]             {self.attempts:.3f}",
            f"  E[cost per task]        {self.cost:.4f}",
        ]
        if self.per_success is None:
            lines.append("  cost per SUCCESS        undefined (p=0)")
        else:
            lines.append(f"  cost per SUCCESS        {self.per_success:.4f}")
            lines.append(
                f"  cost per SUCCESS at r=1 {self.per_success_no_retry:.4f}"
                f"   <- identical; r cancels"
            )
        return "\n".join(lines)


def assess(cost_per_attempt: float, p: float, r: int) -> RetryReport:
    """Full retry accounting, showing that cost per success does not move."""
    _check(p, r)
    return RetryReport(
        p=p,
        r=r,
        cost_per_attempt=cost_per_attempt,
        success=effective_success(p, r),
        attempts=expected_attempts(p, r),
        cost=expected_cost(cost_per_attempt, p, r),
        per_success=cost_per_success(cost_per_attempt, p, r),
        per_success_no_retry=cost_per_success(cost_per_attempt, p, 1),
    )


# --------------------------------------------------------------------------
# the monitoring result
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MaskingReport:
    p_before: float
    p_after: float
    r: int
    success_before: float
    success_after: float
    success_drop_pp: float
    cost_before: float
    cost_after: float
    cost_rise: float
    masking_ratio: float | None
    note: str

    def summary(self) -> str:
        lines = [
            f"per-attempt p fell {self.p_before:.3f} -> {self.p_after:.3f} "
            f"at r={self.r}",
            f"  observed success  {self.success_before:.4f} -> "
            f"{self.success_after:.4f}   ({self.success_drop_pp:+.2f} pp)",
            f"  expected cost     {self.cost_before:.3f} -> "
            f"{self.cost_after:.3f}   ({self.cost_rise:+.1%})",
        ]
        if self.masking_ratio is not None:
            lines.append(
                f"  cost signal is {self.masking_ratio:.1f}x the size of the "
                f"success signal"
            )
        lines.append(f"  {self.note}")
        return "\n".join(lines)


def masking(p_before: float, p_after: float, r: int) -> MaskingReport:
    """How much of a drop in `p` does a retry budget of `r` conceal?

    Compares the movement a success-rate monitor would see against the
    movement a cost monitor would see. The ratio is how much more sensitive
    the cost signal is.
    """
    _check(p_before, r)
    if not 0.0 <= p_after <= 1.0:
        raise ValueError(f"p_after must be in [0, 1], got {p_after}")

    sb, sa = effective_success(p_before, r), effective_success(p_after, r)
    cb, ca = expected_attempts(p_before, r), expected_attempts(p_after, r)
    drop_pp = (sa - sb) * 100.0
    rise = (ca - cb) / cb if cb else 0.0

    success_signal = abs(sa - sb)
    cost_signal = abs(rise)
    ratio = None if success_signal == 0 else cost_signal / success_signal

    if p_after >= p_before:
        note = "no degradation to mask."
    elif ratio is not None and ratio >= 5:
        note = (
            f"the success-rate monitor moves {abs(drop_pp):.2f} pp while cost "
            f"moves {rise:.1%}. A dashboard watching success rate would likely "
            f"miss this; one watching cost per task would not."
        )
    elif abs(drop_pp) < 1.0:
        note = (
            f"success rate moved under one point. Whether that is detectable "
            f"depends on your sample size, and at most realistic volumes it is "
            f"not."
        )
    else:
        note = "both signals moved appreciably."
    return MaskingReport(
        p_before=p_before,
        p_after=p_after,
        r=r,
        success_before=sb,
        success_after=sa,
        success_drop_pp=drop_pp,
        cost_before=cb,
        cost_after=ca,
        cost_rise=rise,
        masking_ratio=ratio,
        note=note,
    )


# --------------------------------------------------------------------------
# correlation
# --------------------------------------------------------------------------

def correlated_effective_success(
    difficulties: Sequence[float], r: int
) -> tuple[float, float]:
    """Effective success with per-item difficulty, and the independent figure.

    `difficulties` gives each item's own success probability. Returns
    (actual, naive) where `naive` applies the independent formula to the mean
    probability.

    The gap is the retry illusion. A population where most items succeed
    first time and a minority almost never succeed has a decent mean `p`, and
    retries do nothing for the minority — they fail again. Averaging `p`
    before applying `1-(1-p)^r` assumes away exactly the structure that
    determines the answer.
    """
    if not difficulties:
        raise ValueError("no difficulties supplied")
    if r < 1:
        raise ValueError(f"r must be >= 1, got {r}")
    for d in difficulties:
        if not 0.0 <= d <= 1.0:
            raise ValueError(f"difficulties must be in [0, 1], got {d}")

    actual = sum(1.0 - (1.0 - d) ** r for d in difficulties) / len(difficulties)
    mean_p = sum(difficulties) / len(difficulties)
    naive = 1.0 - (1.0 - mean_p) ** r
    return actual, naive
