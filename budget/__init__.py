"""budget — retries, cost, and the degradation retries hide.

Cost per successful task is c/p regardless of the retry budget: retries buy
reliability, not efficiency. And a retry budget conceals a falling per-attempt
success rate from a success-rate monitor while cost reports it immediately —
so cost is the leading indicator and success rate the lagging one.
"""

from __future__ import annotations

from .retries import (
    MaskingReport,
    RetryReport,
    assess,
    correlated_effective_success,
    cost_per_success,
    effective_success,
    expected_attempts,
    expected_cost,
    masking,
    retries_for_target,
)

__all__ = [
    "MaskingReport", "RetryReport", "assess", "correlated_effective_success",
    "cost_per_success", "effective_success", "expected_attempts",
    "expected_cost", "masking", "retries_for_target",
]
__version__ = "0.1.0"
