import logging
import math
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard thresholds -- tuned for this domain (intraday stock predictions)
# ---------------------------------------------------------------------------

class StatConfig:
    """Statistical guard thresholds. All configurable via dict override."""
    # Minimum sample sizes
    MIN_ABSOLUTE: int = 10
    MIN_EXPLORATORY: int = 15
    MIN_CONFIRMATORY: int = 25
    MIN_STRATEGIC: int = 40
    MIN_PER_GROUP: int = 8

    # Effect-size minimums
    MIN_COHENS_H: float = 0.20
    MIN_LIFT: float = 0.05
    MIN_COHENS_D: float = 0.15

    # Inference settings
    ALPHA: float = 0.10
    BOOTSTRAP_N: int = 2000
    PERMUTATION_N: int = 5000
    OOS_FOLDS: int = 5

    # Proposal-level gates
    MIN_OCCURRENCES_PATTERN: int = 10
    MIN_OCCURRENCES_RULE: int = 8
    MIN_OCCURRENCES_TEMPORAL: int = 5
    MIN_FAILURES_ANALYSIS: int = 15
    MIN_FAILURE_PCT: float = 0.15

    # Conjunctive rules
    REQUIRE_P_VALUE: bool = True
    REQUIRE_EFFECT_SIZE: bool = True
    REQUIRE_OOS: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StatConfig":
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k.upper()):
                setattr(cfg, k.upper(), v)
            elif hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


CONF = StatConfig()


# ---------------------------------------------------------------------------
# Core statistical primitives
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: List[float],
    metric_func: Callable = np.mean,
    n_bootstrap: int = 2000,
    ci: float = 0.90,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrapped point estimate + confidence interval.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    data_arr = np.asarray(data, dtype=float)
    n = len(data_arr)
    if n < 3:
        point = float(metric_func(data_arr)) if n > 0 else 0.0
        return point, point, point

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data_arr, size=n, replace=True)
        estimates[i] = metric_func(sample)

    estimates.sort()
    low = int(n_bootstrap * (1 - ci) / 2)
    high = int(n_bootstrap * (1 + ci) / 2)
    point = float(metric_func(data_arr))
    return point, float(estimates[low]), float(estimates[high])


def bootstrap_proportion_ci(
    wins: int,
    total: int,
    n_bootstrap: int = 2000,
    ci: float = 0.90,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrapped win rate + confidence interval."""
    if total < 3:
        wr = wins / total if total > 0 else 0.0
        return wr, wr, wr

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample_wins = rng.binomial(n=total, p=wins / total)
        estimates[i] = sample_wins / total

    estimates.sort()
    low = int(n_bootstrap * (1 - ci) / 2)
    high = int(n_bootstrap * (1 + ci) / 2)
    return (wins / total), float(estimates[low]), float(estimates[high])


def permutation_test(
    wins_a: int,
    total_a: int,
    wins_b: int,
    total_b: int,
    n_permutations: int = 5000,
    seed: int = 42,
) -> float:
    """Two-sided permutation test for difference in proportions.

    Returns p-value (proportion of permuted differences >= observed).
    H0: true win rates are equal.
    """
    if total_a < 2 or total_b < 2:
        return 1.0

    p1 = wins_a / total_a
    p2 = wins_b / total_b
    observed_diff = abs(p1 - p2)

    all_outcomes = np.array([1] * wins_a + [0] * (total_a - wins_a) +
                            [1] * wins_b + [0] * (total_b - wins_b), dtype=int)
    n_total = len(all_outcomes)
    rng = np.random.default_rng(seed)
    count_extreme = 0

    for _ in range(n_permutations):
        rng.shuffle(all_outcomes)
        perm_a = all_outcomes[:total_a]
        perm_b = all_outcomes[total_a:]
        perm_diff = abs(perm_a.mean() - perm_b.mean())
        if perm_diff >= observed_diff:
            count_extreme += 1

    return (count_extreme + 1) / (n_permutations + 1)


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions.

    Interpretation: 0.20 = small, 0.50 = medium, 0.80 = large.
    """
    def arcsin_sqrt(p: float) -> float:
        p = max(0.0, min(1.0, p))
        return 2 * math.asin(math.sqrt(p))
    return abs(arcsin_sqrt(p1) - arcsin_sqrt(p2))


def cohens_d(
    mean1: float,
    mean2: float,
    std1: float,
    std2: float,
    n1: int,
    n2: int,
) -> float:
    """Cohen's d effect size for two means (pooled std)."""
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled = math.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
    if pooled < 1e-12:
        return 0.0
    return abs(mean1 - mean2) / pooled


def minimum_sample_required(
    effect_size: float,
    power: float = 0.80,
    alpha: float = 0.05,
) -> int:
    """Approximate minimum sample per group for a two-sample proportions test.

    Uses the normal approximation. Returns samples needed *per group*.
    """
    if effect_size < 1e-6:
        return 999999
    z_alpha = {0.10: 1.645, 0.05: 1.960, 0.01: 2.576}.get(alpha, 1.645)
    z_beta = {0.80: 0.842, 0.90: 1.282, 0.95: 1.645}.get(power, 0.842)
    n = int(2 * ((z_alpha + z_beta) / effect_size) ** 2)
    return max(n, 2)


def wald_ci(
    wins: int,
    total: int,
    z: float = 1.645,
) -> Tuple[float, float]:
    """Wald confidence interval for a binomial proportion."""
    if total < 1:
        return 0.0, 1.0
    p = wins / total
    se = math.sqrt(p * (1 - p) / total)
    return max(0.0, p - z * se), min(1.0, p + z * se)


# ---------------------------------------------------------------------------
# Out-of-Sample Validation
# ---------------------------------------------------------------------------

def out_of_sample_validate(
    data: List[dict],
    metric_fn: Callable[[List[dict]], float],
    n_folds: int = 5,
    test_frac: float = 0.25,
    seed: int = 42,
) -> Dict[str, Any]:
    """Walk-forward / shuffled out-of-sample validation.

    Returns dict with in/out scores, stability, and decay.
    """
    if len(data) < n_folds * 3:
        return {"valid": False, "reason": "insufficient_data", "n": len(data)}

    n = len(data)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    fold_size = max(1, int(n * test_frac))
    in_scores = []
    out_scores = []

    for fold in range(n_folds):
        test_start = fold * fold_size % n
        test_idx = set(range(test_start, min(test_start + fold_size, n)))
        if len(test_idx) < 2:
            continue
        train = [data[i] for i in indices if i not in test_idx]
        test = [data[i] for i in indices if i in test_idx]
        if len(train) < 2 or len(test) < 2:
            continue
        in_scores.append(metric_fn(train))
        out_scores.append(metric_fn(test))

    if len(in_scores) < 2:
        return {"valid": False, "reason": "too_few_folds", "n": len(data)}

    in_mean = float(np.mean(in_scores))
    out_mean = float(np.mean(out_scores))
    in_std = float(np.std(in_scores)) if len(in_scores) > 1 else 0.0
    out_std = float(np.std(out_scores)) if len(out_scores) > 1 else 0.0
    decay = in_mean - out_mean
    stability = 1.0 - min(1.0, (out_std / (abs(out_mean) + 0.001)))

    return {
        "valid": True,
        "in_sample_mean": round(in_mean, 4),
        "out_of_sample_mean": round(out_mean, 4),
        "in_sample_std": round(in_std, 4),
        "out_of_sample_std": round(out_std, 4),
        "decay": round(decay, 4),
        "stability": round(stability, 4),
        "n_folds": len(in_scores),
        "n_total": n,
    }


# ---------------------------------------------------------------------------
# Proposal-level validation gates
# ---------------------------------------------------------------------------

def validate_weight_proposal(
    current_wr: float,
    factor_data: Dict[str, Any],
    cfg: StatConfig = None,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Validate that a weight-rebalancing proposal passes statistical muster.

    Returns (pass, list_of_reasons, metrics_dict).
    """
    if cfg is None:
        cfg = CONF
    reasons = []
    metrics = {}

    for factor, data in factor_data.items():
        if data.get("skip"):
            continue
        n = data.get("sample_size", 0)
        wr = data.get("win_rate", 0)
        ci_low = data.get("ci_90_low", 0)
        ci_high = data.get("ci_90_high", 0)
        delta = data.get("weight_delta", 0)
        conf = data.get("confidence", 0)

        # 1. Sample size
        metrics[f"{factor}_n"] = n
        if n < cfg.MIN_CONFIRMATORY:
            reasons.append(f"{factor}: n={n} < MIN_CONFIRMATORY ({cfg.MIN_CONFIRMATORY})")
        elif n < cfg.MIN_EXPLORATORY:
            reasons.append(f"{factor}: n={n} < MIN_EXPLORATORY ({cfg.MIN_EXPLORATORY})")

        # 2. Effect size (Cohen's h vs baseline win rate)
        h = cohens_h(wr, current_wr)
        metrics[f"{factor}_cohens_h"] = round(h, 4)
        if h < cfg.MIN_COHENS_H:
            reasons.append(f"{factor}: Cohen's h={h:.3f} < MIN_COHENS_H ({cfg.MIN_COHENS_H})")

        # 3. CI includes or crosses 0.50 (null)
        if ci_low <= 0.50 <= ci_high:
            reasons.append(f"{factor}: 90% CI [{ci_low:.1%}, {ci_high:.1%}] crosses 0.50")

        # 4. Lift
        lift = wr - current_wr
        metrics[f"{factor}_lift"] = round(lift, 4)
        if abs(lift) < cfg.MIN_LIFT and abs(delta) >= 0.01:
            reasons.append(f"{factor}: lift={lift:.1%} < MIN_LIFT ({cfg.MIN_LIFT:.0%})")

        # 5. Confidence from bootstrap
        metrics[f"{factor}_bootstrap_conf"] = conf
        if conf < 0.15:
            reasons.append(f"{factor}: bootstrap confidence={conf:.2f} < 0.15")

    # Decide pass/fail
    hard_fails = [r for r in reasons if "MIN_CONFIRMATORY" in r or "crosses 0.50" in r]
    soft_warns = [r for r in reasons if r not in hard_fails]

    passes = len(hard_fails) == 0
    return passes, reasons, metrics


def validate_pattern_discovery(
    field: str,
    above_n: int,
    above_wr: float,
    below_n: int,
    below_wr: float,
    total_n: int,
    cfg: StatConfig = None,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Validate a threshold-based pattern discovery finding."""
    if cfg is None:
        cfg = CONF
    reasons = []
    metrics = {}

    # 1. Minimum absolute occurrences
    metrics["above_n"] = above_n
    if above_n < cfg.MIN_OCCURRENCES_PATTERN:
        reasons.append(f"above_n={above_n} < MIN_OCCURRENCES_PATTERN ({cfg.MIN_OCCURRENCES_PATTERN})")
    if below_n < cfg.MIN_PER_GROUP:
        reasons.append(f"below_n={below_n} < MIN_PER_GROUP ({cfg.MIN_PER_GROUP})")

    # 2. Effect size
    h = cohens_h(above_wr, below_wr)
    metrics["cohens_h"] = round(h, 4)
    if h < cfg.MIN_COHENS_H:
        reasons.append(f"Cohen's h={h:.3f} < MIN_COHENS_H ({cfg.MIN_COHENS_H})")

    # 3. Lift
    lift = above_wr - below_wr
    metrics["lift"] = round(lift, 4)
    if abs(lift) < cfg.MIN_LIFT:
        reasons.append(f"lift={lift:.1%} < MIN_LIFT ({cfg.MIN_LIFT:.0%})")

    # 4. Permutation test p-value
    if above_n > 0 and below_n > 0:
        p = permutation_test(
            int(above_wr * above_n), above_n,
            int(below_wr * below_n), below_n,
        )
        metrics["p_value"] = round(p, 4)
        if p > cfg.ALPHA:
            reasons.append(f"p_value={p:.4f} > ALPHA={cfg.ALPHA}")

    # 5. Fraction of total
    frac = (above_n + below_n) / max(total_n, 1)
    metrics["fraction_of_total"] = round(frac, 4)
    if frac < 0.05 and above_n < cfg.MIN_EXPLORATORY:
        reasons.append(f"fraction_of_total={frac:.1%} < 5%")

    passes = len(reasons) == 0
    return passes, reasons, metrics


def validate_proposal_dict(
    proposal: Dict[str, Any],
    cfg: StatConfig = None,
) -> Tuple[bool, List[str]]:
    """Generic validation for any proposal dict.

    Checks that required statistical fields are present and meet thresholds.
    """
    if cfg is None:
        cfg = CONF
    reasons = []

    sample_sizes = proposal.get("sample_sizes", {})
    conf_metrics = proposal.get("confidence_metrics", {})
    expected_impact = proposal.get("expected_impact", {})
    ptype = proposal.get("proposal_type", "unknown")

    # Every proposal must have at least one sample size >= MIN_ABSOLUTE
    all_n = list(sample_sizes.values()) if isinstance(sample_sizes, dict) else [sample_sizes]
    max_n = max(all_n) if all_n else 0
    if max_n < cfg.MIN_ABSOLUTE:
        reasons.append(f"max sample size ({max_n}) < MIN_ABSOLUTE ({cfg.MIN_ABSOLUTE})")

    # Weight rebalance proposals need higher confidence
    if ptype == "weight_rebalance":
        if max_n < cfg.MIN_CONFIRMATORY:
            reasons.append(f"weight_rebalance needs n >= {cfg.MIN_CONFIRMATORY} (got {max_n})")

    # Temporal patterns need per-group minima
    if ptype == "temporal_insight":
        min_group = min(sample_sizes.values()) if isinstance(sample_sizes, dict) else 0
        if min_group < cfg.MIN_OCCURRENCES_TEMPORAL:
            reasons.append(f"min temporal group ({min_group}) < {cfg.MIN_OCCURRENCES_TEMPORAL}")

    # Failure analysis
    if ptype == "failure_insight":
        if max_n < cfg.MIN_FAILURES_ANALYSIS:
            reasons.append(f"failures ({max_n}) < MIN_FAILURES_ANALYSIS ({cfg.MIN_FAILURES_ANALYSIS})")

    passes = len(reasons) == 0
    return passes, reasons


# ---------------------------------------------------------------------------
# Pre-execution gate: should we even run analysis?
# ---------------------------------------------------------------------------

def should_run_research(
    total_evaluated: int,
    cfg: StatConfig = None,
) -> Tuple[bool, str]:
    """Check if there's enough data to run research at all."""
    if cfg is None:
        cfg = CONF
    if total_evaluated < cfg.MIN_ABSOLUTE:
        return False, f"Need >= {cfg.MIN_ABSOLUTE} evaluated recs (have {total_evaluated})"
    if total_evaluated < cfg.MIN_EXPLORATORY:
        return True, f"Exploratory only: {total_evaluated} recs (recommend >= {cfg.MIN_EXPLORATORY})"
    return True, ""


def should_generate_proposals(
    total_evaluated: int,
    cfg: StatConfig = None,
) -> Tuple[bool, str]:
    """Should we generate strategy proposals from this run?"""
    if cfg is None:
        cfg = CONF
    if total_evaluated < cfg.MIN_EXPLORATORY:
        return False, f"Need >= {cfg.MIN_EXPLORATORY} evaluated recs for proposals (have {total_evaluated})"
    if total_evaluated < cfg.MIN_CONFIRMATORY:
        return True, f"Proposals are exploratory: {total_evaluated} recs (recommend >= {cfg.MIN_CONFIRMATORY})"
    return True, ""


def guard_tag(passes: bool, warns: int = 0) -> str:
    """Return a visual tag for CLI output."""
    if not passes:
        return "[BLOCKED]"
    if warns > 0:
        return "[LOW-CONF]"
    return "[PASS]"
