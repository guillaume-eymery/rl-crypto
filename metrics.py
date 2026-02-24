"""
metrics.py — Métriques quantitatives complètes pour l'évaluation RL trading.

Métriques:
  - Sharpe, Sortino, Calmar annualisés
  - Max drawdown & drawdown duration
  - Win rate, profit factor
  - Information Ratio vs benchmark
  - Tests statistiques : permutation test, bootstrap CI
  - Analyse de stabilité walk-forward
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PerformanceMetrics:
    # Returns
    total_return: float = 0.0
    cagr: float = 0.0
    ann_volatility: float = 0.0

    # Risk-adjusted
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    omega: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0   # en jours
    avg_drawdown: float = 0.0

    # Win/Loss
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # vs Benchmark
    information_ratio: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    tracking_error: float = 0.0

    # Stats
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0        # Value at Risk 95%
    cvar_95: float = 0.0       # Conditional VaR

    # Costs
    total_fees: float = 0.0
    avg_daily_turnover: float = 0.0

    # Bootstrap CI sur Sharpe
    sharpe_ci_low: float = float("nan")
    sharpe_ci_high: float = float("nan")

    def __str__(self) -> str:
        lines = [
            "─" * 50,
            f"  Total Return      : {self.total_return:+.2%}",
            f"  CAGR              : {self.cagr:+.2%}",
            f"  Ann. Volatility   : {self.ann_volatility:.2%}",
            "─" * 50,
            f"  Sharpe            : {self.sharpe:+.3f}  [{self.sharpe_ci_low:.2f}, {self.sharpe_ci_high:.2f}] 95% CI",
            f"  Sortino           : {self.sortino:+.3f}",
            f"  Calmar            : {self.calmar:+.3f}",
            f"  Omega             : {self.omega:.3f}",
            "─" * 50,
            f"  Max Drawdown      : {self.max_drawdown:.2%}",
            f"  Max DD Duration   : {self.max_drawdown_duration}d",
            f"  VaR 95%           : {self.var_95:.2%}",
            f"  CVaR 95%          : {self.cvar_95:.2%}",
            "─" * 50,
            f"  Win Rate          : {self.win_rate:.2%}",
            f"  Profit Factor     : {self.profit_factor:.2f}",
            f"  Skewness          : {self.skewness:+.2f}",
            f"  Kurtosis          : {self.kurtosis:+.2f}",
            "─" * 50,
            f"  Info Ratio (vs BM): {self.information_ratio:+.3f}",
            f"  Beta              : {self.beta:+.3f}",
            f"  Alpha (ann.)      : {self.alpha:+.3f}",
            f"  Tracking Error    : {self.tracking_error:.2%}",
            "─" * 50,
            f"  Total Fees        : {self.total_fees:.2f}",
            f"  Avg Daily Turnover: {self.avg_daily_turnover:.2%}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    equity_curve: np.ndarray,
    benchmark_equity: np.ndarray | None = None,
    fees_per_step: np.ndarray | None = None,
    turnover_per_step: np.ndarray | None = None,
    trading_days_per_year: int = 252,
    n_bootstrap: int = 1000,
    risk_free_rate: float = 0.04,  # 4% annuel
) -> PerformanceMetrics:
    """
    Calcule toutes les métriques à partir d'une courbe d'equity.

    Args:
        equity_curve: valeur du portefeuille à chaque step (daily).
        benchmark_equity: courbe de l'indice de référence (même longueur).
        fees_per_step: frais payés à chaque step.
        turnover_per_step: turnover à chaque step.
        trading_days_per_year: 252 pour actions, 365 pour crypto.
        n_bootstrap: itérations bootstrap pour CI Sharpe.
        risk_free_rate: taux sans risque annuel.
    """
    m = PerformanceMetrics()
    equity_curve = np.asarray(equity_curve, dtype=np.float64)
    n = len(equity_curve)

    if n < 2:
        return m

    # Daily returns
    daily_ret = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-12)
    rf_daily = risk_free_rate / trading_days_per_year
    excess_ret = daily_ret - rf_daily

    # ── Returns ───────────────────────────────────────────────────────────
    m.total_return = float(equity_curve[-1] / equity_curve[0] - 1.0)
    years = n / trading_days_per_year
    m.cagr = float((equity_curve[-1] / equity_curve[0]) ** (1 / max(years, 1e-6)) - 1.0)
    m.ann_volatility = float(daily_ret.std() * np.sqrt(trading_days_per_year))

    # ── Sharpe / Sortino ──────────────────────────────────────────────────
    mu_excess = excess_ret.mean()
    sigma_excess = excess_ret.std() + 1e-12
    m.sharpe = float(mu_excess / sigma_excess * np.sqrt(trading_days_per_year))

    downside = excess_ret[excess_ret < 0]
    downside_vol = downside.std() + 1e-12 if len(downside) > 1 else 1e-12
    m.sortino = float(mu_excess / downside_vol * np.sqrt(trading_days_per_year))

    # ── Drawdown ──────────────────────────────────────────────────────────
    rolling_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - rolling_max) / np.maximum(rolling_max, 1e-12)
    m.max_drawdown = float(-drawdowns.min())
    m.avg_drawdown = float(-drawdowns[drawdowns < 0].mean()) if (drawdowns < 0).any() else 0.0

    # Drawdown duration (longest period below peak)
    in_dd = drawdowns < 0
    max_dur = 0
    cur_dur = 0
    for flag in in_dd:
        if flag:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0
    m.max_drawdown_duration = max_dur

    # ── Calmar ────────────────────────────────────────────────────────────
    m.calmar = float(m.cagr / max(m.max_drawdown, 1e-9))

    # ── Omega ratio ───────────────────────────────────────────────────────
    gains = daily_ret[daily_ret > rf_daily] - rf_daily
    losses = rf_daily - daily_ret[daily_ret <= rf_daily]
    m.omega = float(gains.sum() / max(losses.sum(), 1e-12))

    # ── Win rate / Profit factor ──────────────────────────────────────────
    wins = daily_ret[daily_ret > 0]
    losses_abs = daily_ret[daily_ret < 0]
    m.win_rate = float(len(wins) / max(len(daily_ret), 1))
    m.profit_factor = float(wins.sum() / max(abs(losses_abs.sum()), 1e-12))

    # ── Distribution ─────────────────────────────────────────────────────
    m.skewness = float(_skewness(daily_ret))
    m.kurtosis = float(_kurtosis(daily_ret))
    m.var_95 = float(-np.percentile(daily_ret, 5))
    m.cvar_95 = float(-daily_ret[daily_ret <= -m.var_95].mean()) if (daily_ret <= -m.var_95).any() else m.var_95

    # ── vs Benchmark ─────────────────────────────────────────────────────
    if benchmark_equity is not None:
        bm = np.asarray(benchmark_equity, dtype=np.float64)
        bm_ret = np.diff(bm) / np.maximum(bm[:-1], 1e-12)
        min_len = min(len(daily_ret), len(bm_ret))
        dr = daily_ret[:min_len]
        br = bm_ret[:min_len]

        active_ret = dr - br
        te = active_ret.std() * np.sqrt(trading_days_per_year) + 1e-12
        m.tracking_error = float(te)
        m.information_ratio = float(active_ret.mean() * trading_days_per_year / te)

        # Beta / Alpha (OLS)
        cov = np.cov(dr, br)
        bm_var = bm_ret.var() + 1e-12
        m.beta = float(cov[0, 1] / bm_var)
        m.alpha = float((dr.mean() - m.beta * br.mean()) * trading_days_per_year)

    # ── Costs ─────────────────────────────────────────────────────────────
    if fees_per_step is not None:
        m.total_fees = float(np.sum(fees_per_step))
    if turnover_per_step is not None:
        m.avg_daily_turnover = float(np.mean(turnover_per_step))

    # ── Bootstrap CI Sharpe ───────────────────────────────────────────────
    ci = _bootstrap_sharpe_ci(excess_ret, n_bootstrap, trading_days_per_year)
    m.sharpe_ci_low = ci[0]
    m.sharpe_ci_high = ci[1]

    return m


# ─────────────────────────────────────────────────────────────────────────────
# Statistical tests
# ─────────────────────────────────────────────────────────────────────────────

def permutation_test(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    n_permutations: int = 10_000,
    metric: str = "sharpe",
) -> dict[str, float]:
    """
    Test de permutation : H0 = la stratégie ne bat pas le benchmark.
    Retourne le p-value.

    La statistique de test = Sharpe(stratégie) - Sharpe(benchmark).
    On permute aléatoirement l'attribution des returns entre les deux.
    """
    s = np.asarray(strategy_returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)
    min_len = min(len(s), len(b))
    s, b = s[:min_len], b[:min_len]

    def _stat(a, c):
        if metric == "sharpe":
            return _sharpe(a) - _sharpe(c)
        elif metric == "return":
            return a.mean() - c.mean()
        else:
            return _sharpe(a) - _sharpe(c)

    observed = _stat(s, b)

    combined = np.concatenate([s, b])
    rng = np.random.default_rng(42)
    null_dist = []
    for _ in range(n_permutations):
        perm = rng.permutation(len(combined))
        a = combined[perm[:min_len]]
        c = combined[perm[min_len:]]
        null_dist.append(_stat(a, c))

    null_dist = np.array(null_dist)
    p_value = float((null_dist >= observed).mean())

    return {
        "observed_stat": float(observed),
        "p_value": p_value,
        "significant_5pct": p_value < 0.05,
        "significant_1pct": p_value < 0.01,
        "null_mean": float(null_dist.mean()),
        "null_std": float(null_dist.std()),
    }


def walk_forward_consistency(
    fold_metrics: list[PerformanceMetrics],
) -> dict[str, float]:
    """
    Analyse la consistance des performances across les folds walk-forward.
    Un bon modèle doit être relativement stable, pas juste chanceux sur 1 fold.
    """
    sharpes = [m.sharpe for m in fold_metrics]
    calmars = [m.calmar for m in fold_metrics]
    total_rets = [m.total_return for m in fold_metrics]

    positive_sharpe_pct = float(np.mean([s > 0 for s in sharpes]))

    return {
        "mean_sharpe": float(np.mean(sharpes)),
        "std_sharpe": float(np.std(sharpes)),
        "min_sharpe": float(np.min(sharpes)),
        "max_sharpe": float(np.max(sharpes)),
        "positive_folds_pct": positive_sharpe_pct,
        "mean_calmar": float(np.mean(calmars)),
        "mean_total_ret": float(np.mean(total_rets)),
        "consistency_score": float(positive_sharpe_pct * np.mean(sharpes)),  # heuristic
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sharpe(returns: np.ndarray, rf_daily: float = 0.0) -> float:
    excess = returns - rf_daily
    std = excess.std() + 1e-12
    return float(excess.mean() / std * np.sqrt(252))


def _skewness(x: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mu = x.mean()
    sigma = x.std() + 1e-12
    return float(((x - mu) ** 3).mean() / sigma ** 3)


def _kurtosis(x: np.ndarray) -> float:
    n = len(x)
    if n < 4:
        return 0.0
    mu = x.mean()
    sigma = x.std() + 1e-12
    return float(((x - mu) ** 4).mean() / sigma ** 4 - 3)


def _bootstrap_sharpe_ci(
    excess_returns: np.ndarray,
    n_bootstrap: int,
    trading_days: int,
    ci: float = 0.95,
) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    n = len(excess_returns)
    sharpes = []
    for _ in range(n_bootstrap):
        sample = rng.choice(excess_returns, size=n, replace=True)
        s = sample.mean() / (sample.std() + 1e-12) * np.sqrt(trading_days)
        sharpes.append(s)
    alpha = (1 - ci) / 2
    return (
        float(np.percentile(sharpes, 100 * alpha)),
        float(np.percentile(sharpes, 100 * (1 - alpha))),
    )


def equity_from_returns(
    returns: np.ndarray,
    initial: float = 10_000.0,
) -> np.ndarray:
    """Reconstruit une courbe d'equity depuis des returns daily."""
    curve = np.cumprod(1 + np.asarray(returns)) * initial
    return np.concatenate([[initial], curve])
