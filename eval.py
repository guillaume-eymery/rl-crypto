"""
eval.py — Évaluation cross-market du modèle BTC-LSTM.

Modes :
  Normal (intra-asset) :
    python eval.py --symbol BTC-USD
    python eval.py --symbol ETH-USD   ← out-of-time, features alignées

  Cross-market (nouvel asset, normalisé par l'asset cible) :
    python eval.py --symbol SPY --cross-market
    python eval.py --symbol GC=F --cross-market
    python eval.py --symbol EURUSD=X --cross-market
    python eval.py --cross-market-all   ← évalue TOUS les assets du registry

Différences du mode cross-market :
  1. Fees/annual_days propres à l'asset cible (depuis asset_registry)
  2. Features normalisées avec les stats de l'asset cible (pas BTC)
  3. Features crypto manquantes (halving, funding, etc.) → zéros
  4. VecNormalize chargé en mode passthrough (stats figées BTC ignorées)
  5. Rapport comparatif final si --cross-market-all
"""

from __future__ import annotations
import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from features.make_features import (
    load_data, make_features, load_feature_config,
    normalize_for_target, pad_features_to_expected,
)
from envs.trading_env_lstm import TradingEnvLSTM
from metrics import compute_metrics, permutation_test
from asset_registry import get_asset_config, REGISTRY


# ─────────────────────────────────────────────────────────────────────────────
# Chargement du modèle
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path, vec_env):
    try:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(model_path, env=vec_env)
        print(f"[Eval] Loaded RecurrentPPO (LSTM)")
        return model, "lstm"
    except Exception:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, env=vec_env)
        print(f"[Eval] Loaded PPO (MLP fallback)")
        return model, "mlp"


# ─────────────────────────────────────────────────────────────────────────────
# Épisode de trading
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(model, vec_env, model_type="lstm", deterministic=True):
    obs = vec_env.reset()
    done = False
    equities, positions, fees, turnovers = [], [], [], []
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    while not done:
        if model_type == "lstm":
            action, lstm_states = model.predict(
                obs, state=lstm_states,
                episode_start=episode_starts, deterministic=deterministic,
            )
            episode_starts = np.zeros((1,), dtype=bool)
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, dones, infos = vec_env.step(action)
        done = bool(dones[0])
        if model_type == "lstm":
            episode_starts = dones
        info = infos[0]
        equities.append(info["equity"])
        positions.append(info.get("pos_frac", 0.0))
        fees.append(info.get("fee_paid", 0.0))
        turnovers.append(info.get("turnover", 0.0))
    return {
        "equities": np.array(equities, dtype=np.float64),
        "positions": np.array(positions, dtype=np.float64),
        "fees": np.array(fees, dtype=np.float64),
        "turnovers": np.array(turnovers, dtype=np.float64),
    }


def run_random_agent(X_test, p_test, fee, allow_short, annual_days, n_episodes=10):
    all_equities = []
    for _ in range(n_episodes):
        env = TradingEnvLSTM(X_test, p_test, fee=fee, allow_short=allow_short,
                              randomize_start=False, annual_days=annual_days)
        obs, _ = env.reset()
        done = False
        equities = [10_000.0]
        while not done:
            action = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            equities.append(info["equity"])
        all_equities.append(np.array(equities))
    return all_equities


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline d'évaluation pour UN asset
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_asset(
    symbol: str,
    model_path: str,
    vecnorm_path: str,
    feat_cfg: dict,
    cross_market: bool = False,
    split: float = 0.80,
    save_plot: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Évalue le modèle sur un asset.
    Retourne un dict de métriques.
    """

    # ── Config de l'asset ────────────────────────────────────────────────────
    if cross_market or symbol not in ["BTC-USD"]:
        try:
            acfg = get_asset_config(symbol)
        except ValueError:
            # Asset inconnu → on utilise des defaults crypto
            acfg = {
                "name": symbol, "asset_class": "crypto",
                "start_date": "2017-01-01", "annual_days": 365,
                "fee": 0.001, "funding_cost": 0.0002,
                "allow_short": True, "max_dd_kill": 0.40,
                "min_ep_steps": 252, "is_crypto": True,
                "cross_assets": [], "norm_mode": "target",
            }
    else:
        acfg = get_asset_config("BTC-USD")

    if verbose:
        print(f"\n[Eval] Symbol={symbol} | Class={acfg['asset_class']} | "
              f"Fee={acfg['fee']*100:.2f}bps | annual_days={acfg['annual_days']}")

    # ── Téléchargement des données ───────────────────────────────────────────
    # En mode cross-market, on utilise les cross-assets propres à l'asset
    # En mode intra-asset, on utilise ceux du feat_cfg (training BTC)
    if cross_market:
        cross_syms = [s for s in acfg["cross_assets"] if s != symbol]
    else:
        cross_syms = [s for s in feat_cfg.get("cross_asset_symbols", []) if s != symbol]

    symbols_to_load = [symbol] + cross_syms
    all_data = load_data(symbols_to_load, start=acfg["start_date"])

    df = all_data[symbol]
    cross_closes = {s: all_data[s]["Close"].squeeze()
                    for s in cross_syms if s in all_data}
    cross_vols = {s: all_data[s]["Volume"].squeeze()
                  for s in cross_syms if s in all_data}

    # ── Feature engineering ──────────────────────────────────────────────────
    # is_crypto=True même pour SPY → les features crypto seront à zéro naturellement
    # (dd_from_ath, halving, etc. calculées mais non significatives sur equity)
    # On utilise is_crypto=acfg["is_crypto"] pour générer les bonnes features
    X, prices, idx = make_features(
        df,
        cross_asset_closes=cross_closes or None,
        cross_asset_volumes=cross_vols or None,
        include_calendar=feat_cfg.get("include_calendar", True),
        is_crypto=True,  # Force le schéma crypto pour compatibilité avec le modèle entraîné,
    )
    current_names = make_features._last_feature_names.copy()

    # ── Alignement des features sur le schéma du modèle entraîné ────────────
    expected_names = feat_cfg.get("feature_names", current_names)
    n_expected = feat_cfg.get("n_features", X.shape[1])

    if X.shape[1] != n_expected or set(current_names) != set(expected_names):
        missing = [f for f in expected_names if f not in current_names]
        if missing and verbose:
            print(f"[Eval] Features manquantes (→ zéros): {missing}")
        X = pad_features_to_expected(X, current_names, expected_names)

    if X.shape[1] != n_expected:
        raise ValueError(f"Feature mismatch: got {X.shape[1]}, expected {n_expected}")

    if verbose:
        print(f"[Eval] Features: {X.shape} | {idx[0].date()} → {idx[-1].date()}")

    # ── Split train/test ─────────────────────────────────────────────────────
    split_idx = int(split * len(prices))
    X_train, X_test = X[:split_idx], X[split_idx:]
    p_train, p_test = prices[:split_idx], prices[split_idx:]

    if verbose:
        print(f"[Eval] Test set: {len(X_test)} steps "
              f"({idx[split_idx].date()} → {idx[-1].date()})")

    # ── Normalisation ────────────────────────────────────────────────────────
    # Mode cross-market : normalise avec les stats de l'asset cible
    # → le modèle reçoit des observations dans une plage cohérente
    # Mode intra-asset : normalise avec les stats du train set de l'asset
    # Dans les deux cas : "target" = stats du train set de l'asset évalué
    X_train_s, X_test_s, _, _ = normalize_for_target(
        X_train, X_test, mode="target"
    )

    # ── Env de test ──────────────────────────────────────────────────────────
    def _make_env():
        return TradingEnvLSTM(
            features=X_test_s,
            prices=p_test,
            fee=acfg["fee"],
            funding_cost_daily=acfg["funding_cost"],
            allow_short=acfg["allow_short"],
            reward_type="return",
            randomize_start=False,
            min_episode_steps=acfg["min_ep_steps"],
            annual_days=acfg["annual_days"],
            max_drawdown_kill=acfg["max_dd_kill"],
        )

    vec_env = DummyVecEnv([_make_env])

    # Chargement du VecNormalize
    # En mode cross-market : on le charge mais on le met en passthrough
    # (training=False, norm_obs=False) pour éviter que les stats BTC
    # re-normalisent des features déjà normalisées avec les stats de l'asset cible
    try:
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        if cross_market:
            # Passthrough : désactive la re-normalisation
            # Les features sont déjà normalisées avec les stats de l'asset cible
            vec_env.norm_obs = False
    except Exception as e:
        if verbose:
            print(f"[Eval] VecNormalize non trouvé ({e}) → pas de normalisation")
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=False)
        vec_env.training = False

    # ── Chargement du modèle ─────────────────────────────────────────────────
    model, model_type = load_model(model_path, vec_env)

    # ── Run episode ──────────────────────────────────────────────────────────
    ep = run_episode(model, vec_env, model_type=model_type)
    rl_eq = ep["equities"]
    positions = ep["positions"]
    fees_arr = ep["fees"]
    turnovers_arr = ep["turnovers"]
    n = len(rl_eq)

    # ── Benchmark B&H ────────────────────────────────────────────────────────
    bh_eq = 10_000.0 * p_test[:n] / float(p_test[0])

    # ── Métriques ────────────────────────────────────────────────────────────
    rl_m = compute_metrics(
        rl_eq, benchmark_equity=bh_eq[:n],
        fees_per_step=fees_arr, turnover_per_step=turnovers_arr,
        trading_days_per_year=acfg["annual_days"],
    )
    bh_m = compute_metrics(bh_eq[:n], trading_days_per_year=acfg["annual_days"])

    if verbose:
        print("\n══════════════════════════════════════════════════")
        print(f"  {symbol} ({acfg['name']}) — RL STRATEGY (LSTM)")
        print(rl_m)
        print(f"\n  {symbol} — BUY & HOLD")
        print(bh_m)
        print("══════════════════════════════════════════════════")

    # ── Random baseline ──────────────────────────────────────────────────────
    rand_eps = run_random_agent(X_test_s, p_test, acfg["fee"],
                                 acfg["allow_short"], acfg["annual_days"])
    rand_sharpes = [compute_metrics(e, trading_days_per_year=acfg["annual_days"]).sharpe
                    for e in rand_eps]
    rand_mean = float(np.mean(rand_sharpes))
    rand_std = float(np.std(rand_sharpes))

    if verbose:
        print(f"\n[Eval] Random Sharpe: {rand_mean:.3f} ± {rand_std:.3f}")
        print(f"[Eval] RL beats random: {rl_m.sharpe > rand_mean}")

    # ── Permutation test ─────────────────────────────────────────────────────
    rl_ret = np.diff(rl_eq) / np.maximum(rl_eq[:-1], 1e-12)
    bh_ret = np.diff(bh_eq[:n]) / np.maximum(bh_eq[:n-1], 1e-12)
    perm = permutation_test(rl_ret, bh_ret, n_permutations=10_000)
    if verbose:
        print(f"[Permutation Test] p-value: {perm['p_value']:.4f} | "
              f"Significant: {perm['significant_5pct']}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    if save_plot:
        _plot(symbol, acfg, rl_eq, bh_eq, positions, fees_arr, turnovers_arr,
              rl_m, bh_m, perm, save_plot)

    return {
        "symbol": symbol,
        "asset_class": acfg["asset_class"],
        "cross_market": cross_market,
        "test_steps": n,
        "test_start": str(idx[split_idx].date()),
        "test_end": str(idx[-1].date()),
        "rl": {
            "total_return": float(rl_m.total_return),
            "cagr": float(rl_m.cagr),
            "sharpe": float(rl_m.sharpe),
            "sharpe_ci": [float(rl_m.sharpe_ci_low), float(rl_m.sharpe_ci_high)],
            "sortino": float(rl_m.sortino),
            "calmar": float(rl_m.calmar),
            "max_drawdown": float(rl_m.max_drawdown),
            "information_ratio": float(rl_m.information_ratio),
            "total_fees": float(rl_m.total_fees),
            "avg_turnover": float(rl_m.avg_daily_turnover),
            "win_rate": float(rl_m.win_rate),
        },
        "bh": {
            "total_return": float(bh_m.total_return),
            "sharpe": float(bh_m.sharpe),
            "max_drawdown": float(bh_m.max_drawdown),
        },
        "random_sharpe_mean": rand_mean,
        "random_sharpe_std": rand_std,
        "permutation_p": float(perm["p_value"]),
        "significant": bool(perm["significant_5pct"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def _plot(symbol, acfg, rl_eq, bh_eq, positions, fees, turnovers,
          rl_m, bh_m, perm, save_path):
    n = len(rl_eq)
    bh_n = bh_eq[:n]
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Equity
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(rl_eq, label=f"RL LSTM (Sharpe={rl_m.sharpe:.2f})", lw=1.5, color="#2196F3")
    ax.plot(bh_n, label=f"B&H (Sharpe={bh_m.sharpe:.2f})", lw=1.5, color="#FF5722", alpha=0.8)
    ax.set_title(f"{symbol} ({acfg['name']}) — {acfg['asset_class'].upper()} | "
                 f"fee={acfg['fee']*100:.2f}bps | {'cross-market eval' if True else ''}",
                 fontweight="bold")
    ax.set_ylabel("Equity ($)"); ax.legend(); ax.grid(alpha=0.3)

    # Drawdown
    ax = fig.add_subplot(gs[1, :2])
    rm = np.maximum.accumulate(rl_eq)
    dd = (rl_eq - rm) / np.maximum(rm, 1e-12)
    rm_b = np.maximum.accumulate(bh_n)
    dd_b = (bh_n - rm_b) / np.maximum(rm_b, 1e-12)
    ax.fill_between(range(n), dd*100, alpha=0.5, color="#2196F3",
                    label=f"RL DD (max={rl_m.max_drawdown:.1%})")
    ax.fill_between(range(n), dd_b*100, alpha=0.3, color="#FF5722",
                    label=f"B&H DD (max={bh_m.max_drawdown:.1%})")
    ax.set_title("Drawdown (%)"); ax.legend(); ax.grid(alpha=0.3)

    # Positions
    ax = fig.add_subplot(gs[2, :2])
    pos = np.array(positions)
    ax.plot(pos, lw=0.8, color="#4CAF50", alpha=0.8)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_title(f"Position | mean={pos.mean():.2f} | "
                 f"% long={( pos>0.05).mean():.1%} | "
                 f"% short={(pos<-0.05).mean():.1%} | "
                 f"% flat={( abs(pos)<=0.05).mean():.1%}")
    ax.set_ylabel("Position fraction"); ax.grid(alpha=0.3)

    # Metrics table
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    data = [
        ["Metric", "RL LSTM", "B&H"],
        ["Total Return", f"{rl_m.total_return:+.1%}", f"{bh_m.total_return:+.1%}"],
        ["CAGR", f"{rl_m.cagr:+.1%}", f"{bh_m.cagr:+.1%}"],
        ["Ann. Vol", f"{rl_m.ann_volatility:.1%}", f"{bh_m.ann_volatility:.1%}"],
        ["Sharpe", f"{rl_m.sharpe:.2f}", f"{bh_m.sharpe:.2f}"],
        ["Sortino", f"{rl_m.sortino:.2f}", "—"],
        ["Calmar", f"{rl_m.calmar:.2f}", f"{bh_m.calmar:.2f}"],
        ["Max DD", f"{rl_m.max_drawdown:.1%}", f"{bh_m.max_drawdown:.1%}"],
        ["Info Ratio", f"{rl_m.information_ratio:.2f}", "—"],
        ["Total Fees", f"${rl_m.total_fees:.0f}", "—"],
        ["Avg Turnover", f"{rl_m.avg_daily_turnover:.1%}", "—"],
        ["p-value", f"{perm['p_value']:.3f}", "—"],
    ]
    t = ax.table(cellText=data[1:], colLabels=data[0], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.3)
    ax.set_title("Summary", fontweight="bold", pad=20)

    # Return dist
    ax = fig.add_subplot(gs[1, 2])
    rl_ret = np.diff(rl_eq) / np.maximum(rl_eq[:-1], 1e-12)
    bh_ret = np.diff(bh_n) / np.maximum(bh_n[:-1], 1e-12)
    ax.hist(rl_ret*100, bins=60, alpha=0.7, color="#2196F3", label="RL", density=True)
    ax.hist(bh_ret*100, bins=60, alpha=0.5, color="#FF5722", label="B&H", density=True)
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.set_title(f"Return Dist | Skew={rl_m.skewness:+.1f} Kurt={rl_m.kurtosis:.1f}")
    ax.set_xlabel("Return (%)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Turnover
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(np.array(turnovers)*100, lw=0.7, color="#FF9800", alpha=0.7)
    ax.axhline(np.mean(turnovers)*100, color="#FF9800", lw=1.5, ls="--",
               label=f"Mean={np.mean(turnovers):.1%}")
    ax.set_title("Daily Turnover (%)"); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(f"RL Cross-Market Eval — {symbol}", fontsize=13, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Eval] Plot saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Rapport comparatif multi-asset
# ─────────────────────────────────────────────────────────────────────────────

def print_cross_market_report(results: list[dict]):
    """Affiche un tableau comparatif de tous les assets évalués."""
    print("\n" + "═"*90)
    print("  RAPPORT CROSS-MARKET — Modèle BTC-LSTM évalué sur tous les assets")
    print("═"*90)
    header = f"{'Symbol':<14} {'Class':<10} {'Steps':>6} {'Return':>9} {'Sharpe':>8} "
    header += f"{'Sharpe CI':>14} {'DD':>8} {'Alpha':>9} {'p-val':>7} {'Sig':>5}"
    print(header)
    print("─"*90)

    for r in sorted(results, key=lambda x: -x["rl"]["sharpe"]):
        ci_str = f"[{r['rl']['sharpe_ci'][0]:+.2f}, {r['rl']['sharpe_ci'][1]:+.2f}]"
        sig = "✓" if r["significant"] else "✗"
        print(
            f"{r['symbol']:<14} {r['asset_class']:<10} {r['test_steps']:>6} "
            f"{r['rl']['total_return']:>+9.1%} {r['rl']['sharpe']:>+8.3f} "
            f"{ci_str:>14} {r['rl']['max_drawdown']:>8.1%} "
            f"{r['rl']['information_ratio']:>+9.3f} "
            f"{r['permutation_p']:>7.3f} {sig:>5}"
        )

    print("─"*90)
    sharpes = [r["rl"]["sharpe"] for r in results]
    sig_count = sum(r["significant"] for r in results)
    pos_count = sum(r["rl"]["sharpe"] > 0 for r in results)
    print(f"\nMean Sharpe : {np.mean(sharpes):+.3f} ± {np.std(sharpes):.3f}")
    print(f"Sharpe > 0  : {pos_count}/{len(results)}")
    print(f"p < 0.05    : {sig_count}/{len(results)}")
    print("═"*90)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC-USD",
                        help="Asset à évaluer (ex: BTC-USD, SPY, GC=F, EURUSD=X)")
    parser.add_argument("--cross-market", action="store_true",
                        help="Mode cross-market : normalise avec les stats de l'asset cible")
    parser.add_argument("--cross-market-all", action="store_true",
                        help="Évalue tous les assets du registry")
    parser.add_argument("--model-path", default="models/ppo_trading_continuous")
    parser.add_argument("--vecnorm-path", default="models/vecnormalize.pkl")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    # Charge la config des features du modèle entraîné
    try:
        feat_cfg = load_feature_config("models/feature_config.json")
        print(f"[Eval] Feature config: {feat_cfg['n_features']} features | "
              f"cross-assets: {feat_cfg.get('cross_asset_symbols', [])}")
    except FileNotFoundError:
        print("[Eval] Pas de feature_config.json — config par défaut")
        feat_cfg = {"n_features": None, "feature_names": [],
                    "cross_asset_symbols": [], "include_calendar": True, "is_crypto": True}

    if args.cross_market_all:
        # Évalue tous les assets du registry
        symbols = list(REGISTRY.keys())
        print(f"\n[Eval] Cross-market eval sur {len(symbols)} assets : {symbols}")
        all_results = []
        for sym in symbols:
            try:
                plot_path = f"models/eval_{sym.replace('=','_').replace('^','')}.png" \
                            if not args.no_plot else None
                r = evaluate_asset(
                    symbol=sym,
                    model_path=args.model_path,
                    vecnorm_path=args.vecnorm_path,
                    feat_cfg=feat_cfg,
                    cross_market=(sym != "BTC-USD"),
                    save_plot=plot_path,
                    verbose=True,
                )
                all_results.append(r)
            except Exception as e:
                print(f"[Eval] ⚠ Erreur sur {sym}: {e}")

        print_cross_market_report(all_results)
        with open("models/cross_market_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("\n[Eval] Résultats sauvegardés: models/cross_market_results.json")

    else:
        # Évalue un seul asset
        cross_market = args.cross_market or (args.symbol != "BTC-USD")
        plot_path = f"models/eval_{args.symbol.replace('=','_').replace('^','')}.png" \
                    if not args.no_plot else None
        r = evaluate_asset(
            symbol=args.symbol,
            model_path=args.model_path,
            vecnorm_path=args.vecnorm_path,
            feat_cfg=feat_cfg,
            cross_market=cross_market,
            save_plot=plot_path,
            verbose=True,
        )
        with open("models/eval_results.json", "w") as f:
            json.dump(r, f, indent=2)
        print("[Eval] Results saved: models/eval_results.json")


if __name__ == "__main__":
    main()