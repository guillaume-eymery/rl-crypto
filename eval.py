"""
eval.py — Évaluation complète pour RecurrentPPO + LSTM.
Compatible aussi avec l'ancien modèle PPO (fallback automatique).
"""

from __future__ import annotations
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from features.make_features import (
    load_data, make_features, standardize_train_only,
    load_feature_config,
)
from envs.trading_env_lstm import TradingEnvLSTM
from metrics import compute_metrics, permutation_test


def load_model(model_path, vec_env):
    """Charge RecurrentPPO ou PPO selon ce qui est disponible."""
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


def run_episode(model, vec_env, model_type="lstm", deterministic=True):
    obs = vec_env.reset()
    done = False
    equities, positions, fees, turnovers = [], [], [], []
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    while not done:
        if model_type == "lstm":
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=deterministic
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


def run_random_agent(X_test, p_test, cfg, n_episodes=10):
    all_equities = []
    for _ in range(n_episodes):
        env = TradingEnvLSTM(X_test, p_test, fee=cfg["fee"],
                              allow_short=cfg.get("allow_short", False),
                              randomize_start=False)
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


def plot_full_analysis(rl_equities, bh_equities, positions, fees, turnovers,
                       rl_metrics, bh_metrics, perm_result, symbol, save_path=None):
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    n = len(rl_equities)
    x = np.arange(n)
    rl_ret = np.diff(rl_equities) / np.maximum(rl_equities[:-1], 1e-12)
    bh_eq_n = bh_equities[:n]
    bh_ret = np.diff(bh_eq_n) / np.maximum(bh_eq_n[:-1], 1e-12)

    # 1. Equity curves
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(rl_equities, label=f"RL LSTM (Sharpe={rl_metrics.sharpe:.2f})", lw=1.5, color="#2196F3")
    ax1.plot(bh_eq_n, label=f"B&H (Sharpe={bh_metrics.sharpe:.2f})", lw=1.5, color="#FF5722", alpha=0.8)
    ax1.set_title(f"{symbol} — Equity Curves (Test Set)", fontweight="bold")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(); ax1.grid(alpha=0.3)

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, :2])
    rm_rl = np.maximum.accumulate(rl_equities)
    dd_rl = (rl_equities - rm_rl) / np.maximum(rm_rl, 1e-12)
    rm_bh = np.maximum.accumulate(bh_eq_n)
    dd_bh = (bh_eq_n - rm_bh) / np.maximum(rm_bh, 1e-12)
    ax2.fill_between(x, dd_rl * 100, alpha=0.5, color="#2196F3", label=f"RL DD (max={rl_metrics.max_drawdown:.1%})")
    ax2.fill_between(x, dd_bh * 100, alpha=0.3, color="#FF5722", label=f"B&H DD (max={bh_metrics.max_drawdown:.1%})")
    ax2.set_title("Drawdown (%)"); ax2.set_ylabel("Drawdown (%)")
    ax2.legend(); ax2.grid(alpha=0.3)

    # 3. Position heatmap — mieux que scatter pour visualiser les régimes
    ax3 = fig.add_subplot(gs[2, :2])
    pos_arr = np.array(positions)
    im = ax3.imshow(pos_arr.reshape(1, -1), aspect="auto", cmap="RdYlGn",
                    vmin=0, vmax=1, extent=[0, n, -0.5, 0.5])
    plt.colorbar(im, ax=ax3, label="Position fraction", orientation="horizontal", pad=0.3)
    ax3.set_yticks([])
    ax3.set_title(f"Position Through Time | Mean={pos_arr.mean():.2f} | "
                  f"Std={pos_arr.std():.2f} | % invested={( pos_arr>0.1).mean():.1%}")

    # 4. Metrics table
    ax4 = fig.add_subplot(gs[0, 2])
    ax4.axis("off")
    data = [
        ["Metric", "RL LSTM", "B&H"],
        ["Total Return", f"{rl_metrics.total_return:+.1%}", f"{bh_metrics.total_return:+.1%}"],
        ["CAGR", f"{rl_metrics.cagr:+.1%}", f"{bh_metrics.cagr:+.1%}"],
        ["Ann. Vol", f"{rl_metrics.ann_volatility:.1%}", f"{bh_metrics.ann_volatility:.1%}"],
        ["Sharpe", f"{rl_metrics.sharpe:.2f}", f"{bh_metrics.sharpe:.2f}"],
        ["Sortino", f"{rl_metrics.sortino:.2f}", f"{bh_metrics.sortino:.2f}"],
        ["Calmar", f"{rl_metrics.calmar:.2f}", f"{bh_metrics.calmar:.2f}"],
        ["Max DD", f"{rl_metrics.max_drawdown:.1%}", f"{bh_metrics.max_drawdown:.1%}"],
        ["DD Duration", f"{rl_metrics.max_drawdown_duration}d", f"{bh_metrics.max_drawdown_duration}d"],
        ["VaR 95%", f"{rl_metrics.var_95:.1%}", f"{bh_metrics.var_95:.1%}"],
        ["Win Rate", f"{rl_metrics.win_rate:.1%}", f"{bh_metrics.win_rate:.1%}"],
        ["Info Ratio", f"{rl_metrics.information_ratio:.2f}", "—"],
        ["Total Fees", f"${rl_metrics.total_fees:.0f}", "—"],
        ["Avg Turnover", f"{rl_metrics.avg_daily_turnover:.1%}", "—"],
    ]
    t = ax4.table(cellText=data[1:], colLabels=data[0], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.3)
    # Color RL column green/red based on vs B&H
    ax4.set_title("Performance Summary", fontweight="bold", pad=20)

    # 5. Return distribution
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(rl_ret * 100, bins=60, alpha=0.7, color="#2196F3", label="RL LSTM", density=True)
    ax5.hist(bh_ret * 100, bins=60, alpha=0.5, color="#FF5722", label="B&H", density=True)
    ax5.axvline(0, color="black", ls="--", lw=0.8)
    ax5.set_title(f"Return Dist | Skew={rl_metrics.skewness:+.1f} | Kurt={rl_metrics.kurtosis:.1f}")
    ax5.set_xlabel("Return (%)"); ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

    # 6. Rolling 60d Sharpe
    ax6 = fig.add_subplot(gs[2, 2])
    w = 60
    if len(rl_ret) > w:
        rolling_sharpe = [
            float(np.mean(rl_ret[max(0,i-w):i]) / (np.std(rl_ret[max(0,i-w):i]) + 1e-9) * np.sqrt(252))
            for i in range(w, len(rl_ret))
        ]
        ax6.plot(rolling_sharpe, color="#9C27B0", lw=1, label="60d rolling Sharpe")
        ax6.axhline(0, color="black", ls="--", lw=0.8)
        ax6.axhline(rl_metrics.sharpe, color="#2196F3", ls="--", lw=1, label=f"Full Sharpe={rl_metrics.sharpe:.2f}")
        ax6.fill_between(range(len(rolling_sharpe)),
                         [s if s > 0 else 0 for s in rolling_sharpe],
                         alpha=0.2, color="green")
        ax6.fill_between(range(len(rolling_sharpe)),
                         [s if s < 0 else 0 for s in rolling_sharpe],
                         alpha=0.2, color="red")
    ax6.set_title("Rolling 60d Sharpe"); ax6.legend(fontsize=8); ax6.grid(alpha=0.3)

    # 7. Statistical significance
    ax7 = fig.add_subplot(gs[3, :])
    sig = "✓ p<5%" if perm_result["significant_5pct"] else "✗ NOT significant"
    p1 = " (p<1%)" if perm_result["significant_1pct"] else ""
    text = (
        f"Permutation Test (H0: RL = B&H) | stat={perm_result['observed_stat']:.3f} | "
        f"p={perm_result['p_value']:.4f} | {sig}{p1}\n"
        f"Sharpe 95% CI: [{rl_metrics.sharpe_ci_low:.2f}, {rl_metrics.sharpe_ci_high:.2f}]  |  "
        f"Skewness: {rl_metrics.skewness:+.2f}  |  Kurtosis: {rl_metrics.kurtosis:.1f}  |  "
        f"Alpha (ann.): {rl_metrics.alpha:+.2%}  |  Beta: {rl_metrics.beta:.3f}"
    )
    ax7.text(0.5, 0.5, text, transform=ax7.transAxes, ha="center", va="center",
             fontsize=10, bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.8))
    ax7.axis("off")
    ax7.set_title("Statistical Significance & Risk Decomposition", fontweight="bold")

    fig.suptitle(f"RL Trading Analysis (LSTM) — {symbol}", fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Eval] Plot saved: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--model-path", default="models/ppo_trading_continuous")
    parser.add_argument("--vecnorm-path", default="models/vecnormalize.pkl")
    parser.add_argument("--save-plot", default="models/eval_analysis.png")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-cross-assets", action="store_true")
    args = parser.parse_args()

    cfg = {"fee": 0.0005, "allow_short": False, "reward_type": "return"}
    try:
        with open("models/best_config.json") as f:
            cfg.update(json.load(f))
    except FileNotFoundError:
        pass

    # Feature config
    feat_cfg_path = "models/feature_config.json"
    if args.no_cross_assets:
        feat_cfg = {"cross_asset_symbols": [], "include_calendar": True, "n_features": None}
        cross_asset_symbols = []
    else:
        try:
            feat_cfg = load_feature_config(feat_cfg_path)
            cross_asset_symbols = feat_cfg["cross_asset_symbols"]
            print(f"[Eval] Feature config: {feat_cfg['n_features']} features | cross-assets: {cross_asset_symbols}")
        except FileNotFoundError:
            feat_cfg = {"cross_asset_symbols": [], "include_calendar": True, "n_features": None}
            cross_asset_symbols = []
            print("[Eval] No feature_config.json — no cross-assets")

    # Data
    # Exclure le primary symbol des cross-assets (cas out-of-time eval sur ETH, SOL, etc.)
    # Si on évalue sur ETH alors qu'on a entraîné sur BTC avec ETH comme cross-asset,
    # on retire ETH de la liste cross pour éviter autocorrélation triviale.
    cross_asset_symbols = [s for s in cross_asset_symbols if s != args.symbol]
    if len(cross_asset_symbols) != feat_cfg.get("n_features", 99):
        pass  # on recalcule les features sans le primary dans les cross-assets
    symbols_to_load = [args.symbol] + cross_asset_symbols
    all_data = load_data(symbols_to_load, start=args.start)
    df = all_data[args.symbol]
    cross_asset_closes = {sym: all_data[sym]["Close"].squeeze()
                          for sym in cross_asset_symbols if sym in all_data}

    X, prices, idx = make_features(df,
                                    cross_asset_volumes={sym: all_data[sym]["Volume"].squeeze()
                                        for sym in cross_asset_symbols if sym in all_data} or None,
                                    cross_asset_closes=cross_asset_closes or None,
                                    include_calendar=feat_cfg.get("include_calendar", True),
                                    is_crypto=feat_cfg.get("is_crypto", True))

    # Alignement des features pour l'éval out-of-time (ex: BTC→ETH)
    # Si des cross-asset features manquent (primary retiré de la liste cross),
    # on les remplace par des colonnes de zéros pour maintenir la compatibilité
    # avec le VecNormalize entraîné qui attend n_features fixes.
    n_exp = feat_cfg.get("n_features")
    expected_names = feat_cfg.get("feature_names", [])
    current_names = make_features._last_feature_names
    if n_exp and X.shape[1] != n_exp and expected_names:
        missing = [f for f in expected_names if f not in current_names]
        extra   = [f for f in current_names if f not in expected_names]
        if missing:
            print(f"[Eval] Out-of-time mode: padding {len(missing)} missing features with zeros: {missing}")
            # Construire la matrice finale dans l'ordre exact du training
            X_aligned = np.zeros((X.shape[0], n_exp), dtype=np.float32)
            for j, fname in enumerate(expected_names):
                if fname in current_names:
                    X_aligned[:, j] = X[:, current_names.index(fname)]
                # sinon reste à 0.0
            X = X_aligned
        if X.shape[1] != n_exp:
            raise ValueError(f"Feature mismatch after alignment: got {X.shape[1]}, expected {n_exp}.")
    elif n_exp and X.shape[1] != n_exp:
        raise ValueError(f"Feature mismatch: got {X.shape[1]}, expected {n_exp}. "
                         f"Re-train or use --no-cross-assets.")

    print(f"[Eval] Features: {X.shape} | {idx[0].date()} → {idx[-1].date()}")

    split = int(0.8 * len(prices))
    X_train, p_train = X[:split], prices[:split]
    X_test, p_test = X[split:], prices[split:]
    _, X_test_s, _, _ = standardize_train_only(X_train, X_test)
    print(f"[Eval] Test set: {len(X_test)} steps ({idx[split].date()} → {idx[-1].date()})")

    # Build test env
    def _make_test():
        return TradingEnvLSTM(X_test_s, p_test, fee=cfg["fee"], funding_cost_daily=cfg.get("funding_cost_daily", 0.0002),
                               allow_short=cfg.get("allow_short", False),
                               reward_type="return", randomize_start=False)

    vec_env = DummyVecEnv([_make_test])
    vec_env = VecNormalize.load(args.vecnorm_path, vec_env)
    vec_env.training = False; vec_env.norm_reward = False

    model, model_type = load_model(args.model_path, vec_env)

    # Run episode
    ep = run_episode(model, vec_env, model_type=model_type)
    rl_equities = ep["equities"]
    positions = ep["positions"]
    fees_arr = ep["fees"]
    turnovers_arr = ep["turnovers"]

    # B&H benchmark
    n = len(rl_equities)
    bh_equities = 10_000.0 * p_test[:n] / float(p_test[0])

    # Metrics
    bh_n = bh_equities[:n]
    rl_metrics = compute_metrics(rl_equities, benchmark_equity=bh_n, trading_days_per_year=365,
                                  fees_per_step=fees_arr, turnover_per_step=turnovers_arr)
    bh_metrics = compute_metrics(bh_equities[:n], trading_days_per_year=365)

    print("\n══════════════════════════════════════════════════")
    print("  RL STRATEGY (LSTM)")
    print(rl_metrics)
    print("\n  BUY & HOLD")
    print(bh_metrics)
    print("══════════════════════════════════════════════════")

    # Random baseline
    random_equities = run_random_agent(X_test_s, p_test, cfg, n_episodes=10)
    random_sharpes = [compute_metrics(e).sharpe for e in random_equities]
    print(f"\n[Eval] Random agent Sharpe: {np.mean(random_sharpes):.3f} ± {np.std(random_sharpes):.3f}")
    print(f"[Eval] RL beats random: {rl_metrics.sharpe > np.mean(random_sharpes)}")

    # Permutation test
    rl_ret = np.diff(rl_equities) / rl_equities[:-1]
    bh_ret = np.diff(bh_n) / np.maximum(bh_n[:-1], 1e-12)
    perm = permutation_test(rl_ret, bh_ret, n_permutations=10_000)
    print(f"\n[Permutation Test] p-value: {perm['p_value']:.4f} | Significant: {perm['significant_5pct']}")

    # Save results
    results = {
        "model_type": model_type,
        "rl": {"total_return": rl_metrics.total_return, "sharpe": rl_metrics.sharpe,
               "sharpe_ci": [rl_metrics.sharpe_ci_low, rl_metrics.sharpe_ci_high],
               "sortino": rl_metrics.sortino, "calmar": rl_metrics.calmar,
               "max_drawdown": rl_metrics.max_drawdown, "information_ratio": rl_metrics.information_ratio,
               "total_fees": rl_metrics.total_fees, "avg_turnover": rl_metrics.avg_daily_turnover},
        "bh": {"total_return": bh_metrics.total_return, "sharpe": bh_metrics.sharpe,
               "calmar": bh_metrics.calmar, "max_drawdown": bh_metrics.max_drawdown},
        "random_sharpe_mean": float(np.mean(random_sharpes)),
        "permutation_test": perm,
    }
    with open("models/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[Eval] Results saved: models/eval_results.json")

    if not args.no_plot:
        plot_full_analysis(rl_equities, bh_equities, positions, fees_arr, turnovers_arr,
                           rl_metrics, bh_metrics, perm, args.symbol, args.save_plot)


if __name__ == "__main__":
    main()