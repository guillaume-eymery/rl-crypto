"""
train.py — RecurrentPPO + LSTM pour crypto daily.

Univers : BTC-USD (primary) + ETH-USD, SOL-USD, BNB-USD, ADA-USD (cross-assets)
Data     : yfinance depuis 2017 (plusieurs cycles bull/bear)
Short    : activé par défaut (action ∈ [-1, 1])
Fees     : 10bps par trade (réaliste crypto spot)
"""

import os
import json
import argparse
import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from features.make_features import (
    load_data, make_features, standardize_train_only,
    walk_forward_splits, save_feature_config,
)
from envs.trading_env_lstm import TradingEnvLSTM
from metrics import compute_metrics, walk_forward_consistency
from mlflow_sb3 import MLflowTradingCallback


DEFAULT_CONFIG = {
    # ── Univers crypto ────────────────────────────────────────────────────
    "symbols": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD"],
    "primary_symbol": "BTC-USD",
    "start": "2017-01-01",
    "is_crypto": True,

    # ── Env ───────────────────────────────────────────────────────────────
    "fee": 0.001,                    # 10 bps crypto
    "funding_cost_daily": 0.0002,    # 2 bps/jour sur shorts
    "allow_short": True,             # Short activé
    "reward_type": "sortino",        # Mieux pour queues lourdes crypto
    "sharpe_window": 60,
    "turnover_penalty": 0.005,
    "action_smooth_penalty": 0.003,
    "max_drawdown_kill": 0.40,
    "min_episode_steps": 365,

    # ── Training ──────────────────────────────────────────────────────────
    "total_timesteps": 1_000_000,
    "n_splits_wf": 5,
    "test_frac_wf": 0.10,
    "gap_wf": 5,

    # ── Optuna ───────────────────────────────────────────────────────────
    "use_optuna": True,
    "optuna_trials": 25,
    "optuna_timesteps": 150_000,

    # ── RecurrentPPO ─────────────────────────────────────────────────────
    "n_steps": 512,
    "batch_size": 64,
    "gamma": 0.995,
    "learning_rate": 1e-4,
    "ent_coef": 0.01,
    "clip_range": 0.2,
    "n_epochs": 10,
    "gae_lambda": 0.95,
    "lstm_hidden_size": 256,
    "n_lstm_layers": 1,
    "shared_lstm": False,
    "enable_critic_lstm": True,
    "net_arch": [dict(pi=[64], vf=[64])],
}


def make_env_fn(X, prices, cfg, monitor=True, randomize_start=True):
    def _fn():
        env = TradingEnvLSTM(
            features=X, prices=prices,
            fee=cfg["fee"],
            funding_cost_daily=cfg.get("funding_cost_daily", 0.0002),
            allow_short=cfg["allow_short"],
            reward_type=cfg["reward_type"],
            sharpe_window=cfg.get("sharpe_window", 60),
            turnover_penalty=cfg["turnover_penalty"],
            action_smooth_penalty=cfg.get("action_smooth_penalty", 0.003),
            max_drawdown_kill=cfg["max_drawdown_kill"],
            randomize_start=randomize_start,
            min_episode_steps=cfg.get("min_episode_steps", 365),
        )
        if monitor:
            env = Monitor(env)
        return env
    return _fn


def build_vec_env(X, prices, cfg, randomize_start=True):
    fns = [make_env_fn(X, prices, cfg, monitor=True, randomize_start=randomize_start)]
    vec_env = DummyVecEnv(fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec_env


def build_rppo(vec_env, cfg):
    policy_kwargs = {
        "lstm_hidden_size": cfg.get("lstm_hidden_size", 256),
        "n_lstm_layers": cfg.get("n_lstm_layers", 1),
        "shared_lstm": cfg.get("shared_lstm", False),
        "enable_critic_lstm": cfg.get("enable_critic_lstm", True),
        "net_arch": cfg.get("net_arch", [dict(pi=[64], vf=[64])]),
    }
    return RecurrentPPO(
        "MlpLstmPolicy", vec_env,
        verbose=0,
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        gamma=cfg["gamma"],
        learning_rate=cfg["learning_rate"],
        ent_coef=cfg["ent_coef"],
        clip_range=cfg["clip_range"],
        n_epochs=cfg["n_epochs"],
        gae_lambda=cfg["gae_lambda"],
        policy_kwargs=policy_kwargs,
    )


def run_lstm_episode(model, vec, deterministic=True):
    obs = vec.reset()
    done = False
    equities, fees, turnovers, positions = [10_000.0], [], [], []
    lstm_states = None
    ep_starts = np.ones((1,), dtype=bool)
    while not done:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=ep_starts, deterministic=deterministic
        )
        obs, _, dones, infos = vec.step(action)
        done = bool(dones[0])
        ep_starts = dones
        equities.append(infos[0]["equity"])
        fees.append(infos[0].get("fee_paid", 0.0))
        turnovers.append(infos[0].get("turnover", 0.0))
        positions.append(infos[0].get("pos_frac", 0.0))
    return np.array(equities), np.array(fees), np.array(turnovers), np.array(positions)


def optuna_search(X_train, p_train, X_test, p_test, cfg, n_trials=25):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("pip install optuna"); return cfg

    import tempfile

    def objective(trial):
        tc = {
            **cfg,
            "n_steps": trial.suggest_categorical("n_steps", [256, 512, 1024]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "gamma": trial.suggest_float("gamma", 0.97, 0.999, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True),
            "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.02),
            "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
            "n_epochs": trial.suggest_categorical("n_epochs", [5, 10]),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.90, 0.99),
            "lstm_hidden_size": trial.suggest_categorical("lstm_hidden_size", [64, 128, 256]),
            "n_lstm_layers": trial.suggest_categorical("n_lstm_layers", [1, 2]),
            "turnover_penalty": trial.suggest_float("turnover_penalty", 0.001, 0.01, log=True),
            "action_smooth_penalty": trial.suggest_float("action_smooth_penalty", 0.001, 0.01, log=True),
            "net_arch": trial.suggest_categorical("net_arch", [
                [dict(pi=[64], vf=[64])],
                [dict(pi=[128, 64], vf=[128, 64])],
            ]),
        }
        if tc["n_steps"] < tc["batch_size"]: return -999.0
        try:
            ve = build_vec_env(X_train, p_train, tc)
            m = build_rppo(ve, tc)
            m.learn(total_timesteps=cfg["optuna_timesteps"])
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                tmp = f.name
            ve.save(tmp)
            def _mk():
                return TradingEnvLSTM(X_test, p_test, fee=cfg["fee"],
                                      allow_short=cfg["allow_short"],
                                      reward_type="return", randomize_start=False)
            v2 = DummyVecEnv([_mk])
            v2 = VecNormalize.load(tmp, v2)
            v2.training = False; v2.norm_reward = False
            eq, _, _, _ = run_lstm_episode(m, v2)
            os.unlink(tmp); ve.close(); v2.close()
            return compute_metrics(eq).sharpe
        except Exception as e:
            print(f"  Trial failed: {e}"); return -999.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\n[Optuna] Best Sharpe: {study.best_value:.3f}")
    print(f"[Optuna] Best params:\n{json.dumps(study.best_params, indent=2)}")
    return {**cfg, **study.best_params}


def walk_forward_eval(X, prices, cfg, model_path):
    splits = walk_forward_splits(len(prices), n_splits=cfg["n_splits_wf"],
                                  test_frac=cfg["test_frac_wf"], gap=cfg["gap_wf"])
    fold_metrics = []
    for i, (tr_idx, te_idx) in enumerate(splits):
        _, X_te_s, _, _ = standardize_train_only(X[tr_idx], X[te_idx])
        p_te = prices[te_idx]

        def _mk():
            return TradingEnvLSTM(X_te_s, p_te, fee=cfg["fee"],
                                  allow_short=cfg["allow_short"],
                                  reward_type="return", randomize_start=False)
        vec = DummyVecEnv([_mk])
        try:
            vec = VecNormalize.load("models/vecnormalize.pkl", vec)
            vec.training = False; vec.norm_reward = False
        except Exception:
            vec = VecNormalize(vec, norm_obs=True, norm_reward=False)
            vec.training = False

        model = RecurrentPPO.load(model_path, env=vec)
        eq, fees, to, _ = run_lstm_episode(model, vec)

        bh = 10_000.0 * p_te[:len(eq)] / float(p_te[0])
        m = compute_metrics(eq,
                            benchmark_equity=bh if len(bh) == len(eq) else None,
                            fees_per_step=fees, turnover_per_step=to,
                            trading_days_per_year=365)
        fold_metrics.append(m)
        print(f"\n[WF Fold {i+1}/{len(splits)}] "
              f"Sharpe={m.sharpe:.3f} | Calmar={m.calmar:.3f} | "
              f"DD={m.max_drawdown:.2%} | IR={m.information_ratio:.3f} | "
              f"Turnover={m.avg_daily_turnover:.2%}")

    consistency = walk_forward_consistency(fold_metrics)
    print(f"\n[WF Consistency]\n{json.dumps({k: round(v, 3) for k, v in consistency.items()}, indent=2)}")
    return fold_metrics, consistency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-optuna", action="store_true")
    parser.add_argument("--symbol", default="BTC-USD")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG,
           "primary_symbol": args.symbol,
           "total_timesteps": args.timesteps,
           "use_optuna": not args.no_optuna}

    print(f"[Train] Architecture : RecurrentPPO + MlpLstmPolicy | Short=ON")
    print(f"[Train] Universe     : {cfg['symbols']}")
    print(f"[Train] Primary      : {cfg['primary_symbol']} | Start: {cfg['start']}")

    # ── Data ──────────────────────────────────────────────────────────────
    all_data = load_data(cfg["symbols"], start=cfg["start"])
    primary_df = all_data[cfg["primary_symbol"]]
    cross_closes = {s: df["Close"].squeeze() for s, df in all_data.items()
                    if s != cfg["primary_symbol"]}
    cross_volumes = {s: df["Volume"].squeeze() for s, df in all_data.items()
                     if s != cfg["primary_symbol"]}

    X, prices, idx = make_features(
        primary_df,
        cross_asset_closes=cross_closes,
        cross_asset_volumes=cross_volumes,
        include_calendar=True,
        is_crypto=cfg["is_crypto"],
    )
    print(f"[Features] {X.shape} | {idx[0].date()} → {idx[-1].date()}")

    os.makedirs("models", exist_ok=True)
    save_feature_config(
        "models/feature_config.json", X.shape[1],
        make_features._last_feature_names,
        [s for s in cfg["symbols"] if s != cfg["primary_symbol"]],
        include_calendar=True, is_crypto=cfg["is_crypto"],
    )

    # ── Split ─────────────────────────────────────────────────────────────
    split = int(0.8 * len(prices))
    X_train_s, X_test_s, mean, std = standardize_train_only(X[:split], X[split:])
    print(f"[Split] Train: {split} ({idx[0].date()} → {idx[split-1].date()}) | "
          f"Test: {len(prices)-split} ({idx[split].date()} → {idx[-1].date()})")

    # ── Optuna ────────────────────────────────────────────────────────────
    if cfg["use_optuna"]:
        print(f"\n[Optuna] {cfg['optuna_trials']} trials × {cfg['optuna_timesteps']} steps...")
        cfg = optuna_search(X_train_s, prices[:split], X_test_s, prices[split:], cfg,
                            n_trials=cfg["optuna_trials"])
        safe = {k: (list(v) if isinstance(v, (list, tuple)) and not isinstance(v, str) else v)
                for k, v in cfg.items()}
        with open("models/best_config.json", "w") as f:
            json.dump(safe, f, indent=2)

    # ── Training final ────────────────────────────────────────────────────
    print(f"\n[Train] {cfg['total_timesteps']} timesteps | "
          f"LSTM {cfg.get('lstm_hidden_size', 256)}×{cfg.get('n_lstm_layers', 1)}")
    vec_env = build_vec_env(X_train_s, prices[:split], cfg)
    model = build_rppo(vec_env, cfg)

    cb = MLflowTradingCallback(
        experiment_name="trading-rl-crypto",
        run_name=f"RecurrentPPO_{cfg['primary_symbol'].replace('-','_')}",
        tracking_uri="file:./mlruns",
        log_every_steps=5000, checkpoint_every_steps=200_000,
    )
    model.learn(total_timesteps=cfg["total_timesteps"], callback=cb, progress_bar=True)
    model.save("models/ppo_trading_continuous")
    vec_env.save("models/vecnormalize.pkl")
    np.savez("models/feature_scaler.npz", mean=mean, std=std)
    print("\n[Train] Model saved.")

    # ── Walk-forward ──────────────────────────────────────────────────────
    print("\n[WF] Walk-forward out-of-sample evaluation...")
    fold_metrics, consistency = walk_forward_eval(X, prices, cfg, "models/ppo_trading_continuous")
    with open("models/wf_consistency.json", "w") as f:
        json.dump({k: round(v, 4) for k, v in consistency.items()}, f, indent=2)

    print(f"\n[Done] WF positive folds: {consistency['positive_folds_pct']:.0%} | "
          f"Mean Sharpe: {consistency['mean_sharpe']:.3f} ± {consistency['std_sharpe']:.3f}")


if __name__ == "__main__":
    main()