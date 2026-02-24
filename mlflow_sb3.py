"""
mlflow_sb3.py — MLflow callback pour SB3.
Logs: épisodes, métriques trading, Sharpe rolling, checkpoints.
"""

import os
import time
from typing import Optional

import mlflow
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MLflowTradingCallback(BaseCallback):
    """
    Logs to MLflow:
    - SB3 episode metrics via Monitor (reward, length)
    - Trading metrics from info dict (equity, turnover, fees, positions)
    - Rolling Sharpe approximation
    - Model checkpoints as artifacts
    """

    def __init__(
        self,
        experiment_name: str = "trading-rl",
        run_name: Optional[str] = None,
        tracking_uri: str = "file:./mlruns",
        log_every_steps: int = 2000,
        checkpoint_every_steps: int = 100_000,
        checkpoint_dir: str = "./models",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{int(time.time())}"
        self.tracking_uri = tracking_uri
        self.log_every_steps = log_every_steps
        self.checkpoint_every_steps = checkpoint_every_steps
        self.checkpoint_dir = checkpoint_dir

        self._last_log_step = 0
        self._last_ckpt_step = 0

        self._equity_buf = []
        self._turnover_buf = []
        self._fees_buf = []
        self._pos_buf = []
        self._ep_rew_buf = []
        self._ep_len_buf = []
        self._step_ret_buf = []  # pour Sharpe rolling

    def _on_training_start(self) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)

        mlflow.log_param("algo", self.model.__class__.__name__)
        mlflow.log_param("policy", self.model.policy.__class__.__name__)
        for k in ["n_steps", "batch_size", "gamma", "learning_rate", "ent_coef", "clip_range", "n_epochs"]:
            if hasattr(self.model, k):
                v = getattr(self.model, k)
                mlflow.log_param(k, v if isinstance(v, (int, float, str, bool)) else repr(v))

    def _flush_metrics(self) -> None:
        step = int(self.num_timesteps)

        def _mean(x):
            return float(np.mean(x)) if len(x) else float("nan")

        if self._ep_rew_buf:
            mlflow.log_metric("rollout/ep_rew_mean", _mean(self._ep_rew_buf), step=step)
            mlflow.log_metric("rollout/ep_len_mean", _mean(self._ep_len_buf), step=step)
            self._ep_rew_buf.clear()
            self._ep_len_buf.clear()

        if self._equity_buf:
            mlflow.log_metric("trading/equity_mean", _mean(self._equity_buf), step=step)
            mlflow.log_metric("trading/equity_last", float(self._equity_buf[-1]), step=step)
            mlflow.log_metric("trading/turnover_mean", _mean(self._turnover_buf), step=step)
            mlflow.log_metric("trading/fee_paid_mean", _mean(self._fees_buf), step=step)
            mlflow.log_metric("trading/pos_frac_mean", _mean(self._pos_buf), step=step)

            self._equity_buf.clear()
            self._turnover_buf.clear()
            self._fees_buf.clear()
            self._pos_buf.clear()

        # Rolling Sharpe approximation
        if len(self._step_ret_buf) >= 20:
            arr = np.array(self._step_ret_buf[-252:])
            rolling_sharpe = float(arr.mean() / (arr.std() + 1e-9) * np.sqrt(252))
            mlflow.log_metric("trading/rolling_sharpe", rolling_sharpe, step=step)

    def _maybe_checkpoint(self) -> None:
        if self.num_timesteps - self._last_ckpt_step >= self.checkpoint_every_steps:
            path = os.path.join(self.checkpoint_dir, f"model_step_{self.num_timesteps}.zip")
            self.model.save(path)
            try:
                mlflow.log_artifact(path, artifact_path="checkpoints")
            except Exception:
                pass
            self._last_ckpt_step = self.num_timesteps

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos:
            info0 = infos[0]

            ep = info0.get("episode")
            if ep is not None:
                self._ep_rew_buf.append(ep.get("r", 0.0))
                self._ep_len_buf.append(ep.get("l", 0.0))

            if "equity" in info0:
                self._equity_buf.append(info0["equity"])
            if "turnover" in info0:
                self._turnover_buf.append(info0["turnover"])
            if "fee_paid" in info0:
                self._fees_buf.append(info0["fee_paid"])
            if "pos_frac" in info0:
                self._pos_buf.append(info0["pos_frac"])
            if "step_return" in info0:
                self._step_ret_buf.append(info0["step_return"])

        if self.num_timesteps - self._last_log_step >= self.log_every_steps:
            mlflow.log_metric("time/total_timesteps", int(self.num_timesteps), step=int(self.num_timesteps))
            self._flush_metrics()
            self._maybe_checkpoint()
            self._last_log_step = int(self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        self._flush_metrics()
        try:
            final_path = os.path.join(self.checkpoint_dir, "model_final.zip")
            self.model.save(final_path)
            mlflow.log_artifact(final_path, artifact_path="final")
        except Exception:
            pass
        mlflow.end_run()
