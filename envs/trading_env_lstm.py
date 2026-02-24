"""
trading_env_lstm.py — Environnement LSTM adapté crypto.

Changements vs version équités :
  - allow_short=True par défaut : position ∈ [-1, 1]
    -1 = 100% short, 0 = flat, +1 = 100% long
  - fee=0.001 (10bps, plus réaliste crypto vs 5bps équités)
  - Reward Sortino par défaut (mieux adapté queues lourdes crypto)
  - annual_days=365 pour les calculs de vol et Sharpe
  - Margin modèle simplifié pour le short :
    short position = emprunter l'asset, rembourser avec intérêt implicite
    (modélisé via un coût de portage = fee * |position| par step)
  - Kill-switch à 40% drawdown (crypto est plus volatile)
  - Info dict enrichi : unrealized_pnl, funding_cost
"""

from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TradingEnvLSTM(gym.Env):
    """
    Env crypto optimisé pour RecurrentPPO + MlpLstmPolicy.

    Observation à chaque step t :
      [F features standardisées du step courant]
      + [pos_frac, dd_ratio, vol_20d_norm, days_elapsed_norm]
    → obs_dim = F + 4

    Action : target_frac ∈ [-1, 1] si allow_short=True, sinon [0, 1]
      - Négatif = position short (emprunter et vendre)
      - 0 = flat (cash)
      - Positif = long

    Coûts :
      - fee sur le notional tradé à chaque rebalance (10bps)
      - funding_cost : coût de portage sur les positions short (2bps/jour)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        fee: float = 0.001,               # 10 bps — réaliste crypto spot/perp
        funding_cost_daily: float = 0.0002, # 2 bps/jour sur positions short
        initial_cash: float = 10_000.0,
        allow_short: bool = True,
        reward_type: str = "sortino",     # sortino > sharpe pour crypto (queues lourdes)
        sharpe_window: int = 60,
        turnover_penalty: float = 0.005,
        action_smooth_penalty: float = 0.003,
        max_drawdown_kill: float = 0.40,   # 40% : crypto est plus volatile
        randomize_start: bool = True,
        min_episode_steps: int = 365,      # 1 an minimum (crypto = 365j)
    ):
        super().__init__()
        assert len(features) == len(prices)
        assert reward_type in ("sharpe", "sortino", "return")

        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float32)
        self.fee = float(fee)
        self.funding_cost_daily = float(funding_cost_daily)
        self.initial_cash = float(initial_cash)
        self.allow_short = allow_short
        self.reward_type = reward_type
        self.sharpe_window = sharpe_window
        self.turnover_penalty = float(turnover_penalty)
        self.action_smooth_penalty = float(action_smooth_penalty)
        self.max_drawdown_kill = float(max_drawdown_kill)
        self.randomize_start = randomize_start
        self.min_episode_steps = min_episode_steps

        self.T, self.F = self.features.shape
        self.annual_days = 365  # crypto

        obs_dim = self.F + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        lo = -1.0 if allow_short else 0.0
        self.action_space = spaces.Box(
            low=np.array([lo], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._reset_state(seed=None)

    def _reset_state(self, seed):
        if self.randomize_start and seed is not None:
            rng = np.random.default_rng(seed)
            max_start = max(1, self.T - self.min_episode_steps - 1)
            self.episode_start = int(rng.integers(1, max(2, max_start)))
        else:
            self.episode_start = 1

        self.t = self.episode_start
        self.episode_length = 0
        self.cash = self.initial_cash
        self.shares = 0.0          # positif = long, négatif = short
        self.prev_equity = self.initial_cash
        self.peak_equity = self.initial_cash
        self.prev_action = 0.0
        self._step_returns: list[float] = []

    def _price(self) -> float:
        return float(self.prices[self.t])

    def _equity(self) -> float:
        """
        Equity nette.
        Long : cash + shares * price
        Short : cash - |shares| * price + collateral_implicite
        Simplifié : equity = cash + shares * price (shares négatif si short)
        """
        return float(self.cash + self.shares * self._price())

    def _pos_fraction(self) -> float:
        eq = self._equity()
        if abs(eq) < 1e-12:
            return 0.0
        lo = -1.0 if self.allow_short else 0.0
        return float(np.clip(self.shares * self._price() / eq, lo, 1.0))

    def _drawdown_ratio(self) -> float:
        eq = self._equity()
        dd = (self.peak_equity - eq) / (self.peak_equity + 1e-9)
        return float(np.clip(dd, 0.0, 1.0))

    def _vol_norm(self) -> float:
        if len(self._step_returns) < 2:
            return 0.0
        w = self._step_returns[-20:]
        vol = float(np.std(w) * np.sqrt(self.annual_days))
        return float(np.clip(vol / 2.0, 0.0, 1.0))  # 200% vol annualisée = 1.0 (crypto)

    def _get_obs(self) -> np.ndarray:
        feat = self.features[self.t]
        days_norm = float(self.episode_length) / max(self.min_episode_steps, 1)
        extras = np.array([
            self._pos_fraction(),
            self._drawdown_ratio(),
            self._vol_norm(),
            np.clip(days_norm, 0.0, 3.0),
        ], dtype=np.float32)
        return np.concatenate([feat, extras]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state(seed=seed if seed is not None else np.random.randint(0, 1_000_000))
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        lo = -1.0 if self.allow_short else 0.0
        target_frac = float(np.clip(action[0], lo, 1.0))

        price = self._price()
        equity_before = self._equity()

        # Position actuelle en valeur
        current_pos_value = self.shares * price
        target_pos_value = target_frac * equity_before

        delta_value = target_pos_value - current_pos_value
        traded_notional = abs(delta_value)
        fee_paid = self.fee * traded_notional

        # ── Exécution ─────────────────────────────────────────────────────
        if traded_notional > 1e-9:
            delta_shares = delta_value / price

            if delta_value > 0:
                # Achat (ou rachat de short)
                cost = delta_value + fee_paid
                if cost > self.cash:
                    # Pas assez de cash : scale down
                    scale = self.cash / max(cost, 1e-12)
                    delta_value *= scale
                    delta_shares = delta_value / price
                    traded_notional = abs(delta_value)
                    fee_paid = self.fee * traded_notional
                    cost = delta_value + fee_paid
                self.cash -= cost
            else:
                # Vente (ou ouverture de short)
                proceeds = abs(delta_value) - fee_paid
                self.cash += max(proceeds, 0.0)

            self.shares += delta_shares

        # ── Funding cost sur positions short ──────────────────────────────
        funding_cost = 0.0
        if self.shares < 0:
            funding_cost = self.funding_cost_daily * abs(self.shares * price)
            self.cash -= funding_cost

        # ── Advance time ──────────────────────────────────────────────────
        self.t += 1
        self.episode_length += 1
        equity_after = self._equity()

        if equity_after > self.peak_equity:
            self.peak_equity = equity_after

        step_ret = (equity_after - self.prev_equity) / max(abs(self.prev_equity), 1e-12)
        self._step_returns.append(step_ret)
        self.prev_equity = equity_after

        # ── Reward ────────────────────────────────────────────────────────
        reward = self._compute_reward(
            step_ret=step_ret,
            traded_notional=traded_notional,
            equity_before=equity_before,
            action_change=abs(target_frac - self.prev_action),
        )
        self.prev_action = target_frac

        # ── Termination ───────────────────────────────────────────────────
        terminated = self.t >= (self.T - 1)
        dd = self._drawdown_ratio()
        if dd >= self.max_drawdown_kill:
            terminated = True
            reward -= 2.0

        info = {
            "equity": equity_after,
            "price": float(self.prices[self.t] if self.t < self.T else self.prices[-1]),
            "target_frac": target_frac,
            "pos_frac": self._pos_fraction(),
            "cash_frac": float(np.clip(self.cash / max(abs(equity_after), 1e-12), -2.0, 2.0)),
            "fee_paid": float(fee_paid),
            "funding_cost": float(funding_cost),
            "turnover": float(traded_notional / max(abs(equity_before), 1e-12)),
            "step_return": step_ret,
            "drawdown": dd,
            "running_sharpe": self._running_metric(),
            "is_short": float(self.shares < 0),
        }
        return self._get_obs(), float(reward), terminated, False, info

    def _compute_reward(self, step_ret, traded_notional, equity_before, action_change):
        w = self._step_returns[-self.sharpe_window:]

        if self.reward_type == "sortino":
            if len(w) < 10:
                base = step_ret
            else:
                arr = np.array(w)
                neg = arr[arr < 0]
                dv = neg.std() + 1e-9 if len(neg) > 1 else arr.std() + 1e-9
                base = step_ret / dv
        elif self.reward_type == "sharpe":
            if len(w) < 10:
                base = step_ret
            else:
                arr = np.array(w)
                base = step_ret / (arr.std() + 1e-9)
        else:
            base = step_ret

        to = traded_notional / max(abs(equity_before), 1e-12)
        # Pénalité quadratique : décourage les très grosses rotations
        turnover_pen = self.turnover_penalty * to + 0.5 * self.turnover_penalty * (to ** 2)
        smooth_pen = self.action_smooth_penalty * action_change

        return float(base - turnover_pen - smooth_pen)

    def _running_metric(self) -> float:
        w = self._step_returns[-self.sharpe_window:]
        if len(w) < 5:
            return 0.0
        arr = np.array(w)
        return float(arr.mean() / (arr.std() + 1e-9) * np.sqrt(self.annual_days))