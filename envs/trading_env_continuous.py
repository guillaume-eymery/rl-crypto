from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TradingEnvContinuous(gym.Env):
    """
    1 asset, action = target fraction in [-1, 1] (si allow_short=True) ou [0, 1].
    Execution : instant rebalance avec fees.

    Observation:
      [window × F features] + [pos_frac, cash_frac, running_sharpe_z, dd_ratio]

    Reward: Sharpe-adjusted return
      r_t = Δequity/equity_prev  (return step)
      reward = r_t / (running_vol + eps)  — risk-adjusted
      avec pénalité sur le turnover et sur le changement de position.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,           # (T, F) standardisé
        prices: np.ndarray,             # (T,) prix close
        window: int = 50,
        fee: float = 0.0005,            # 5 bps / notional tradé
        initial_cash: float = 10_000.0,
        allow_short: bool = False,      # True → action ∈ [-1, 1]
        reward_type: str = "sharpe",    # "sharpe" | "return" | "sortino"
        sharpe_window: int = 20,        # fenêtre rolling pour Sharpe reward
        turnover_penalty: float = 0.001, # pénalité sur le turnover (décourage overtrading)
        action_smooth_penalty: float = 0.0005,  # pénalité changement d'action
        max_drawdown_kill: float = 0.25, # termine l'épisode si DD > 25%
        randomize_start: bool = True,   # start aléatoire dans la série (anti-overfit)
        min_episode_steps: int = 252,   # min steps par épisode si randomize_start
    ):
        super().__init__()
        assert len(features) == len(prices)
        assert window >= 2
        assert reward_type in ("sharpe", "return", "sortino")

        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float32)
        self.window = window
        self.fee = float(fee)
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

        # Observation: window * F + [pos_frac, cash_frac, sharpe_z, dd_ratio]
        obs_dim = self.window * self.F + 4
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

    # ── State ─────────────────────────────────────────────────────────────

    def _reset_state(self, seed):
        # Randomized start
        if self.randomize_start and seed is not None:
            rng = np.random.default_rng(seed)
            max_start = max(self.window, self.T - self.min_episode_steps - 1)
            self.episode_start = int(rng.integers(self.window, max(self.window + 1, max_start)))
        else:
            self.episode_start = self.window

        self.t = self.episode_start
        self.cash = self.initial_cash
        self.shares = 0.0
        self.prev_equity = self.initial_cash
        self.peak_equity = self.initial_cash
        self.prev_action = 0.0

        # Buffers pour Sharpe rolling
        self._step_returns: list[float] = []

    def _equity(self) -> float:
        return float(self.cash + self.shares * float(self.prices[self.t]))

    def _pos_fraction(self) -> float:
        eq = self._equity()
        if eq <= 1e-12:
            return 0.0
        pos_value = self.shares * float(self.prices[self.t])
        lo = -1.0 if self.allow_short else 0.0
        return float(np.clip(pos_value / eq, lo, 1.0))

    def _cash_fraction(self) -> float:
        eq = self._equity()
        if eq <= 1e-12:
            return 1.0
        return float(np.clip(self.cash / eq, 0.0, 1.0))

    def _running_sharpe(self) -> float:
        """Sharpe annualisé sur la fenêtre rolling des returns."""
        w = self._step_returns[-self.sharpe_window:]
        if len(w) < 5:
            return 0.0
        arr = np.array(w)
        mu = arr.mean()
        sigma = arr.std() + 1e-9
        return float(mu / sigma * np.sqrt(252))

    def _drawdown_ratio(self) -> float:
        eq = self._equity()
        dd = (self.peak_equity - eq) / (self.peak_equity + 1e-9)
        return float(np.clip(dd, 0.0, 1.0))

    def _get_obs(self) -> np.ndarray:
        w = self.features[self.t - self.window : self.t].reshape(-1)
        sharpe_z = np.clip(self._running_sharpe() / 3.0, -1.0, 1.0)  # normalize ~[-1,1]
        extras = np.array([
            self._pos_fraction(),
            self._cash_fraction(),
            sharpe_z,
            self._drawdown_ratio(),
        ], dtype=np.float32)
        return np.concatenate([w, extras]).astype(np.float32)

    # ── Gymnasium API ─────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state(seed=seed if seed is not None else np.random.randint(0, 1_000_000))
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        lo = -1.0 if self.allow_short else 0.0
        target_frac = float(np.clip(action[0], lo, 1.0))

        price = float(self.prices[self.t])
        equity_before = self._equity()

        # Current position value (peut être négatif si short)
        pos_value = self.shares * price
        current_frac = 0.0 if equity_before <= 1e-12 else pos_value / equity_before

        target_pos_value = target_frac * equity_before
        delta_value = target_pos_value - pos_value
        traded_notional = abs(delta_value)
        fee_paid = self.fee * traded_notional

        # Execute
        if traded_notional > 1e-9:
            if delta_value > 0:
                cost = delta_value + fee_paid
                if cost > self.cash:
                    scale = self.cash / max(cost, 1e-12)
                    delta_value *= scale
                    traded_notional = abs(delta_value)
                    fee_paid = self.fee * traded_notional
                    cost = delta_value + fee_paid
                self.cash -= cost
                self.shares += delta_value / price
            else:
                # Sell / short
                proceeds = (-delta_value) - fee_paid
                if not self.allow_short:
                    # Clamp par les shares disponibles
                    max_sell_value = self.shares * price
                    if (-delta_value) > max_sell_value:
                        delta_value = -max_sell_value
                        traded_notional = abs(delta_value)
                        fee_paid = self.fee * traded_notional
                        proceeds = (-delta_value) - fee_paid
                self.shares += delta_value / price
                self.cash += max(proceeds, 0.0)

        # Advance time
        self.t += 1
        equity_after = self._equity()

        # Peak tracking pour drawdown
        if equity_after > self.peak_equity:
            self.peak_equity = equity_after

        # Step return
        step_ret = (equity_after - self.prev_equity) / max(self.prev_equity, 1e-12)
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

        # Kill-switch drawdown
        dd = self._drawdown_ratio()
        if dd >= self.max_drawdown_kill:
            terminated = True
            reward -= 1.0  # pénalité pour avoir atteint le kill-switch

        truncated = False

        info = {
            "equity": equity_after,
            "price": float(self.prices[self.t] if self.t < self.T else self.prices[-1]),
            "target_frac": target_frac,
            "pos_frac": self._pos_fraction(),
            "cash_frac": self._cash_fraction(),
            "fee_paid": float(fee_paid),
            "turnover": float(traded_notional / max(equity_before, 1e-12)),
            "step_return": step_ret,
            "drawdown": dd,
            "running_sharpe": self._running_sharpe(),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _compute_reward(
        self,
        step_ret: float,
        traded_notional: float,
        equity_before: float,
        action_change: float,
    ) -> float:
        w = self._step_returns[-self.sharpe_window:]

        if self.reward_type == "sharpe":
            if len(w) < 5:
                base_reward = step_ret
            else:
                arr = np.array(w)
                vol = arr.std() + 1e-9
                base_reward = step_ret / vol
        elif self.reward_type == "sortino":
            if len(w) < 5:
                base_reward = step_ret
            else:
                arr = np.array(w)
                downside = arr[arr < 0]
                downside_vol = downside.std() if len(downside) > 1 else 1e-9
                base_reward = step_ret / (downside_vol + 1e-9)
        else:  # "return"
            base_reward = step_ret

        # Pénalités
        turnover_penalty = self.turnover_penalty * (traded_notional / max(equity_before, 1e-12))
        smooth_penalty = self.action_smooth_penalty * action_change

        return float(base_reward - turnover_penalty - smooth_penalty)
