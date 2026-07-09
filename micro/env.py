"""
LOBReplayEnv — Custom Gymnasium environment for LOB replay.

Supports:
- Real LOBSTER data and synthetic data (auto-detection)
- Configurable reward functions: log-return, sharpe, pnl
- Market frictions: taker fees, probabilistic maker rebates
- Time-aligned pre-computed macro vectors from the LLM swarm
- Train/test windowing (start_frac/end_frac) and random episode starts
- Inventory penalty to discourage excessive accumulation
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os

# Macro observation layout: 4 consensus scalars + 128 pooled embedding dims
MACRO_SCALAR_DIM = 4
MACRO_EMBED_DIM = 128
MACRO_DIM = MACRO_SCALAR_DIM + MACRO_EMBED_DIM  # 132


def canonical_lob_columns(levels=10):
    """Canonical 40-column order shared with CVML's (4, levels) reshape:
    [bid_p1..N, ask_p1..N, bid_s1..N, ask_s1..N] — numeric level order."""
    rng = range(1, levels + 1)
    return (
        [f"bid_price_{i}" for i in rng]
        + [f"ask_price_{i}" for i in rng]
        + [f"bid_size_{i}" for i in rng]
        + [f"ask_size_{i}" for i in rng]
    )


class LOBReplayEnv(gym.Env):
    """
    A custom Gymnasium replay environment that iterates over historical LOB data.

    Args:
        data_path: Path to LOB CSV (40 price/size columns required)
        use_macro_vector: Whether to include the 132D macro vector in obs
        macro_vectors_path: Path to .npz with keys: timestamps (N,),
            scalars (N,4: direction, magnitude, confidence, agreement),
            embeddings (N,384). Vectors are aligned to LOB rows by timestamp:
            each row sees the most recent event at or before its own time.
        zero_macro: Ablation switch — keep the obs shape but feed zeros
        reward_type: 'log_return' | 'sharpe' | 'pnl' (default: 'log_return')
        inventory_penalty: Quadratic penalty coefficient for large positions
        inactivity_penalty: Per-step cost for consecutive holds (default 0 = off)
        initial_cash: Starting cash — also serves as the natural position limit
        max_steps: Max steps per episode (None = run to end of data window)
        start_frac / end_frac: Fraction of the dataset this env may replay
            (e.g. train 0.0–0.7, eval 0.7–1.0)
        random_start: Randomize the episode start row within the window
    """

    REWARD_TYPES = ("log_return", "sharpe", "pnl")

    def __init__(
        self,
        data_path,
        window_size=50,
        use_macro_vector=True,
        macro_vectors_path=None,
        zero_macro=False,
        reward_type="log_return",
        inventory_penalty=0.0001,
        inactivity_penalty=0.0,
        initial_cash=10000.0,
        max_inventory=50,
        invalid_action_penalty=0.01,
        max_steps=None,
        start_frac=0.0,
        end_frac=1.0,
        random_start=False,
    ):
        super().__init__()

        self.data_path = data_path
        self.window_size = window_size
        self.use_macro_vector = use_macro_vector
        self.zero_macro = zero_macro
        self.reward_type = reward_type
        self.inventory_penalty = inventory_penalty
        self.inactivity_penalty = inactivity_penalty
        self.initial_cash = initial_cash
        self.max_inventory = max_inventory
        self.invalid_action_penalty = invalid_action_penalty
        self.random_start = random_start

        assert reward_type in self.REWARD_TYPES, f"reward_type must be one of {self.REWARD_TYPES}"
        assert 0.0 <= start_frac < end_frac <= 1.0, "need 0 <= start_frac < end_frac <= 1"

        # ── Load Data ─────────────────────────────────────────────────────
        self.df = pd.read_csv(data_path)

        # Canonical order — must match CVML's (Batch, 4, levels, 1) reshape.
        self.feature_cols = canonical_lob_columns(levels=10)
        missing = [c for c in self.feature_cols if c not in self.df.columns]
        assert not missing, f"LOB data missing columns: {missing[:4]}..."

        # ── Normalize prices for stable training ──────────────────────────
        self._mid_prices = (
            (self.df["bid_price_1"] + self.df["ask_price_1"]) / 2.0
        ).values
        self._price_scale = self._mid_prices.mean()

        # ── Data window (train/test split) ────────────────────────────────
        n = len(self.df)
        self._window_lo = int(n * start_frac)
        self._window_hi = int(n * end_frac) - 1  # last replayable row index
        assert self._window_hi - self._window_lo >= 2, "data window too small"
        self.max_steps = max_steps  # per-episode step cap (None = to window end)
        self.max_step = self._window_hi  # kept for baseline scripts
        self.current_step = self._window_lo
        self._episode_steps = 0

        # ── Row timestamps as epoch seconds (numeric or datetime strings) ──
        self._row_ts = None
        if "timestamp" in self.df.columns:
            ts_col = self.df["timestamp"]
            if pd.api.types.is_numeric_dtype(ts_col):
                self._row_ts = ts_col.to_numpy(dtype=np.float64)
            else:
                try:
                    dt = pd.to_datetime(ts_col, utc=True)
                    # Unit-independent epoch seconds (ns vs us resolution)
                    self._row_ts = (
                        (dt - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
                    ).to_numpy(dtype=np.float64)
                except (ValueError, TypeError):
                    self._row_ts = None

        # Median bar spacing (for Sharpe annualization)
        if self._row_ts is not None and len(self._row_ts) > 1:
            self._bar_seconds = float(np.median(np.diff(self._row_ts)))
        else:
            self._bar_seconds = 1.0

        # ── Time-aligned macro vectors ────────────────────────────────────
        # Pre-computed per-row: row i sees the latest news event published
        # at or before its timestamp. Rows before the first event see zeros.
        self._macro_by_row = None
        if use_macro_vector and macro_vectors_path and os.path.exists(macro_vectors_path):
            self._macro_by_row = self._build_aligned_macro(macro_vectors_path)

        # ── Action & Observation Spaces ───────────────────────────────────
        # Actions: 0=Hold, 1=Market Buy, 2=Market Sell, 3=Limit Buy, 4=Limit Sell
        self.action_space = spaces.Discrete(5)

        self.lob_space = spaces.Box(low=-10, high=10, shape=(40,), dtype=np.float32)

        if self.use_macro_vector:
            self.macro_space = spaces.Box(low=-1.0, high=1.0, shape=(MACRO_DIM,), dtype=np.float32)
            self.observation_space = spaces.Dict({
                "lob": self.lob_space,
                "macro": self.macro_space,
            })
        else:
            self.observation_space = self.lob_space

        # ── Portfolio State ───────────────────────────────────────────────
        self.inventory = 0
        self.cash = self.initial_cash
        self.prev_portfolio_value = self.initial_cash

        # Rolling reward buffer for Sharpe ratio calculation
        self._reward_buffer = []
        self._sharpe_window = 50

        # Trade tracking
        self.trades = []
        self.portfolio_history = []
        self._zero_macro_vec = np.zeros(MACRO_DIM, dtype=np.float32)

    def _build_aligned_macro(self, npz_path):
        """Load {timestamps, scalars, embeddings} and align to LOB rows."""
        data = np.load(npz_path, allow_pickle=True)
        if not all(k in data for k in ("timestamps", "scalars", "embeddings")):
            print(f"[ENV] {os.path.basename(npz_path)} lacks timestamps/scalars/embeddings "
                  f"(old format?) — macro obs will be zeros")
            return None
        if self._row_ts is None:
            print("[ENV] LOB data has no usable timestamp column — macro obs will be zeros")
            return None

        ts = data["timestamps"].astype(np.float64)
        scalars = data["scalars"].astype(np.float32)          # (N, 4)
        emb = data["embeddings"].astype(np.float32)           # (N, 384)

        order = np.argsort(ts)
        ts, scalars, emb = ts[order], scalars[order], emb[order]

        # Pool 384 → 128 (mean over consecutive triples), then L2-normalize
        pooled = emb.reshape(len(emb), MACRO_EMBED_DIM, -1).mean(axis=2)
        norms = np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-8)
        pooled = pooled / norms

        vectors = np.concatenate([np.clip(scalars, -1.0, 1.0), pooled], axis=1)  # (N, 132)

        row_ts = self._row_ts
        idx = np.searchsorted(ts, row_ts, side="right") - 1   # latest event ≤ row time
        aligned = np.zeros((len(self.df), MACRO_DIM), dtype=np.float32)
        has_event = idx >= 0
        aligned[has_event] = vectors[idx[has_event]]

        print(f"[ENV] Aligned {len(ts)} macro events to {int(has_event.sum())}/{len(self.df)} LOB rows")
        return aligned

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        lo, hi = self._window_lo, self._window_hi
        if self.random_start:
            # Leave room for at least a short episode
            span = self.max_steps if self.max_steps else (hi - lo)
            latest = max(lo, hi - max(2, min(span, hi - lo)))
            self.current_step = int(self.np_random.integers(lo, latest + 1))
        else:
            self.current_step = lo
        self._episode_steps = 0

        self.inventory = 0
        self.cash = self.initial_cash
        self.prev_portfolio_value = self.initial_cash
        self._reward_buffer = []
        self.trades = []
        self.portfolio_history = [self.initial_cash]
        self._consecutive_holds = 0

        return self._get_obs(), {}

    def _get_macro_obs(self):
        if self.zero_macro or self._macro_by_row is None:
            return self._zero_macro_vec
        return self._macro_by_row[self.current_step]

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        lob_features = row[self.feature_cols].values.astype(np.float32)

        # Normalize features: prices relative to mid, sizes log-scaled
        # Prices: (price - mid) / mid  → centered around 0 (basis points)
        # Sizes: log1p(size) / 10     → compressed scale
        mid = self._mid_prices[self.current_step]
        if mid > 0:
            # First 20 cols are prices, next 20 are sizes (canonical order)
            lob_features[:20] = (lob_features[:20] - mid) / mid * 100
            lob_features[20:] = np.log1p(lob_features[20:]) / 10.0

        if self.use_macro_vector:
            return {
                "lob": lob_features,
                "macro": self._get_macro_obs(),
            }
        return lob_features

    def _get_mid_price(self):
        row = self.df.iloc[self.current_step]
        return (float(row["bid_price_1"]) + float(row["ask_price_1"])) / 2.0

    def _get_portfolio_value(self):
        mid = self._get_mid_price()
        return self.cash + (self.inventory * mid)

    def _get_terminal_portfolio_value(self, best_bid, taker_fee):
        """Mark remaining long inventory to the bid on the terminal step."""
        return self.cash + max(self.inventory, 0) * best_bid * (1 - taker_fee)

    def step(self, action):
        row = self.df.iloc[self.current_step]
        best_bid = float(row["bid_price_1"])
        best_ask = float(row["ask_price_1"])
        mid_price = (best_bid + best_ask) / 2.0

        # ── Friction Model ────────────────────────────────────────────────
        maker_fee = -0.0001   # rebate (1 basis point)
        taker_fee = 0.0003    # cost (3 basis points)

        trade_executed = False
        execution_price = 0.0
        trade_side = None
        invalid_action = False

        if action == 1:  # Market Buy (cross the spread, take from asks)
            execution_price = best_ask * (1 + taker_fee)
            if self.cash >= execution_price and self.inventory < self.max_inventory:
                self.cash -= execution_price
                self.inventory += 1
                trade_executed = True
                trade_side = "buy"
            else:
                invalid_action = True

        elif action == 2:  # Market Sell (cross the spread, take from bids)
            execution_price = best_bid * (1 - taker_fee)
            if self.inventory > 0:
                self.cash += execution_price
                self.inventory -= 1
                trade_executed = True
                trade_side = "sell"
            else:
                invalid_action = True

        elif action == 3:  # Limit Buy at Best Bid
            fill_prob = min(0.5, float(row.get("bid_size_1", 500)) / 1000.0)
            execution_price = best_bid * (1 + maker_fee)
            if self.cash >= execution_price and self.inventory < self.max_inventory:
                if self.np_random.random() < fill_prob:
                    self.cash -= execution_price
                    self.inventory += 1
                    trade_executed = True
                    trade_side = "limit_buy"
            else:
                invalid_action = True

        elif action == 4:  # Limit Sell at Best Ask
            fill_prob = min(0.5, float(row.get("ask_size_1", 500)) / 1000.0)
            execution_price = best_ask * (1 - maker_fee)
            if self.inventory > 0:
                if self.np_random.random() < fill_prob:
                    self.cash += execution_price
                    self.inventory -= 1
                    trade_executed = True
                    trade_side = "limit_sell"
            else:
                invalid_action = True

        # Record trade
        if trade_executed:
            self.trades.append({
                "step": self.current_step,
                "action": int(action),
                "side": trade_side,
                "price": execution_price,
                "inventory": self.inventory,
                "mid_price": mid_price,
            })

        # ── Advance time ──────────────────────────────────────────────────
        self.current_step += 1
        self._episode_steps += 1
        done = self.current_step >= self._window_hi
        if self.max_steps is not None and self._episode_steps >= self.max_steps:
            done = True
        truncated = False

        # ── Compute Reward ────────────────────────────────────────────────
        portfolio_value = self._get_terminal_portfolio_value(best_bid, taker_fee) if done else self._get_portfolio_value()
        self.portfolio_history.append(portfolio_value)

        reward = self._compute_reward(portfolio_value)

        # Quadratic inventory penalty — small positions are cheap,
        # large positions get increasingly expensive
        inv_penalty = self.inventory_penalty * (self.inventory ** 2)
        reward -= inv_penalty

        # Optional inactivity penalty (off by default — fees and log-return
        # should shape behavior, not a bribe to trade)
        if action == 0:
            self._consecutive_holds += 1
            if self.inactivity_penalty > 0:
                reward -= self.inactivity_penalty * min(self._consecutive_holds / 10.0, 1.0)
        else:
            self._consecutive_holds = 0

        if invalid_action:
            reward -= self.invalid_action_penalty

        self.prev_portfolio_value = portfolio_value

        info = {
            "portfolio_value": portfolio_value,
            "inventory": self.inventory,
            "mid_price": mid_price,
            "trade_executed": trade_executed,
            "step": self.current_step,
        }
        if done:
            info["terminal_portfolio_value"] = portfolio_value

        return self._get_obs(), float(reward), done, truncated, info

    def _compute_reward(self, portfolio_value):
        """Compute reward based on the configured reward type."""

        if self.reward_type == "log_return":
            # Log-return: log(PV_t / PV_{t-1}), guarded against negative/zero PV
            pv = max(portfolio_value, 1e-8)
            prev_pv = max(self.prev_portfolio_value, 1e-8)
            log_ret = np.log(pv / prev_pv)
            return float(np.clip(log_ret * 10000, -100, 100))  # Clip to prevent extreme rewards

        elif self.reward_type == "sharpe":
            # Rolling Sharpe ratio
            if self.prev_portfolio_value > 0:
                ret = (portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
                self._reward_buffer.append(ret)

                if len(self._reward_buffer) >= self._sharpe_window:
                    window = self._reward_buffer[-self._sharpe_window:]
                    mean_ret = np.mean(window)
                    std_ret = np.std(window)
                    sharpe = mean_ret / (std_ret + 1e-8)
                    return float(sharpe)
                else:
                    return float(ret * 100)
            return 0.0

        elif self.reward_type == "pnl":
            # Simple PnL delta (original method, improved)
            pnl_delta = portfolio_value - self.prev_portfolio_value
            return float(pnl_delta / self.initial_cash * 100)

        return 0.0

    def get_episode_stats(self):
        """Compute summary statistics for the completed episode."""
        pv = np.array(self.portfolio_history)
        returns = np.diff(pv) / pv[:-1] if len(pv) > 1 else np.array([0.0])

        total_return = (pv[-1] - pv[0]) / pv[0] if pv[0] > 0 else 0.0
        # Annualize by actual bar spacing: 252 trading days × 6.5h
        periods_per_year = (252 * 6.5 * 3600) / max(self._bar_seconds, 1e-8)
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(periods_per_year) if len(returns) > 1 else 0.0
        max_drawdown = np.min(pv / np.maximum.accumulate(pv) - 1) if len(pv) > 1 else 0.0

        buy_trades = [t for t in self.trades if t["side"] in ("buy", "limit_buy")]
        sell_trades = [t for t in self.trades if t["side"] in ("sell", "limit_sell")]

        return {
            "total_return_pct": round(total_return * 100, 4),
            "sharpe_ratio": round(float(sharpe), 4),
            "max_drawdown_pct": round(float(max_drawdown) * 100, 4),
            "total_trades": len(self.trades),
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "final_pv": round(float(pv[-1]), 2),
            "final_inventory": self.inventory,
        }


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "synthetic_lob_10_level.csv")
    if os.path.exists(data_path):
        print("Testing LOBReplayEnv with log-return reward...")
        env = LOBReplayEnv(data_path=data_path, reward_type="log_return")
        obs, info = env.reset()
        print(f"Obs keys: {list(obs.keys())}")

        done = False
        total_reward = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if truncated:
                break

        stats = env.get_episode_stats()
        print(f"\nEpisode Stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print(f"  Total reward: {total_reward:.4f}")
    else:
        print("Data file not found. Run data/download_lob.py first.")
