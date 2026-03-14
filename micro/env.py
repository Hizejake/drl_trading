"""
LOBReplayEnv — Custom Gymnasium environment for LOB tick-by-tick replay.

Supports:
- Real LOBSTER data and synthetic data (auto-detection)
- Configurable reward functions: log-return, sharpe, pnl
- Market frictions: taker fees, probabilistic maker rebates
- Optional pre-computed macro vectors from LLM swarm
- Inventory penalty to discourage excessive accumulation
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os


class LOBReplayEnv(gym.Env):
    """
    A custom Gymnasium replay environment that iterates over historical LOB data.
    
    Args:
        data_path: Path to LOB CSV (40 price/size columns required)
        window_size: Not used currently (reserved for stacked observations)
        use_macro_vector: Whether to include 128D macro vector in obs
        macro_vectors_path: Path to .npz file with pre-computed macro vectors
        reward_type: 'log_return' | 'sharpe' | 'pnl' (default: 'log_return')
        inventory_penalty: Penalty coefficient for inventory accumulation
        initial_cash: Starting cash amount
    """
    
    REWARD_TYPES = ("log_return", "sharpe", "pnl")
    
    def __init__(
        self,
        data_path,
        window_size=50,
        use_macro_vector=True,
        macro_vectors_path=None,
        reward_type="log_return",
        inventory_penalty=0.001,
        initial_cash=10000.0,
        max_steps=None,
    ):
        super().__init__()
        
        self.data_path = data_path
        self.window_size = window_size
        self.use_macro_vector = use_macro_vector
        self.reward_type = reward_type
        self.inventory_penalty = inventory_penalty
        self.initial_cash = initial_cash
        
        assert reward_type in self.REWARD_TYPES, f"reward_type must be one of {self.REWARD_TYPES}"
        
        # ── Load Data ─────────────────────────────────────────────────────
        self.df = pd.read_csv(data_path)
        
        # Auto-detect columns: both LOBSTER parsed and synthetic have same format
        price_cols = sorted([c for c in self.df.columns if "price" in c])
        size_cols = sorted([c for c in self.df.columns if "size" in c])
        self.feature_cols = price_cols + size_cols
        
        assert len(self.feature_cols) == 40, (
            f"Expected 40 LOB features (10 levels × 4 fields), got {len(self.feature_cols)}. "
            f"Columns found: {self.feature_cols}"
        )
        
        # ── Normalize prices for stable training ──────────────────────────
        # Compute normalization factors from the data
        self._mid_prices = (
            (self.df["bid_price_1"] + self.df["ask_price_1"]) / 2.0
        ).values
        self._price_scale = self._mid_prices.mean()
        
        self._data_len = len(self.df) - 1
        self.max_step = min(self._data_len, max_steps) if max_steps else self._data_len
        self.current_step = 0
        
        # ── Load pre-computed macro vectors ───────────────────────────────
        self.macro_vectors = None
        if macro_vectors_path and os.path.exists(macro_vectors_path):
            data = np.load(macro_vectors_path, allow_pickle=True)
            if "embeddings" in data:
                self.macro_vectors = data["embeddings"].astype(np.float32)
                # Normalize to [-1, 1] range
                norms = np.linalg.norm(self.macro_vectors, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                self.macro_vectors = self.macro_vectors / norms
                print(f"[ENV] Loaded {len(self.macro_vectors)} pre-computed macro vectors")
        
        # ── Action & Observation Spaces ───────────────────────────────────
        # Actions: 0=Hold, 1=Market Buy, 2=Market Sell, 3=Limit Buy, 4=Limit Sell
        self.action_space = spaces.Discrete(5)
        
        self.lob_space = spaces.Box(low=-10, high=10, shape=(40,), dtype=np.float32)
        
        if self.use_macro_vector:
            self.macro_space = spaces.Box(low=-1.0, high=1.0, shape=(128,), dtype=np.float32)
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
        
        # Episode macro vector (randomized if no pre-computed vectors)
        self.episode_macro_vector = np.zeros(128, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.inventory = 0
        self.cash = self.initial_cash
        self.prev_portfolio_value = self.initial_cash
        self._reward_buffer = []
        self.trades = []
        self.portfolio_history = [self.initial_cash]
        
        # Select macro vector for episode
        if self.macro_vectors is not None and len(self.macro_vectors) > 0:
            idx = self.np_random.integers(0, len(self.macro_vectors))
            full_vec = self.macro_vectors[idx]
            # Project 384D → 128D via simple chunked averaging if needed
            if len(full_vec) > 128:
                # Take first 128 or average-pool
                self.episode_macro_vector = full_vec[:128]
            else:
                self.episode_macro_vector = full_vec
        else:
            self.episode_macro_vector = self.np_random.uniform(
                -0.1, 0.1, size=(128,)
            ).astype(np.float32)
        
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        lob_features = row[self.feature_cols].values.astype(np.float32)
        
        # Normalize features: prices relative to mid, sizes log-scaled
        # Prices: (price - mid) / mid  → centered around 0
        # Sizes: log1p(size) / 10     → compressed scale
        mid = self._mid_prices[self.current_step]
        if mid > 0:
            # First 20 cols are prices, next 20 are sizes
            lob_features[:20] = (lob_features[:20] - mid) / mid * 100  # basis points
            lob_features[20:] = np.log1p(lob_features[20:]) / 10.0
        
        if self.use_macro_vector:
            return {
                "lob": lob_features,
                "macro": self.episode_macro_vector,
            }
        return lob_features

    def _get_mid_price(self):
        row = self.df.iloc[self.current_step]
        return (float(row["bid_price_1"]) + float(row["ask_price_1"])) / 2.0

    def _get_portfolio_value(self):
        mid = self._get_mid_price()
        return self.cash + (self.inventory * mid)

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
        
        if action == 1:  # Market Buy (cross the spread, take from asks)
            execution_price = best_ask * (1 + taker_fee)
            self.cash -= execution_price
            self.inventory += 1
            trade_executed = True
            trade_side = "buy"
            
        elif action == 2:  # Market Sell (cross the spread, take from bids)
            execution_price = best_bid * (1 - taker_fee)
            self.cash += execution_price
            self.inventory -= 1
            trade_executed = True
            trade_side = "sell"
            
        elif action == 3:  # Limit Buy at Best Bid
            fill_prob = min(0.5, float(row.get("bid_size_1", 500)) / 1000.0)
            if self.np_random.random() < fill_prob:
                execution_price = best_bid * (1 + maker_fee)
                self.cash -= execution_price
                self.inventory += 1
                trade_executed = True
                trade_side = "limit_buy"
                
        elif action == 4:  # Limit Sell at Best Ask
            fill_prob = min(0.5, float(row.get("ask_size_1", 500)) / 1000.0)
            if self.np_random.random() < fill_prob:
                execution_price = best_ask * (1 - maker_fee)
                self.cash += execution_price
                self.inventory -= 1
                trade_executed = True
                trade_side = "limit_sell"
        
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
        done = self.current_step >= self.max_step
        truncated = False
        
        # ── Compute Reward ────────────────────────────────────────────────
        portfolio_value = self._get_portfolio_value()
        self.portfolio_history.append(portfolio_value)
        
        reward = self._compute_reward(portfolio_value)
        
        # Inventory penalty — discourages accumulating large positions
        inv_penalty = self.inventory_penalty * abs(self.inventory)
        reward -= inv_penalty
        
        self.prev_portfolio_value = portfolio_value
        
        info = {
            "portfolio_value": portfolio_value,
            "inventory": self.inventory,
            "mid_price": mid_price,
            "trade_executed": trade_executed,
            "step": self.current_step,
        }
        
        return self._get_obs(), float(reward), done, truncated, info

    def _compute_reward(self, portfolio_value):
        """Compute reward based on the configured reward type."""
        
        if self.reward_type == "log_return":
            # Log-return: log(PV_t / PV_{t-1})
            if self.prev_portfolio_value > 0:
                log_ret = np.log(portfolio_value / self.prev_portfolio_value)
                return float(log_ret * 10000)  # Scale to basis points for numerical stability
            return 0.0
        
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
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252 * 6.5 * 3600 / len(pv)) if len(returns) > 1 else 0.0
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
