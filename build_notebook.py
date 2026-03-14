"""Script to generate the Kaggle-ready Jupyter notebook."""
import json, os

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.split("\n")})

def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "source": source.split("\n"), "outputs": [], "execution_count": None})

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1: Title
# ═══════════════════════════════════════════════════════════════════════════════
md("""# 🤖 Hierarchical Dual-Frequency DRL Trading Bot

**Architecture**: Macro LLM Swarm (GDELT news → 5 LLM personas → 128D embeddings) + Micro CVML (LOBSTER LOB → Conv2D → 64D) → PPO

**This notebook is self-contained** — all code is inline. Upload your LOBSTER data to Kaggle and run with GPU.

---""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2: Install
# ═══════════════════════════════════════════════════════════════════════════════
code("""# ── Install Dependencies ───────────────────────────────────────────────────────
!pip install -q stable-baselines3 gymnasium litellm sentence-transformers python-dotenv

import os, sys, time, json, csv, re, asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch {torch.__version__} | Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3: Config
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 1. Configuration

Set your data paths. On Kaggle, upload LOBSTER data as a dataset and update `DATA_DIR`.
""")

code("""# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Update these paths for your environment:

# Local:
DATA_DIR = "data/raw"

# Kaggle (uncomment if running on Kaggle):
# DATA_DIR = "/kaggle/input/lobster-lob-data"

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# API key for LLM Swarm
OPENROUTER_API_KEY = ""

# Try to load from Kaggle Secrets securely if running on Kaggle
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    OPENROUTER_API_KEY = user_secrets.get_secret("OPENROUTER_API_KEY")
    print("Successfully loaded OPENROUTER_API_KEY from Kaggle Secrets")
except Exception:
    pass

# Training config
TRAIN_TIMESTEPS = 500_000   # Full scale run
REWARD_TYPE = "log_return"  # "log_return" | "sharpe" | "pnl"
TICKER = "aapl"             # Which LOBSTER dataset to use

print(f"Data dir: {DATA_DIR}")
print(f"Models dir: {MODELS_DIR}")
print(f"Training timesteps: {TRAIN_TIMESTEPS:,}")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4: LOBSTER Parser
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 2. Data Pipeline — LOBSTER Parser

Parses raw LOBSTER orderbook + message CSVs into our standardized 40-column format:
`bid_price_1..10, ask_price_1..10, bid_size_1..10, ask_size_1..10`
""")

code("""import glob

def parse_lobster_data(ticker="AAPL", data_dir=DATA_DIR):
    \"\"\"Parse LOBSTER orderbook + message CSVs into standardized format.\"\"\"
    ob_pattern = os.path.join(data_dir, f"{ticker}_*_orderbook_10.csv")
    msg_pattern = os.path.join(data_dir, f"{ticker}_*_message_10.csv")
    
    ob_files = glob.glob(ob_pattern)
    msg_files = glob.glob(msg_pattern)
    
    if not ob_files:
        print(f"No orderbook file found for {ticker}. Generating synthetic data...")
        return generate_synthetic_lob(data_dir=data_dir)
    
    ob_path = ob_files[0]
    msg_path = msg_files[0] if msg_files else None
    
    print(f"Parsing {ticker}...")
    
    # LOBSTER interleaves: ask_p1, ask_s1, bid_p1, bid_s1, ...
    lobster_cols = []
    for level in range(1, 11):
        lobster_cols.extend([
            f"ask_price_{level}", f"ask_size_{level}",
            f"bid_price_{level}", f"bid_size_{level}",
        ])
    
    df_ob = pd.read_csv(ob_path, header=None, names=lobster_cols)
    print(f"  Raw: {len(df_ob):,} rows × {len(df_ob.columns)} cols")
    
    # Timestamps from message file
    timestamps = np.arange(len(df_ob), dtype=float)
    if msg_path:
        df_msg = pd.read_csv(msg_path, header=None,
                             names=["timestamp", "event_type", "order_id", "size", "price", "direction"])
        timestamps = df_msg["timestamp"].values
    
    # Convert integer prices to dollars (÷ 10000)
    for level in range(1, 11):
        df_ob[f"ask_price_{level}"] = df_ob[f"ask_price_{level}"] / 10000.0
        df_ob[f"bid_price_{level}"] = df_ob[f"bid_price_{level}"] / 10000.0
    
    # Filter dummy rows
    mask = (df_ob["bid_size_1"] > 0) & (df_ob["ask_size_1"] > 0)
    mask &= (df_ob["bid_price_1"] > 0) & (df_ob["ask_price_1"] < 999999)
    df_ob = df_ob[mask].reset_index(drop=True)
    timestamps = timestamps[mask.values][:len(df_ob)]
    
    # Reorder to our standard format
    out_cols = (
        [f"bid_price_{i}" for i in range(1, 11)] +
        [f"ask_price_{i}" for i in range(1, 11)] +
        [f"bid_size_{i}" for i in range(1, 11)] +
        [f"ask_size_{i}" for i in range(1, 11)]
    )
    
    df_out = pd.DataFrame()
    df_out["timestamp"] = timestamps
    df_out["symbol"] = ticker
    for col in out_cols:
        df_out[col] = df_ob[col].values
    
    # Save to current working directory (e.g. /kaggle/working)
    csv_path = f"lobster_{ticker.lower()}_10_level.csv"
    df_out.to_csv(csv_path, index=False)
    
    mid = (df_out["bid_price_1"] + df_out["ask_price_1"]) / 2
    spread = df_out["ask_price_1"] - df_out["bid_price_1"]
    print(f"  Saved: {len(df_out):,} ticks, mid=${mid.mean():.2f}, spread={spread.mean()/mid.mean()*10000:.1f}bps")
    return csv_path


def generate_synthetic_lob(ticks=5000, data_dir=DATA_DIR):
    \"\"\"Fallback synthetic LOB data.\"\"\"
    os.makedirs(data_dir, exist_ok=True)
    base_price = 150.0
    data = {}
    price_walk = np.cumsum(np.random.normal(0, 0.02, ticks))
    for level in range(1, 11):
        data[f"bid_price_{level}"] = np.round(base_price + price_walk - level * 0.01, 4)
        data[f"ask_price_{level}"] = np.round(base_price + price_walk + level * 0.01, 4)
        data[f"bid_size_{level}"] = np.random.randint(50 * level, 500 * level, size=ticks)
        data[f"ask_size_{level}"] = np.random.randint(50 * level, 500 * level, size=ticks)
    df = pd.DataFrame(data)
    df.insert(0, "timestamp", pd.date_range("2024-01-01 09:30:00", periods=ticks, freq="100ms"))
    df.insert(1, "symbol", "SYNTH")
    path = "synthetic_lob_10_level.csv"
    df.to_csv(path, index=False)
    print(f"Synthetic: {ticks} ticks saved to {path}")
    return path""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5: Parse Data
# ═══════════════════════════════════════════════════════════════════════════════
code("""# ── Parse / Generate Data ──────────────────────────────────────────────────────
# Check if a pre-parsed file already exists in either the working dir or DATA_DIR
local_parsed = f"lobster_{TICKER.lower()}_10_level.csv"
dataset_parsed = os.path.join(DATA_DIR, local_parsed)

if os.path.exists(local_parsed):
    print(f"Using existing parsed data: {local_parsed}")
    DATA_PATH = local_parsed
elif os.path.exists(dataset_parsed):
    print(f"Using parsed data from dataset: {dataset_parsed}")
    DATA_PATH = dataset_parsed
else:
    DATA_PATH = parse_lobster_data(TICKER.upper(), data_dir=DATA_DIR)

df_preview = pd.read_csv(DATA_PATH, nrows=5)
print(f"\\nData shape: {pd.read_csv(DATA_PATH).shape}")
df_preview.head()""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6: CVML Module
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 3. CVML — Convolutional Cross-Variate Mixing Layer

Reconstructs flat LOB features into a 2D spatial tensor `(Batch, 4, 10, 1)` and applies
depthwise convolution along price levels + pointwise convolution across variates.
""")

code("""class CVML(nn.Module):
    \"\"\"
    Convolutional Cross-Variate Mixing Layer for Limit Order Book data.
    Input: (Batch, 40) flat LOB features
    Output: (Batch, out_dim) extracted microstructure features
    \"\"\"
    def __init__(self, in_channels=4, levels=10, out_dim=64):
        super().__init__()
        self.levels = levels
        self.in_channels = in_channels
        
        # Depthwise: convolve along 3 adjacent price levels per channel
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(3, 1), padding=(1, 0), groups=in_channels
        )
        
        # Pointwise: mix the 4 variates (bid_p, ask_p, bid_s, ask_s)
        self.pointwise = nn.Conv2d(in_channels, 16, kernel_size=1)
        
        self.activation = nn.ReLU()
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(16 * levels, out_dim)
    
    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.in_channels, self.levels, 1)
        x = self.activation(self.depthwise(x))
        x = self.activation(self.pointwise(x))
        x = self.projection(self.flatten(x))
        return x

# Quick test
_m = CVML().to(DEVICE)
_x = torch.randn(4, 40).to(DEVICE)
print(f"CVML: {_x.shape} → {_m(_x).shape}")
del _m, _x""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7: LOBReplayEnv
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 4. LOBReplayEnv — Gymnasium Environment

Tick-by-tick replay with:
- 5 actions: Hold, Market Buy, Market Sell, Limit Buy, Limit Sell
- Market frictions: 3bps taker fee, 1bps maker rebate
- 3 reward functions: log-return, rolling Sharpe, PnL delta
- Inventory penalty
""")

code("""class LOBReplayEnv(gym.Env):
    REWARD_TYPES = ("log_return", "sharpe", "pnl")
    
    def __init__(self, data_path, use_macro_vector=True, macro_vectors_path=None,
                 reward_type="log_return", inventory_penalty=0.001, initial_cash=10000.0, max_steps=None):
        super().__init__()
        self.data_path = data_path
        self.use_macro_vector = use_macro_vector
        self.reward_type = reward_type
        self.inventory_penalty = inventory_penalty
        self.initial_cash = initial_cash
        
        self.df = pd.read_csv(data_path)
        price_cols = sorted([c for c in self.df.columns if "price" in c])
        size_cols = sorted([c for c in self.df.columns if "size" in c])
        self.feature_cols = price_cols + size_cols
        assert len(self.feature_cols) == 40
        
        self._mid_prices = ((self.df["bid_price_1"] + self.df["ask_price_1"]) / 2.0).values
        self._data_len = len(self.df) - 1
        self.max_step = min(self._data_len, max_steps) if max_steps else self._data_len
        self.current_step = 0
        
        # Load macro vectors
        self.macro_vectors = None
        if macro_vectors_path and os.path.exists(macro_vectors_path):
            data = np.load(macro_vectors_path, allow_pickle=True)
            if "embeddings" in data:
                self.macro_vectors = data["embeddings"].astype(np.float32)
                norms = np.maximum(np.linalg.norm(self.macro_vectors, axis=1, keepdims=True), 1e-8)
                self.macro_vectors = self.macro_vectors / norms
        
        self.action_space = spaces.Discrete(5)
        self.lob_space = spaces.Box(low=-10, high=10, shape=(40,), dtype=np.float32)
        
        if self.use_macro_vector:
            self.macro_space = spaces.Box(low=-1.0, high=1.0, shape=(128,), dtype=np.float32)
            self.observation_space = spaces.Dict({"lob": self.lob_space, "macro": self.macro_space})
        else:
            self.observation_space = self.lob_space
        
        self.inventory = 0
        self.cash = initial_cash
        self.prev_portfolio_value = initial_cash
        self._reward_buffer = []
        self._sharpe_window = 50
        self.trades = []
        self.portfolio_history = [initial_cash]
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
        
        if self.macro_vectors is not None and len(self.macro_vectors) > 0:
            idx = self.np_random.integers(0, len(self.macro_vectors))
            full_vec = self.macro_vectors[idx]
            self.episode_macro_vector = full_vec[:128] if len(full_vec) > 128 else full_vec
        else:
            self.episode_macro_vector = self.np_random.uniform(-0.1, 0.1, size=(128,)).astype(np.float32)
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        lob = row[self.feature_cols].values.astype(np.float32)
        mid = self._mid_prices[self.current_step]
        if mid > 0:
            lob[:20] = (lob[:20] - mid) / mid * 100
            lob[20:] = np.log1p(lob[20:]) / 10.0
        if self.use_macro_vector:
            return {"lob": lob, "macro": self.episode_macro_vector}
        return lob

    def step(self, action):
        row = self.df.iloc[self.current_step]
        best_bid = float(row["bid_price_1"])
        best_ask = float(row["ask_price_1"])
        mid_price = (best_bid + best_ask) / 2.0
        
        maker_fee, taker_fee = -0.0001, 0.0003
        trade_executed, trade_side = False, None
        
        if action == 1:  # Market Buy
            self.cash -= best_ask * (1 + taker_fee)
            self.inventory += 1
            trade_executed, trade_side = True, "buy"
        elif action == 2:  # Market Sell
            self.cash += best_bid * (1 - taker_fee)
            self.inventory -= 1
            trade_executed, trade_side = True, "sell"
        elif action == 3:  # Limit Buy
            fill_prob = min(0.5, float(row.get("bid_size_1", 500)) / 1000.0)
            if self.np_random.random() < fill_prob:
                self.cash -= best_bid * (1 + maker_fee)
                self.inventory += 1
                trade_executed, trade_side = True, "limit_buy"
        elif action == 4:  # Limit Sell
            fill_prob = min(0.5, float(row.get("ask_size_1", 500)) / 1000.0)
            if self.np_random.random() < fill_prob:
                self.cash += best_ask * (1 - maker_fee)
                self.inventory -= 1
                trade_executed, trade_side = True, "limit_sell"
        
        if trade_executed:
            self.trades.append({"step": self.current_step, "action": int(action),
                                "side": trade_side, "price": best_ask if "buy" in trade_side else best_bid,
                                "inventory": self.inventory, "mid_price": mid_price})
        
        self.current_step += 1
        done = self.current_step >= self.max_step
        
        pv = self.cash + self.inventory * mid_price
        self.portfolio_history.append(pv)
        reward = self._compute_reward(pv) - self.inventory_penalty * abs(self.inventory)
        self.prev_portfolio_value = pv
        
        return self._get_obs(), float(reward), done, False, \\
               {"portfolio_value": pv, "inventory": self.inventory,
                "mid_price": mid_price, "trade_executed": trade_executed, "step": self.current_step}

    def _compute_reward(self, pv):
        if self.reward_type == "log_return":
            return float(np.log(pv / self.prev_portfolio_value) * 10000) if self.prev_portfolio_value > 0 else 0.0
        elif self.reward_type == "sharpe":
            if self.prev_portfolio_value > 0:
                ret = (pv - self.prev_portfolio_value) / self.prev_portfolio_value
                self._reward_buffer.append(ret)
                if len(self._reward_buffer) >= self._sharpe_window:
                    w = self._reward_buffer[-self._sharpe_window:]
                    return float(np.mean(w) / (np.std(w) + 1e-8))
                return float(ret * 100)
            return 0.0
        elif self.reward_type == "pnl":
            return float((pv - self.prev_portfolio_value) / self.initial_cash * 100)
        return 0.0

    def get_episode_stats(self):
        pv = np.array(self.portfolio_history)
        rets = np.diff(pv) / pv[:-1] if len(pv) > 1 else np.array([0.0])
        total_ret = (pv[-1] - pv[0]) / pv[0] if pv[0] > 0 else 0.0
        sharpe = (np.mean(rets) / (np.std(rets) + 1e-8)) * np.sqrt(252 * 6.5 * 3600 / len(pv)) if len(rets) > 1 else 0.0
        max_dd = np.min(pv / np.maximum.accumulate(pv) - 1) if len(pv) > 1 else 0.0
        buys = [t for t in self.trades if "buy" in t["side"]]
        sells = [t for t in self.trades if "sell" in t["side"]]
        return {"total_return_pct": round(total_ret * 100, 4), "sharpe_ratio": round(float(sharpe), 4),
                "max_drawdown_pct": round(float(max_dd) * 100, 4), "total_trades": len(self.trades),
                "buy_trades": len(buys), "sell_trades": len(sells),
                "final_pv": round(float(pv[-1]), 2), "final_inventory": self.inventory}

# Quick test
_env = LOBReplayEnv(DATA_PATH)
_obs, _ = _env.reset()
print(f"Env OK: obs_keys={list(_obs.keys())}, lob_shape={_obs['lob'].shape}, macro_shape={_obs['macro'].shape}")
_obs, r, d, t, info = _env.step(1)
print(f"Step: reward={r:.4f}, pv=${info['portfolio_value']:.2f}")
del _env""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8: Feature Extractors
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 5. Feature Extractors — SB3 Custom Policies""")

code("""class HierarchicalFeatureExtractor(BaseFeaturesExtractor):
    \"\"\"CVML (64D) + Macro (128D) → 192D combined features.\"\"\"
    def __init__(self, observation_space: spaces.Dict, cvml_out_dim=64, **kwargs):
        super().__init__(observation_space, features_dim=cvml_out_dim + 128)
        self.cvml = CVML(in_channels=4, levels=10, out_dim=cvml_out_dim)

    def forward(self, observations):
        lob = observations["lob"].float()
        macro = observations["macro"].float()
        return torch.cat([self.cvml(lob), macro], dim=1)


class FlatMLPFeatureExtractor(BaseFeaturesExtractor):
    \"\"\"Ablation baseline: flat MLP instead of CVML.\"\"\"
    def __init__(self, observation_space: spaces.Dict, features_dim=192, **kwargs):
        super().__init__(observation_space, features_dim=features_dim)
        self.mlp = nn.Sequential(nn.Linear(40, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())

    def forward(self, observations):
        return torch.cat([self.mlp(observations["lob"].float()), observations["macro"].float()], dim=1)

print("Feature extractors defined ✓")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9: Training
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 6. Training — PPO with CVML

Training the full hierarchical model on LOBSTER LOB data.
""")

code("""def linear_schedule(initial_lr):
    def schedule(progress_remaining):
        return progress_remaining * initial_lr
    return schedule


class MetricsLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.log_data = []
    
    def _on_step(self):
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
        return True
    
    def _on_rollout_end(self):
        if self.episode_rewards:
            self.log_data.append({
                "timestep": self.num_timesteps,
                "mean_reward_10": np.mean(self.episode_rewards[-10:]),
                "total_episodes": len(self.episode_rewards),
            })


def train_model(data_path, use_cvml=True, total_timesteps=10000, reward_type="log_return"):
    name = "PPO+CVML" if use_cvml else "PPO+FlatMLP"
    print(f"\\n{'='*60}")
    print(f"Training {name} | {total_timesteps:,} steps | reward={reward_type} | device={DEVICE}")
    print(f"{'='*60}")
    
    train_max_steps = min(total_timesteps * 2, 50000)
    env = LOBReplayEnv(data_path, reward_type=reward_type, max_steps=train_max_steps)
    
    extractor = HierarchicalFeatureExtractor if use_cvml else FlatMLPFeatureExtractor
    model = PPO(
        "MultiInputPolicy", env,
        policy_kwargs={"features_extractor_class": extractor, "features_extractor_kwargs": {},
                       "net_arch": [256, 128, 64]},
        learning_rate=linear_schedule(3e-4),
        n_steps=min(2048, env.max_step // 2),
        batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
        verbose=1, device=DEVICE,
    )
    
    logger = MetricsLoggerCallback()
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=[logger])
    elapsed = time.time() - t0
    
    save_name = "ppo_cvml_final" if use_cvml else "ppo_flat_final"
    model.save(os.path.join(MODELS_DIR, save_name))
    print(f"\\n{name} trained in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    return model, logger

print("Training function defined ✓")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10: Run Training
# ═══════════════════════════════════════════════════════════════════════════════
md("""### 6a. Train CVML Model""")

code("""cvml_model, cvml_logger = train_model(DATA_PATH, use_cvml=True,
                                      total_timesteps=TRAIN_TIMESTEPS,
                                      reward_type=REWARD_TYPE)""")

md("""### 6b. Train Flat MLP Baseline (Ablation)""")

code("""flat_model, flat_logger = train_model(DATA_PATH, use_cvml=False,
                                      total_timesteps=TRAIN_TIMESTEPS,
                                      reward_type=REWARD_TYPE)""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 11: Baselines
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 7. Baselines — TWAP, VWAP, Random Agent""")

code("""def evaluate_agent(model, env, name="Agent"):
    obs, _ = env.reset()
    done, total_reward, steps = False, 0.0, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if truncated: break
    stats = env.get_episode_stats()
    stats["total_reward"] = round(total_reward, 4)
    stats["name"] = name
    return stats


def run_twap(env):
    obs, _ = env.reset()
    done, steps = False, 0
    while not done:
        if steps % 50 == 0 and steps < env.max_step - 100:
            action = 1
        elif steps >= env.max_step - 50 and env.inventory > 0:
            action = 2
        else:
            action = 0
        obs, r, done, trunc, info = env.step(action)
        steps += 1
        if trunc: break
    s = env.get_episode_stats(); s["name"] = "TWAP"; return s


def run_vwap(env):
    obs, _ = env.reset()
    done, steps, total_vol = False, 0, 0
    while not done:
        row = env.df.iloc[env.current_step]
        vol = sum(float(row.get(f"bid_size_{i}", 0)) + float(row.get(f"ask_size_{i}", 0)) for i in range(1, 6))
        total_vol += vol
        avg_vol = total_vol / max(steps + 1, 1)
        if vol > avg_vol * 1.2 and steps < env.max_step - 100:
            action = 1
        elif steps >= env.max_step - 50 and env.inventory > 0:
            action = 2
        elif vol < avg_vol * 0.8 and env.inventory > 0:
            action = 2
        else:
            action = 0
        obs, r, done, trunc, info = env.step(action)
        steps += 1
        if trunc: break
    s = env.get_episode_stats(); s["name"] = "VWAP"; return s


def run_random(env, seed=42):
    obs, _ = env.reset()
    done, rng = False, np.random.RandomState(seed)
    while not done:
        obs, r, done, trunc, info = env.step(rng.randint(0, 5))
        if trunc: break
    s = env.get_episode_stats(); s["name"] = "Random"; return s

print("Baseline functions defined ✓")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 12: Run Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 8. Evaluation — All Models vs Baselines""")

code("""all_stats = []

# Evaluate trained models
for model, name in [(cvml_model, "PPO+CVML"), (flat_model, "PPO+FlatMLP")]:
    env = LOBReplayEnv(DATA_PATH, reward_type=REWARD_TYPE, max_steps=5000)
    all_stats.append(evaluate_agent(model, env, name))

# Run baselines
for fn in [run_twap, run_vwap, run_random]:
    env = LOBReplayEnv(DATA_PATH, reward_type=REWARD_TYPE, max_steps=5000)
    all_stats.append(fn(env))

# Print comparison table
print(f"\\n{'='*95}")
print(f"{'EVALUATION RESULTS':^95}")
print(f"{'='*95}")
print(f"{'Model':>15} {'Return%':>10} {'Sharpe':>10} {'MaxDD%':>10} {'Trades':>8} {'Buy':>6} {'Sell':>6} {'Final PV':>12} {'Inv':>6}")
print("-" * 95)
for s in all_stats:
    print(f"{s['name']:>15} {s['total_return_pct']:>10.4f} {s['sharpe_ratio']:>10.4f} "
          f"{s['max_drawdown_pct']:>10.4f} {s['total_trades']:>8} {s['buy_trades']:>6} "
          f"{s['sell_trades']:>6}   ${s['final_pv']:>9.2f} {s['final_inventory']:>6}")
print(f"{'='*95}")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 13: Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 9. Interactive Backtest Dashboard

Generates an HTML dashboard with PnL curve, trade markers, inventory, and metrics.
""")

code("""from IPython.display import HTML, display
import json as _json

def generate_dashboard(model, data_path, model_name="PPO+CVML", max_ticks=None):
    env = LOBReplayEnv(data_path, reward_type=REWARD_TYPE)
    obs, _ = env.reset()
    ticks, done, step = [], False, 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        ticks.append({"step": step, "mid_price": info["mid_price"],
                      "portfolio_value": info["portfolio_value"],
                      "inventory": info["inventory"], "reward": float(reward)})
        step += 1
        if truncated or (max_ticks and step >= max_ticks): break
    
    stats = env.get_episode_stats()
    trades = env.trades
    
    # Subsample for display
    if len(ticks) > 5000:
        s = len(ticks) // 5000
        ticks = ticks[::s]
    
    steps = [t["step"] for t in ticks]
    mid_p = [round(t["mid_price"], 4) for t in ticks]
    pv = [round(t["portfolio_value"], 2) for t in ticks]
    inv = [t["inventory"] for t in ticks]
    
    buy_t = [{"x": t["step"], "y": round(t["price"], 4)} for t in trades if "buy" in t["side"]][:500]
    sell_t = [{"x": t["step"], "y": round(t["price"], 4)} for t in trades if "sell" in t["side"]][:500]
    
    cum_r = []
    total = 0
    for t in ticks:
        total += t["reward"]
        cum_r.append(round(total, 4))
    
    ret_class = "positive" if stats["total_return_pct"] >= 0 else "negative"
    sharpe_class = "positive" if stats["sharpe_ratio"] >= 0 else "negative"

    html = f'''
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        .dash {{ font-family: Inter, Arial, sans-serif; background: #0a0a0f; color: #e0e0e8; padding: 20px; border-radius: 12px; }}
        .dash h2 {{ background: linear-gradient(135deg, #00d2ff, #7b2ff7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .metrics {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }}
        .metric {{ background: #1a1a2e; border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 14px 18px; min-width: 140px; }}
        .metric .label {{ font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #6666a0; }}
        .metric .val {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
        .positive {{ color: #00e676; }} .negative {{ color: #ff5252; }} .neutral {{ color: #e0e0e8; }}
        .chart-box {{ background: #1a1a2e; border-radius: 10px; padding: 16px; margin: 12px 0; }}
        .chart-box h4 {{ color: #a0a0c0; margin-bottom: 10px; }}
    </style>
    <div class="dash">
        <h2>DRL Trading Bot — Backtest Dashboard</h2>
        <p style="color:#888; font-size:13px;">{model_name} | {len(ticks):,} ticks</p>
        <div class="metrics">
            <div class="metric"><div class="label">Return</div><div class="val {ret_class}">{stats["total_return_pct"]:+.4f}%</div></div>
            <div class="metric"><div class="label">Final PV</div><div class="val neutral">${stats["final_pv"]:,.2f}</div></div>
            <div class="metric"><div class="label">Sharpe</div><div class="val {sharpe_class}">{stats["sharpe_ratio"]:+.4f}</div></div>
            <div class="metric"><div class="label">Max DD</div><div class="val negative">{stats["max_drawdown_pct"]:.4f}%</div></div>
            <div class="metric"><div class="label">Trades</div><div class="val neutral">{stats["total_trades"]:,}</div></div>
            <div class="metric"><div class="label">Buy/Sell</div><div class="val neutral">{stats["buy_trades"]}/{stats["sell_trades"]}</div></div>
            <div class="metric"><div class="label">Inventory</div><div class="val neutral">{stats["final_inventory"]}</div></div>
        </div>
        <div class="chart-box"><h4>Portfolio Value</h4><canvas id="c1" height="200"></canvas></div>
        <div class="chart-box"><h4>Mid Price + Trades</h4><canvas id="c2" height="200"></canvas></div>
        <div class="chart-box"><h4>Inventory</h4><canvas id="c3" height="150"></canvas></div>
        <div class="chart-box"><h4>Cumulative Reward</h4><canvas id="c4" height="150"></canvas></div>
    </div>
    <script>
        const S={_json.dumps(steps)}, M={_json.dumps(mid_p)}, P={_json.dumps(pv)},
              I={_json.dumps(inv)}, R={_json.dumps(cum_r)},
              BT={_json.dumps(buy_t)}, ST={_json.dumps(sell_t)};
        const D={{responsive:true,animation:false,plugins:{{legend:{{labels:{{color:'#888',font:{{size:10}}}}}}}},
            scales:{{x:{{grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:'#666',maxTicksLimit:8}}}},
                     y:{{grid:{{color:'rgba(255,255,255,0.04)'}},ticks:{{color:'#666'}}}}}}}};
        new Chart(document.getElementById('c1'),{{type:'line',data:{{labels:S,datasets:[{{label:'PV($)',data:P,borderColor:'#00d2ff',borderWidth:1.5,pointRadius:0,fill:true,backgroundColor:'rgba(0,210,255,0.08)'}}]}},options:D}});
        new Chart(document.getElementById('c2'),{{type:'line',data:{{labels:S,datasets:[{{label:'Mid',data:M,borderColor:'#7b2ff7',borderWidth:1,pointRadius:0,order:2}},{{label:'Buy',data:BT,type:'scatter',backgroundColor:'#00e676',pointRadius:2,order:1}},{{label:'Sell',data:ST,type:'scatter',backgroundColor:'#ff5252',pointRadius:2,order:1}}]}},options:{{...D,scales:{{...D.scales,x:{{...D.scales.x,type:'linear'}}}}}}}});
        new Chart(document.getElementById('c3'),{{type:'line',data:{{labels:S,datasets:[{{label:'Inv',data:I,borderColor:'#ff9800',borderWidth:1.5,pointRadius:0,stepped:true,fill:true,backgroundColor:'rgba(255,152,0,0.08)'}}]}},options:D}});
        new Chart(document.getElementById('c4'),{{type:'line',data:{{labels:S,datasets:[{{label:'CumReward',data:R,borderColor:'#00e676',borderWidth:1.5,pointRadius:0,fill:true,backgroundColor:'rgba(0,230,118,0.08)'}}]}},options:D}});
    </script>
    '''
    display(HTML(html))
    return stats

print("Dashboard function defined ✓")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 14: Show Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
md("""### Generate Dashboard for PPO+CVML""")

code("""dashboard_stats = generate_dashboard(cvml_model, DATA_PATH, "PPO+CVML")""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 15: LLM Swarm (optional)
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 10. LLM Swarm (Optional)

Run the 5-persona LLM swarm on sample news events. **Requires `OPENROUTER_API_KEY`** above.

Skip this cell if you don't have an API key — the model will use random macro vectors instead.
""")

code("""# ── LLM Swarm (Optional — runs if OPENROUTER_API_KEY is set) ─────────────────
import asyncio, json
from litellm import acompletion
from sentence_transformers import SentenceTransformer

if OPENROUTER_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
    print("Running LLM Swarm to generate Macro Vectors...")
    
    PERSONAS = {
        "momentum": (
            "You are a momentum trader. Predict if initial price move continues. "
            "Respond ONLY with valid JSON: {'direction': 'up'|'down'|'neutral', 'magnitude': 0.0-1.0, 'confidence': 0.0-1.0, 'reasoning': '...'}"
        ),
        "mean_reversion": (
            "You are a mean-reversion trader. Predict if initial price move fades. "
            "Respond ONLY with valid JSON: {'direction': 'up'|'down'|'neutral', 'magnitude': 0.0-1.0, 'confidence': 0.0-1.0, 'reasoning': '...'}"
        ),
        "macro_risk": (
            "You are a macro risk analyst. Assess SYSTEMIC RISK flows. "
            "Respond ONLY with valid JSON: {'direction': 'up'|'down'|'neutral', 'magnitude': 0.0-1.0, 'confidence': 0.0-1.0, 'reasoning': '...'}"
        ),
        "liquidity": (
            "You are a liquidity analyst. Predict BID-ASK SPREAD effect ('up' = wider spreads). "
            "Respond ONLY with valid JSON: {'direction': 'up'|'down'|'neutral', 'magnitude': 0.0-1.0, 'confidence': 0.0-1.0, 'reasoning': '...'}"
        ),
        "volatility": (
            "You are a volatility trader. Predict REALIZED VOLATILITY effect. "
            "Respond ONLY with valid JSON: {'direction': 'up'|'down'|'neutral', 'magnitude': 0.0-1.0, 'confidence': 0.0-1.0, 'reasoning': '...'}"
        )
    }
    
    PERSONA_MODELS = {
        "momentum":       "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
        "mean_reversion": "openrouter/nvidia/nemotron-3-nano-30b-a3b:free",
        "macro_risk":     "openrouter/nvidia/nemotron-nano-9b-v2:free",
        "liquidity":      "openrouter/mistralai/mistral-small-3.1-24b-instruct:free",
        "volatility":     "openrouter/arcee-ai/trinity-large-preview:free",
    }
    
    async def _call_agent(name, prompt, text, model):
        try:
            resp = await acompletion(model=model, messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Analyze:\\n\\n{text}"}
            ], response_format={"type": "json_object"}, temperature=0.3, max_tokens=200)
            content = resp.choices[0].message.content
            import re
            js = json.loads(re.sub(r'```(?:json)?\s*', '', content).strip().rstrip('`').strip())
            return {"persona": name, "parsed": js, "error": None}
        except Exception as e:
            return {"persona": name, "parsed": None, "error": str(e)}

    async def run_swarm(event_text):
        tasks = [_call_agent(n, p, event_text, PERSONA_MODELS[n]) for n, p in PERSONAS.items()]
        return await asyncio.gather(*tasks)

    def aggregate_consensus(results):
        dir_map = {"up": 1.0, "down": -1.0, "neutral": 0.0}
        w_dir, t_conf = 0.0, 0.0
        mags, dirs, reas = [], [], []
        
        for r in results:
            if r["error"]: continue
            p = r["parsed"]
            conf = float(p.get("confidence", 0.5))
            d = dir_map.get(str(p.get("direction", "neutral")).lower(), 0.0)
            w_dir += d * conf; t_conf += conf
            mags.append(float(p.get("magnitude", 0)))
            dirs.append(d)
            reas.append(f"[{r['persona']}]: {p.get('reasoning', '')}")
            
        c_dir = (w_dir / t_conf) if t_conf > 0 else 0.0
        return {
            "combined_reasoning": " ".join(reas),
            "consensus_direction": c_dir,
        }

    # Generate synthetic news events (since GDELT parsing requires extra dataset setup)
    sample_events = [
        "Federal Reserve unexpectedly cuts interest rates by 50bps, citing inflation cooling faster than expected.",
        "Major tech earnings report shows 15% revenue miss across semiconductor sector, leading to massive selloff."
    ]
    
    macro_vectors = []
    print("Loading SentenceTransformer...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    for i, event in enumerate(sample_events):
        print(f"\\nProcessing Event {i+1}: {event[:80]}...")
        # Since notebooks run async loop already, we can use await directly in Kaggle/Jupyter
        swarm_res = await run_swarm(event)
        consensus = aggregate_consensus(swarm_res)
        emb = encoder.encode(consensus["combined_reasoning"], convert_to_numpy=True)
        macro_vectors.append(emb)
        print(f"  Swarm Consensus Dir: {consensus['consensus_direction']:.2f}")
    
    np.savez(os.path.join(DATA_DIR, "macro_vectors.npz"), embeddings=np.array(macro_vectors, dtype=np.float32))
    MACRO_VECTORS_PATH = os.path.join(DATA_DIR, "macro_vectors.npz")
    print(f"\\n✅ Embedded and saved macro vectors to {MACRO_VECTORS_PATH}")
else:
    print("No OPENROUTER_API_KEY set. The RL environment will default to using randomly generated macro vectors.")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 16: Scale Up Instructions
# ═══════════════════════════════════════════════════════════════════════════════
md("""## 🚀 Scale Up for Real Training

To run full-scale training, update the config cell at the top:

```python
TRAIN_TIMESTEPS = 500_000   # or 1_000_000 for thorough training
REWARD_TYPE = "log_return"  # best for HFT
TICKER = "aapl"             # try different tickers
```

Then re-run all cells. With a Kaggle GPU (T4/P100), 500K steps should take ~15-30 minutes.

### Tips:
- **Compare tickers**: Train on AAPL, MSFT, INTC to see how the model generalizes
- **Reward ablation**: Try `"sharpe"` and `"pnl"` to compare reward functions
- **Architecture ablation**: Compare PPO+CVML vs PPO+FlatMLP in the results table
""")


# ═══════════════════════════════════════════════════════════════════════════════
# Build the notebook
# ═══════════════════════════════════════════════════════════════════════════════
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "isGpuEnabled": True,
            "isInternetEnabled": True
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = os.path.join(os.path.dirname(__file__), "notebooks", "drl_trading_pipeline.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook saved to: {out_path}")
print(f"Cells: {len(cells)}")
