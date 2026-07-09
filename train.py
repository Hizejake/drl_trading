"""
Training & Evaluation Pipeline for the Hierarchical DRL Trading Bot.

Supports:
- PPO training with CVML + Macro feature extractor (full model)
- Macro ablation: same architecture with the macro vector zeroed
- PPO training with flat MLP (architecture ablation)
- Baselines: Buy & Hold, TWAP, VWAP, Random Agent
- Train/test split: models train on the first TRAIN_FRAC of the data
  (random episode starts) and are evaluated out-of-sample on the rest
- Hyperparameter configuration, LR scheduling, metrics logging, checkpoints

Usage:
    python train.py                          # Train on AAPL data (500K steps)
    python train.py --timesteps 10000        # Quick smoke train
    python train.py --data synthetic         # Use synthetic data
    python train.py --eval-only              # Evaluate saved models only
"""

import os
import sys
import time
import argparse
import csv
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback
)

from micro.env import LOBReplayEnv
from micro.policy import HierarchicalFeatureExtractor, FlatMLPFeatureExtractor

# Fix Windows OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "notebooks", "models")
LOGS_DIR = os.path.join(BASE_DIR, "notebooks")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_PATHS = {
    "aapl": os.path.join(BASE_DIR, "data", "raw", "lobster_aapl_10_level.csv"),
    "amzn": os.path.join(BASE_DIR, "data", "raw", "lobster_amzn_10_level.csv"),
    "goog": os.path.join(BASE_DIR, "data", "raw", "lobster_goog_10_level.csv"),
    "intc": os.path.join(BASE_DIR, "data", "raw", "lobster_intc_10_level.csv"),
    "msft": os.path.join(BASE_DIR, "data", "raw", "lobster_msft_10_level.csv"),
    "synthetic": os.path.join(BASE_DIR, "data", "raw", "synthetic_lob_10_level.csv"),
}

MACRO_VECTORS_PATH = os.path.join(BASE_DIR, "data", "raw", "macro_vectors.npz")

# ── Train/test split ─────────────────────────────────────────────────────────
TRAIN_FRAC = 0.7          # first 70% of rows = training window
TRAIN_EPISODE_STEPS = 256  # episode length during training (random starts)

# ── Hyperparameters ───────────────────────────────────────────────────────────
HYPERPARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "clip_range": 0.2,
}


def make_env(data_path, macro_path, reward_type, train=True, zero_macro=False,
             max_steps=None):
    """Build a windowed env: train = first TRAIN_FRAC with random starts,
    eval = held-out tail replayed from its beginning."""
    if train:
        return LOBReplayEnv(
            data_path=data_path,
            use_macro_vector=True,
            macro_vectors_path=macro_path,
            zero_macro=zero_macro,
            reward_type=reward_type,
            inventory_penalty=0.001,
            max_inventory=50,
            invalid_action_penalty=0.01,
            start_frac=0.0, end_frac=TRAIN_FRAC,
            random_start=True,
            max_steps=TRAIN_EPISODE_STEPS,
        )
    return LOBReplayEnv(
        data_path=data_path,
        use_macro_vector=True,
        macro_vectors_path=macro_path,
        zero_macro=zero_macro,
        reward_type=reward_type,
        inventory_penalty=0.001,
        max_inventory=50,
        invalid_action_penalty=0.01,
        start_frac=TRAIN_FRAC, end_frac=1.0,
        random_start=False,
        max_steps=max_steps,
    )


# ── Custom Callbacks ──────────────────────────────────────────────────────────
class MetricsLoggerCallback(BaseCallback):
    """Logs training metrics to a CSV file after each rollout."""

    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards = []
        self.episode_lengths = []
        self._csv_initialized = False

    def _init_callback(self):
        if not self._csv_initialized:
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestep", "episodes", "mean_reward", "mean_length",
                    "entropy_loss", "policy_loss", "value_loss", "learning_rate"
                ])
            self._csv_initialized = True

    def _on_step(self):
        # Collect episode info from the monitor
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])
        return True

    def _on_rollout_end(self):
        if self.episode_rewards:
            logger = self.model.logger

            row = [
                self.num_timesteps,
                len(self.episode_rewards),
                np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0,
                np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0,
                logger.name_to_value.get("train/entropy_loss", 0),
                logger.name_to_value.get("train/policy_gradient_loss", 0),
                logger.name_to_value.get("train/value_loss", 0),
                self.model.lr_schedule(self.model._current_progress_remaining),
            ]

            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)


def linear_schedule(initial_lr):
    """Returns a callable that decays learning rate linearly from initial_lr to 0."""
    def schedule(progress_remaining):
        return progress_remaining * initial_lr
    return schedule


# ── Training ──────────────────────────────────────────────────────────────────
def train_ppo(data_path, model_name, use_cvml=True, zero_macro=False,
              total_timesteps=500000, reward_type="log_return"):
    """Train a PPO agent. Variants:
    - ppo_cvml:         CVML extractor + time-aligned macro vector
    - ppo_cvml_nomacro: CVML extractor, macro zeroed (macro ablation)
    - ppo_flat:         flat MLP extractor + macro (architecture ablation)
    """
    print(f"\n{'='*70}")
    print(f"TRAINING: {model_name.upper()} | {total_timesteps:,} timesteps | reward={reward_type}")
    print(f"Data: {os.path.basename(data_path)} (train window: first {TRAIN_FRAC:.0%})")
    print(f"{'='*70}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing LOB data at {data_path}")

    # Check for pre-computed macro vectors
    macro_path = MACRO_VECTORS_PATH if os.path.exists(MACRO_VECTORS_PATH) else None

    env = make_env(data_path, macro_path, reward_type, train=True, zero_macro=zero_macro)

    # Feature extractor selection
    extractor_class = HierarchicalFeatureExtractor if use_cvml else FlatMLPFeatureExtractor
    policy_kwargs = {
        "features_extractor_class": extractor_class,
        "features_extractor_kwargs": {},
        "net_arch": [256, 128, 64],
    }

    # Initialize PPO with linear LR decay
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=linear_schedule(HYPERPARAMS["learning_rate"]),
        n_steps=HYPERPARAMS["n_steps"],
        batch_size=HYPERPARAMS["batch_size"],
        n_epochs=HYPERPARAMS["n_epochs"],
        gamma=HYPERPARAMS["gamma"],
        gae_lambda=HYPERPARAMS["gae_lambda"],
        ent_coef=HYPERPARAMS["ent_coef"],
        vf_coef=HYPERPARAMS["vf_coef"],
        max_grad_norm=HYPERPARAMS["max_grad_norm"],
        clip_range=HYPERPARAMS["clip_range"],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Callbacks
    log_path = os.path.join(LOGS_DIR, f"training_log_{model_name}.csv")
    callbacks = [
        CheckpointCallback(
            save_freq=50000,
            save_path=MODELS_DIR,
            name_prefix=model_name,
        ),
        MetricsLoggerCallback(log_path=log_path),
    ]

    # Train
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    elapsed = time.time() - start_time

    # Save final model
    final_path = os.path.join(MODELS_DIR, f"{model_name}_final")
    model.save(final_path)
    print(f"\nModel saved to {final_path}")
    print(f"Training time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Metrics log: {log_path}")

    return model


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_agent(model, env, name="Agent"):
    """Run a full episode evaluation and return stats."""
    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if truncated:
            break

    stats = env.get_episode_stats()
    stats["total_reward"] = round(total_reward, 4)
    stats["name"] = name
    return stats


def run_buy_hold_baseline(env):
    """Buy & Hold: accumulate while cash allows early on, then hold to the end
    (final inventory is marked to market — no forced flatten)."""
    obs, _ = env.reset()
    done = False

    while not done:
        row = env.df.iloc[env.current_step]
        can_buy = env.cash >= float(row["ask_price_1"]) * 1.001 and env.inventory < env.max_inventory
        action = 1 if can_buy else 0
        obs, reward, done, truncated, info = env.step(action)
        if truncated:
            break

    stats = env.get_episode_stats()
    stats["name"] = "Buy&Hold"
    return stats


def run_twap_baseline(env):
    """TWAP baseline: Buy 1 unit every N steps, then sell all at end."""
    obs, _ = env.reset()
    done = False
    steps = 0
    buy_interval = 50  # Buy every 50 steps
    horizon = env._window_hi - env.current_step

    while not done:
        if steps % buy_interval == 0 and steps < horizon - 100 and env.inventory < env.max_inventory:
            action = 1  # Market Buy
        elif steps >= horizon - 50 and env.inventory > 0:
            action = 2  # Market Sell to flatten
        else:
            action = 0  # Hold

        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        if truncated:
            break

    stats = env.get_episode_stats()
    stats["name"] = "TWAP"
    return stats


def run_vwap_baseline(env):
    """VWAP baseline: Scale order size by volume at each level."""
    obs, _ = env.reset()
    done = False
    steps = 0
    total_volume_seen = 0
    horizon = env._window_hi - env.current_step

    while not done:
        row = env.df.iloc[env.current_step]
        bid_vol = sum(float(row.get(f"bid_size_{i}", 0)) for i in range(1, 6))
        ask_vol = sum(float(row.get(f"ask_size_{i}", 0)) for i in range(1, 6))
        total_vol = bid_vol + ask_vol
        total_volume_seen += total_vol

        # VWAP logic: buy when volume is high relative to average
        avg_vol = total_volume_seen / max(steps + 1, 1)

        if total_vol > avg_vol * 1.2 and steps < horizon - 100 and env.inventory < env.max_inventory:
            action = 1  # Market Buy on high volume
        elif steps >= horizon - 50 and env.inventory > 0:
            action = 2  # Flatten
        elif total_vol < avg_vol * 0.8 and env.inventory > 0:
            action = 2  # Sell on low volume
        else:
            action = 0  # Hold

        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        if truncated:
            break

    stats = env.get_episode_stats()
    stats["name"] = "VWAP"
    return stats


def run_random_baseline(env, seed=42):
    """Random agent baseline: Uniform random action selection."""
    obs, _ = env.reset()
    done = False
    rng = np.random.RandomState(seed)

    while not done:
        action = rng.randint(0, 5)
        obs, reward, done, truncated, info = env.step(action)
        if truncated:
            break

    stats = env.get_episode_stats()
    stats["name"] = "Random"
    return stats


def print_comparison_table(all_stats):
    """Print a formatted comparison table of all models/baselines."""
    print(f"\n{'='*100}")
    print(f"{'OUT-OF-SAMPLE EVALUATION (held-out last ' + str(round((1-TRAIN_FRAC)*100)) + '% of data)':^100}")
    print(f"{'='*100}")

    headers = ["Model", "Return%", "Sharpe", "MaxDD%", "Trades", "Buy", "Sell", "Final PV", "Inventory"]
    widths = [18, 10, 10, 10, 8, 6, 6, 12, 10]

    header_line = ""
    for h, w in zip(headers, widths):
        header_line += f"{h:>{w}}"
    print(header_line)
    print("-" * 100)

    for s in all_stats:
        row = (
            f"{s['name']:>18}"
            f"{s['total_return_pct']:>10.4f}"
            f"{s['sharpe_ratio']:>10.4f}"
            f"{s['max_drawdown_pct']:>10.4f}"
            f"{s['total_trades']:>8}"
            f"{s['buy_trades']:>6}"
            f"{s['sell_trades']:>6}"
            f"  ${s['final_pv']:>9.2f}"
            f"{s['final_inventory']:>10}"
        )
        print(row)

    print(f"{'='*100}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRL Trading Bot Training Pipeline")
    parser.add_argument("--data", default="aapl", choices=list(DATA_PATHS.keys()),
                        help="Dataset to use (default: aapl)")
    parser.add_argument("--timesteps", type=int, default=500000,
                        help="Total training timesteps (default: 500000)")
    parser.add_argument("--reward", default="log_return",
                        choices=["log_return", "sharpe", "pnl"],
                        help="Reward function type (default: log_return)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate existing models")
    parser.add_argument("--skip-flat", action="store_true",
                        help="Skip flat MLP ablation training")
    parser.add_argument("--skip-nomacro", action="store_true",
                        help="Skip macro-zeroed ablation training")
    parser.add_argument("--eval-ticks", type=int, default=None,
                        help="Max steps per evaluation episode (default: full test window)")
    args = parser.parse_args()

    data_path = DATA_PATHS[args.data]

    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        print(f"Available datasets: {[k for k, v in DATA_PATHS.items() if os.path.exists(v)]}")
        sys.exit(1)

    macro_path = MACRO_VECTORS_PATH if os.path.exists(MACRO_VECTORS_PATH) else None

    if not args.eval_only:
        # Full model: CVML + time-aligned macro
        train_ppo(data_path, "ppo_cvml", use_cvml=True,
                  total_timesteps=args.timesteps, reward_type=args.reward)

        # Macro ablation: identical architecture, macro zeroed
        if not args.skip_nomacro:
            train_ppo(data_path, "ppo_cvml_nomacro", use_cvml=True, zero_macro=True,
                      total_timesteps=args.timesteps, reward_type=args.reward)

        # Architecture ablation: flat MLP instead of CVML
        if not args.skip_flat:
            train_ppo(data_path, "ppo_flat", use_cvml=False,
                      total_timesteps=args.timesteps, reward_type=args.reward)

    # ── Evaluate all models & baselines out-of-sample ──────────────────────
    print(f"\n\nRunning out-of-sample evaluation (last {round((1-TRAIN_FRAC)*100)}% of data)...")
    all_stats = []

    model_specs = [
        ("ppo_cvml_final.zip", "PPO+CVML+Macro", {"zero_macro": False}),
        ("ppo_cvml_nomacro_final.zip", "PPO+CVML", {"zero_macro": True}),
        ("ppo_flat_final.zip", "PPO+FlatMLP", {"zero_macro": False}),
    ]
    for fname, label, env_kwargs in model_specs:
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            continue
        env = make_env(data_path, macro_path, args.reward, train=False,
                       max_steps=args.eval_ticks, **env_kwargs)
        model = PPO.load(path, env=env)
        all_stats.append(evaluate_agent(model, env, label))

    # Baselines (each gets a fresh env over the same test window)
    for runner in (run_buy_hold_baseline, run_twap_baseline, run_vwap_baseline, run_random_baseline):
        env = make_env(data_path, macro_path, args.reward, train=False,
                       max_steps=args.eval_ticks)
        all_stats.append(runner(env))

    print_comparison_table(all_stats)
