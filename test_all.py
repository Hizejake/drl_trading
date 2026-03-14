"""
Comprehensive test script for all DRL Trading Bot components.
Run: python test_all.py
"""
import os
import sys
import traceback

# Fix OpenMP duplicate library issue on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "synthetic_lob_10_level.csv")

results = {}

def run_test(name, fn):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        fn()
        results[name] = "PASS"
        print(f"  >>> PASS")
    except Exception as e:
        results[name] = f"FAIL: {e}"
        print(f"  >>> FAIL: {e}")
        traceback.print_exc()

# ── Test 1: Data files exist ─────────────────────────────────────────────────
def test_data_files():
    assert os.path.exists(DATA_PATH), f"Missing: {DATA_PATH}"
    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    assert len(df) == 5000, f"Expected 5000 rows, got {len(df)}"
    expected_cols = ['bid_price_1', 'ask_price_1', 'bid_size_1', 'ask_size_1']
    for c in expected_cols:
        assert c in df.columns, f"Missing column: {c}"
    price_cols = [c for c in df.columns if 'price' in c]
    size_cols = [c for c in df.columns if 'size' in c]
    assert len(price_cols) == 20, f"Expected 20 price cols, got {len(price_cols)}"
    assert len(size_cols) == 20, f"Expected 20 size cols, got {len(size_cols)}"
    print(f"  Data: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Price cols: {len(price_cols)}, Size cols: {len(size_cols)}")

# ── Test 2: CVML Module ──────────────────────────────────────────────────────
def test_cvml():
    import torch
    from micro.cvml import CVML
    model = CVML()
    x = torch.randn(32, 40)
    out = model(x)
    assert out.shape == (32, 64), f"Expected (32,64), got {out.shape}"
    # Test single sample
    x1 = torch.randn(1, 40)
    out1 = model(x1)
    assert out1.shape == (1, 64), f"Expected (1,64), got {out1.shape}"
    # Test gradient flow
    loss = out.sum()
    loss.backward()
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    print(f"  Gradient flow: OK (backward pass completed)")

# ── Test 3: LOBReplayEnv ─────────────────────────────────────────────────────
def test_env():
    from micro.env import LOBReplayEnv
    env = LOBReplayEnv(data_path=DATA_PATH)
    obs, info = env.reset()
    assert "lob" in obs, "Missing 'lob' key in obs"
    assert "macro" in obs, "Missing 'macro' key in obs"
    assert obs["lob"].shape == (40,), f"LOB shape: {obs['lob'].shape}"
    assert obs["macro"].shape == (128,), f"Macro shape: {obs['macro'].shape}"
    print(f"  Obs keys: {list(obs.keys())}")
    print(f"  LOB shape: {obs['lob'].shape}, Macro shape: {obs['macro'].shape}")

    # Test all 5 actions
    actions = {0: 'Hold', 1: 'Market Buy', 2: 'Market Sell', 3: 'Limit Buy', 4: 'Limit Sell'}
    for a, name in actions.items():
        obs, reward, done, truncated, info = env.step(a)
        print(f"  Action {a} ({name}): reward={reward:.4f}, portfolio=${info['portfolio_value']:.2f}, inv={info['inventory']}")
    
    # Test episode completion
    env.reset()
    steps = 0
    done = False
    while not done:
        obs, reward, done, truncated, info = env.step(0)
        steps += 1
    print(f"  Full episode: {steps} steps, final portfolio=${info['portfolio_value']:.2f}")

# ── Test 4: Policy / Feature Extractor ────────────────────────────────────────
def test_policy():
    import torch
    from gymnasium import spaces
    import numpy as np
    from micro.policy import HierarchicalFeatureExtractor, FlatMLPFeatureExtractor

    obs_space = spaces.Dict({
        "lob": spaces.Box(low=0, high=np.inf, shape=(40,), dtype=np.float32),
        "macro": spaces.Box(low=-1.0, high=1.0, shape=(128,), dtype=np.float32)
    })

    # Test HierarchicalFeatureExtractor (CVML)
    hfe = HierarchicalFeatureExtractor(obs_space)
    test_obs = {
        "lob": torch.randn(8, 40),
        "macro": torch.randn(8, 128)
    }
    out = hfe(test_obs)
    assert out.shape == (8, 192), f"Expected (8,192), got {out.shape}"
    print(f"  HierarchicalFeatureExtractor: {out.shape} (64 CVML + 128 macro)")

    # Test FlatMLPFeatureExtractor (ablation baseline)
    flat = FlatMLPFeatureExtractor(obs_space)
    out_flat = flat(test_obs)
    assert out_flat.shape == (8, 192), f"Expected (8,192), got {out_flat.shape}"
    print(f"  FlatMLPFeatureExtractor: {out_flat.shape}")

# ── Test 5: Swarm module imports / consensus logic ────────────────────────────
def test_swarm_offline():
    """Test the swarm's consensus logic without calling LLM APIs."""
    import numpy as np
    # We import the module but skip the actual API calls
    # Instead test aggregate_consensus with mock data
    sys.path.insert(0, BASE_DIR)
    from macro.swarm import aggregate_consensus

    mock_results = [
        {"persona": "momentum", "model": "test", "parsed": {"direction": "up", "magnitude": 0.7, "confidence": 0.8, "reasoning": "Strong uptrend"}, "error": None},
        {"persona": "mean_reversion", "model": "test", "parsed": {"direction": "down", "magnitude": 0.3, "confidence": 0.6, "reasoning": "Mean revert expected"}, "error": None},
        {"persona": "macro_risk", "model": "test", "parsed": {"direction": "up", "magnitude": 0.5, "confidence": 0.7, "reasoning": "Low systemic risk"}, "error": None},
        {"persona": "liquidity", "model": "test", "parsed": {"direction": "neutral", "magnitude": 0.2, "confidence": 0.5, "reasoning": "Spreads stable"}, "error": None},
        {"persona": "volatility", "model": "test", "parsed": {"direction": "up", "magnitude": 0.6, "confidence": 0.9, "reasoning": "Vol spike expected"}, "error": None},
    ]

    consensus = aggregate_consensus(mock_results)
    assert "consensus_direction" in consensus
    assert "avg_magnitude" in consensus
    assert "avg_confidence" in consensus
    assert "agreement_score" in consensus
    assert "combined_reasoning" in consensus
    print(f"  Consensus direction: {consensus['consensus_direction']}")
    print(f"  Avg magnitude: {consensus['avg_magnitude']}")
    print(f"  Avg confidence: {consensus['avg_confidence']}")
    print(f"  Agreement score: {consensus['agreement_score']}")
    print(f"  Combined reasoning length: {len(consensus['combined_reasoning'])} chars")

# ── Test 6: SentenceTransformer Encoding ──────────────────────────────────────
def test_semantic_encoder():
    from macro.swarm import SemanticEncoder
    encoder = SemanticEncoder()
    vec = encoder.encode("Apple earnings beat expectations, revenue slightly misses.")
    assert vec.shape == (384,), f"Expected (384,), got {vec.shape}"
    assert vec.dtype.name == "float32"
    print(f"  Embedding shape: {vec.shape}, dtype: {vec.dtype}")
    print(f"  First 5 dims: {vec[:5]}")

# ── Test 7: PPO + CVML Integration (quick sanity) ────────────────────────────
def test_ppo_integration():
    import torch
    from stable_baselines3 import PPO
    from micro.env import LOBReplayEnv
    from micro.policy import HierarchicalFeatureExtractor

    env = LOBReplayEnv(data_path=DATA_PATH, use_macro_vector=True)
    policy_kwargs = {
        "features_extractor_class": HierarchicalFeatureExtractor,
        "features_extractor_kwargs": {},
        "net_arch": [128, 64]
    }
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        n_steps=64,
        batch_size=32,
        device="cpu"
    )
    # Quick 64-step learn just to test the pipeline compiles
    model.learn(total_timesteps=64)
    print(f"  PPO+CVML 64-step micro-train: OK")

    # Test prediction
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    print(f"  Prediction: action={action}")

# ── Test 8: Load saved model ──────────────────────────────────────────────────
def test_load_saved_model():
    from stable_baselines3 import PPO
    from micro.env import LOBReplayEnv
    
    model_path = os.path.join(BASE_DIR, "notebooks", "models", "ppo_cvml_final.zip")
    if not os.path.exists(model_path):
        print(f"  SKIP: No saved model at {model_path}")
        return
    
    env = LOBReplayEnv(data_path=DATA_PATH, use_macro_vector=True)
    model = PPO.load(model_path, env=env)
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    print(f"  Loaded model prediction: action={action}")
    
    # Quick evaluation
    total_reward = 0.0
    done = False
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if truncated:
            break
    print(f"  Eval: {steps} steps, total_reward={total_reward:.4f}, final_pv=${info['portfolio_value']:.2f}")


# ── Run all tests ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_test("1. Data Files", test_data_files)
    run_test("2. CVML Module", test_cvml)
    run_test("3. LOBReplayEnv", test_env)
    run_test("4. Policy/Feature Extractors", test_policy)
    run_test("5. Swarm Consensus (offline)", test_swarm_offline)
    run_test("6. Semantic Encoder", test_semantic_encoder)
    run_test("7. PPO+CVML Integration", test_ppo_integration)
    run_test("8. Load Saved Model", test_load_saved_model)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for v in results.values() if v == "PASS")
    total = len(results)
    for name, status in results.items():
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon} {name}: {status}")
    print(f"\n  {passed}/{total} tests passed")
    if passed == total:
        print("  🎉 All tests passed!")
    else:
        print("  ⚠️  Some tests failed. See details above.")
