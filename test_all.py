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
    from micro.env import LOBReplayEnv, MACRO_DIM, canonical_lob_columns
    env = LOBReplayEnv(data_path=DATA_PATH)
    obs, info = env.reset()
    assert "lob" in obs, "Missing 'lob' key in obs"
    assert "macro" in obs, "Missing 'macro' key in obs"
    assert obs["lob"].shape == (40,), f"LOB shape: {obs['lob'].shape}"
    assert obs["macro"].shape == (MACRO_DIM,), f"Macro shape: {obs['macro'].shape}"
    # Canonical ordering: numeric level order, blocks match CVML's (4,10) reshape
    cols = canonical_lob_columns()
    assert env.feature_cols == cols
    assert cols[:3] == ["bid_price_1", "bid_price_2", "bid_price_3"], "level order must be numeric"
    assert cols[10] == "ask_price_1" and cols[20] == "bid_size_1" and cols[30] == "ask_size_1"
    print(f"  Obs keys: {list(obs.keys())}")
    print(f"  LOB shape: {obs['lob'].shape}, Macro shape: {obs['macro'].shape}")
    print(f"  Canonical column order: OK")

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

    from micro.env import MACRO_DIM

    obs_space = spaces.Dict({
        "lob": spaces.Box(low=0, high=np.inf, shape=(40,), dtype=np.float32),
        "macro": spaces.Box(low=-1.0, high=1.0, shape=(MACRO_DIM,), dtype=np.float32)
    })

    # Test HierarchicalFeatureExtractor (CVML)
    hfe = HierarchicalFeatureExtractor(obs_space)
    test_obs = {
        "lob": torch.randn(8, 40),
        "macro": torch.randn(8, MACRO_DIM)
    }
    out = hfe(test_obs)
    assert out.shape == (8, 64 + MACRO_DIM), f"Expected (8,{64+MACRO_DIM}), got {out.shape}"
    print(f"  HierarchicalFeatureExtractor: {out.shape} (64 CVML + {MACRO_DIM} macro)")

    # Test FlatMLPFeatureExtractor (ablation baseline)
    flat = FlatMLPFeatureExtractor(obs_space)
    out_flat = flat(test_obs)
    assert out_flat.shape == (8, 64 + MACRO_DIM), f"Expected (8,{64+MACRO_DIM}), got {out_flat.shape}"
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


# ── Test 9: Train/test windowing + random starts ─────────────────────────────
def test_windowing():
    from micro.env import LOBReplayEnv
    train_env = LOBReplayEnv(data_path=DATA_PATH, start_frac=0.0, end_frac=0.7,
                             random_start=True, max_steps=50)
    eval_env = LOBReplayEnv(data_path=DATA_PATH, start_frac=0.7, end_frac=1.0)

    starts = set()
    for _ in range(10):
        train_env.reset(seed=None)
        starts.add(train_env.current_step)
        assert train_env._window_lo <= train_env.current_step <= train_env._window_hi
    assert len(starts) > 1, "random_start should vary episode start rows"

    # Train window never touches eval rows; episodes respect max_steps
    obs, _ = train_env.reset()
    done, steps = False, 0
    while not done:
        obs, r, done, trunc, info = train_env.step(0)
        steps += 1
    assert steps <= 50, f"episode exceeded max_steps: {steps}"
    assert train_env.current_step <= train_env._window_hi <= eval_env._window_lo
    print(f"  {len(starts)} distinct random starts; episode len {steps} <= 50")
    print(f"  train window [{train_env._window_lo},{train_env._window_hi}] "
          f"| eval window [{eval_env._window_lo},{eval_env._window_hi}]")

# ── Test 10: Time-aligned macro vectors ──────────────────────────────────────
def test_macro_alignment():
    import numpy as np
    import pandas as pd
    import tempfile
    from micro.env import LOBReplayEnv, MACRO_DIM

    df = pd.read_csv(DATA_PATH)
    # Use known numeric timestamps regardless of what the source file has
    df["timestamp"] = 1_000_000 + np.arange(len(df)) * 60.0
    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "lob.csv")
        df.to_csv(data_path, index=False)

        ts = df["timestamp"].values
        # Two events: one before row 100, one before row 300
        rng = np.random.RandomState(0)
        npz_path = os.path.join(tmp, "macro.npz")
        np.savez(npz_path,
                 timestamps=np.array([ts[100] - 1, ts[300] - 1], dtype=np.float64),
                 scalars=np.array([[0.5, 0.3, 0.8, 0.9], [-0.7, 0.6, 0.7, 0.5]], dtype=np.float32),
                 embeddings=rng.randn(2, 384).astype(np.float32))

        env = LOBReplayEnv(data_path=data_path, macro_vectors_path=npz_path)
        assert env._macro_by_row is not None, "aligned macro vectors not loaded"
        assert env._macro_by_row.shape == (len(df), MACRO_DIM)
        # Before first event: zeros. After: scalars prefix must match the event.
        assert np.all(env._macro_by_row[50] == 0), "rows before first event must be zeros"
        assert abs(env._macro_by_row[150][0] - 0.5) < 1e-6, "row 150 should see event 1"
        assert abs(env._macro_by_row[350][0] - -0.7) < 1e-6, "row 350 should see event 2"
        # zero_macro ablation keeps shape but zeroes content
        env_z = LOBReplayEnv(data_path=data_path, macro_vectors_path=npz_path, zero_macro=True)
        obs, _ = env_z.reset()
        assert np.all(obs["macro"] == 0)
    print(f"  Alignment: zeros before first event, correct event per row, "
          f"zero_macro ablation OK")

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
    run_test("9. Train/Test Windowing", test_windowing)
    run_test("10. Time-Aligned Macro", test_macro_alignment)

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
