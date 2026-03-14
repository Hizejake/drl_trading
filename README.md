# Hierarchical Dual-Frequency DRL Trading Bot

A deep reinforcement learning trading system that combines **macro-level LLM analysis** with **micro-level order book execution** using a hierarchical dual-frequency architecture.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hierarchical DRL Agent                       │
│                                                                 │
│  ┌──────────────────────┐    ┌──────────────────────────────┐  │
│  │   MACRO LAYER        │    │   MICRO LAYER                │  │
│  │   (Low Frequency)    │    │   (High Frequency)           │  │
│  │                      │    │                              │  │
│  │  GDELT News Events   │    │  LOBSTER LOB Tick Data       │  │
│  │       ↓              │    │       ↓                      │  │
│  │  5-Persona LLM Swarm │    │  10-Level Order Book         │  │
│  │  (OpenRouter API)    │    │       ↓                      │  │
│  │       ↓              │    │  CVML Feature Extractor      │  │
│  │  Consensus Voting    │    │  (Conv2D Depthwise+Pointwise)│  │
│  │       ↓              │    │       ↓                      │  │
│  │  384D Semantic Embed │    │  64D Microstructure Features  │  │
│  │  → 128D Projection   │    │                              │  │
│  └──────────┬───────────┘    └──────────────┬───────────────┘  │
│             │                               │                   │
│             └───────────┬───────────────────┘                   │
│                         ↓                                       │
│              192D Combined Feature Vector                       │
│                         ↓                                       │
│                PPO Policy Network                               │
│              [256 → 128 → 64] MLP                               │
│                         ↓                                       │
│           Action: Hold | Buy | Sell | LimitBuy | LimitSell      │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Real NASDAQ LOB Data**: Parses LOBSTER tick-level order book data (AAPL, AMZN, GOOG, INTC, MSFT)
- **LLM Swarm**: 5 concurrent LLM personas (Momentum, Mean-Reversion, Macro Risk, Liquidity, Volatility) via OpenRouter
- **CVML**: Convolutional Cross-Variate Mixing Layer for microstructure feature extraction
- **3 Reward Functions**: Log-return, rolling Sharpe ratio, PnL delta — with inventory penalty
- **Market Frictions**: Taker fees (3bps), maker rebates (1bps), probabilistic limit order fills
- **Baselines**: TWAP, VWAP, Random Agent, Flat MLP ablation
- **Interactive Dashboard**: Self-contained HTML backtest visualization with Chart.js

## Project Structure

```
drl_trading/
├── data/
│   ├── download_gdelt.py       # GDELT news data scraper
│   ├── download_lob.py         # LOBSTER LOB data parser
│   └── raw/                    # Data files (user downloads)
├── macro/
│   ├── swarm.py                # LLM swarm (5 personas via OpenRouter)
│   └── run_swarm_batch.py      # Batch processing for GDELT events
├── micro/
│   ├── env.py                  # LOBReplayEnv (Gymnasium)
│   ├── cvml.py                 # CVML feature extractor (PyTorch)
│   └── policy.py               # SB3 custom feature extractors
├── notebooks/
│   ├── models/                 # Saved model checkpoints
│   └── backtest_dashboard.html # Generated dashboard
├── train.py                    # Training pipeline + baselines
├── dashboard.py                # Dashboard generator
├── test_all.py                 # Comprehensive test suite
├── requirements.txt
└── .env                        # API keys (not tracked)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup API Keys

Create a `.env` file:
```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### 3. Download LOB Data

Download LOBSTER sample files from [lobsterdata.com](https://lobsterdata.com/info/DataSamples.php) and place the extracted CSVs in `data/raw/`.

Then parse them:
```bash
python data/download_lob.py
```

### 4. Download News Data

```bash
python data/download_gdelt.py
```

### 5. Run LLM Swarm (Optional)

Process GDELT news through the LLM swarm to generate macro vectors:
```bash
python macro/run_swarm_batch.py --max-events 20
```

### 6. Train

```bash
# Train on AAPL LOBSTER data (500K timesteps)
python train.py --data aapl --timesteps 500000

# Quick test
python train.py --data synthetic --timesteps 10000
```

### 7. Generate Dashboard

```bash
python dashboard.py --data aapl
# Opens: notebooks/backtest_dashboard.html
```

### 8. Run Tests

```bash
python test_all.py
```

## Evaluation Baselines

| Model | Description |
|-------|-------------|
| **PPO+CVML** | Full model: CVML microstructure + LLM macro vectors |
| **PPO+FlatMLP** | Ablation: flat MLP instead of CVML |
| **TWAP** | Time-Weighted Average Price execution |
| **VWAP** | Volume-Weighted Average Price execution |
| **Random** | Uniform random action selection |

## Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| [LOBSTER](https://lobsterdata.com) | NASDAQ tick-level LOB (10 levels) | Free samples |
| [GDELT](https://www.gdeltproject.org) | Global news event database (15-min updates) | Free / Public |
| [OpenRouter](https://openrouter.ai) | Multi-model LLM API | Free tier available |

## Tech Stack

- **RL**: Stable Baselines 3, Gymnasium
- **Neural Networks**: PyTorch
- **NLP**: LiteLLM, Sentence Transformers
- **Data**: Pandas, NumPy
- **Visualization**: Chart.js (embedded in HTML)

## License

MIT
