"""
Backtest Dashboard Generator — Creates an interactive HTML dashboard.

Runs a model evaluation episode, collects per-tick data, and generates
a self-contained HTML file with Chart.js for visualization.

Charts:
- Portfolio Value (PnL) curve over time
- Inventory position over time
- Mid-price with buy/sell trade markers
- Metrics summary: return, Sharpe, max drawdown, trades, win rate

Usage:
    python dashboard.py                          # Default: evaluate PPO+CVML on AAPL
    python dashboard.py --data aapl              # Specify dataset
    python dashboard.py --model notebooks/models/ppo_cvml_final
"""

import os
import sys
import json
import argparse
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from stable_baselines3 import PPO
from micro.env import LOBReplayEnv

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "notebooks", "models")

DATA_PATHS = {
    "aapl": os.path.join(BASE_DIR, "data", "raw", "lobster_aapl_10_level.csv"),
    "amzn": os.path.join(BASE_DIR, "data", "raw", "lobster_amzn_10_level.csv"),
    "goog": os.path.join(BASE_DIR, "data", "raw", "lobster_goog_10_level.csv"),
    "intc": os.path.join(BASE_DIR, "data", "raw", "lobster_intc_10_level.csv"),
    "msft": os.path.join(BASE_DIR, "data", "raw", "lobster_msft_10_level.csv"),
    "synthetic": os.path.join(BASE_DIR, "data", "raw", "synthetic_lob_10_level.csv"),
}

MACRO_VECTORS_PATH = os.path.join(BASE_DIR, "data", "raw", "macro_vectors.npz")


def collect_episode_data(model, env, max_ticks=None):
    """Run a full evaluation episode and collect per-tick data."""
    obs, _ = env.reset()
    
    ticks = []
    done = False
    step = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        ticks.append({
            "step": step,
            "mid_price": info["mid_price"],
            "portfolio_value": info["portfolio_value"],
            "inventory": info["inventory"],
            "action": int(action),
            "reward": float(reward),
            "trade_executed": info["trade_executed"],
        })
        
        step += 1
        if truncated:
            break
        if max_ticks and step >= max_ticks:
            break
    
    stats = env.get_episode_stats()
    return ticks, stats, env.trades


def generate_html(ticks, stats, trades, output_path, model_name="PPO+CVML", dataset_name="AAPL"):
    """Generate a self-contained interactive HTML dashboard."""
    
    ticks_display = ticks
    
    # Prepare data arrays
    steps = [t["step"] for t in ticks_display]
    mid_prices = [round(t["mid_price"], 4) for t in ticks_display]
    portfolio_values = [round(t["portfolio_value"], 2) for t in ticks_display]
    inventory = [t["inventory"] for t in ticks_display]
    rewards = [round(t["reward"], 6) for t in ticks_display]
    
    # Trade markers
    buy_trades = [{"x": t["step"], "y": t["price"]} for t in trades if t["side"] in ("buy", "limit_buy")]
    sell_trades = [{"x": t["step"], "y": t["price"]} for t in trades if t["side"] in ("sell", "limit_sell")]
    
    # Cumulative reward
    cum_rewards = []
    total = 0
    for t in ticks_display:
        total += t["reward"]
        cum_rewards.append(round(total, 4))
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DRL Trading Bot — Backtest Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0a0a0f;
            color: #e0e0e8;
            min-height: 100vh;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 28px 40px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }}
        
        .header h1 {{
            font-size: 26px;
            font-weight: 700;
            background: linear-gradient(135deg, #00d2ff, #7b2ff7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 6px;
        }}
        
        .header .subtitle {{
            font-size: 14px;
            color: #8888a0;
            font-weight: 400;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            padding: 24px 40px;
        }}
        
        .metric-card {{
            background: linear-gradient(145deg, #1a1a2e, #141421);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 20px 22px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        }}
        
        .metric-label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: #6666a0;
            margin-bottom: 8px;
            font-weight: 600;
        }}
        
        .metric-value {{
            font-size: 26px;
            font-weight: 700;
        }}
        
        .metric-value.positive {{ color: #00e676; }}
        .metric-value.negative {{ color: #ff5252; }}
        .metric-value.neutral {{ color: #e0e0e8; }}
        
        .charts-container {{
            padding: 16px 40px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        
        .chart-card {{
            background: linear-gradient(145deg, #1a1a2e, #141421);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 24px;
        }}
        
        .chart-card h3 {{
            font-size: 15px;
            font-weight: 600;
            color: #a0a0c0;
            margin-bottom: 16px;
        }}
        
        canvas {{
            width: 100% !important;
            max-height: 300px;
        }}
        
        .footer {{
            text-align: center;
            padding: 24px;
            color: #444;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>DRL Trading Bot — Backtest Dashboard</h1>
        <div class="subtitle">Model: {model_name} | Dataset: {dataset_name} LOBSTER | {len(ticks):,} ticks evaluated</div>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Total Return</div>
            <div class="metric-value {'positive' if stats['total_return_pct'] >= 0 else 'negative'}">{stats['total_return_pct']:+.4f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Final Portfolio</div>
            <div class="metric-value neutral">${stats['final_pv']:,.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value {'positive' if stats['sharpe_ratio'] >= 0 else 'negative'}">{stats['sharpe_ratio']:+.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value negative">{stats['max_drawdown_pct']:.4f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Trades</div>
            <div class="metric-value neutral">{stats['total_trades']:,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Buys / Sells</div>
            <div class="metric-value neutral">{stats['buy_trades']} / {stats['sell_trades']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Final Inventory</div>
            <div class="metric-value {'negative' if abs(stats['final_inventory']) > 5 else 'neutral'}">{stats['final_inventory']}</div>
        </div>
    </div>
    
    <div class="charts-container">
        <div class="chart-card">
            <h3>Portfolio Value (PnL Curve)</h3>
            <canvas id="pvChart"></canvas>
        </div>
        <div class="chart-card">
            <h3>Mid Price with Trade Markers</h3>
            <canvas id="priceChart"></canvas>
        </div>
        <div class="chart-card">
            <h3>Inventory Position</h3>
            <canvas id="invChart"></canvas>
        </div>
        <div class="chart-card">
            <h3>Cumulative Reward</h3>
            <canvas id="rewardChart"></canvas>
        </div>
    </div>
    
    <div class="footer">
        Generated by DRL Trading Bot | Hierarchical Dual-Frequency Architecture
    </div>

    <script>
        const steps = {json.dumps(steps)};
        const midPrices = {json.dumps(mid_prices)};
        const portfolioValues = {json.dumps(portfolio_values)};
        const inventoryData = {json.dumps(inventory)};
        const cumRewards = {json.dumps(cum_rewards)};
        const buyTrades = {json.dumps(buy_trades)};
        const sellTrades = {json.dumps(sell_trades)};
        
        const chartDefaults = {{
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {{
                legend: {{
                    labels: {{ color: '#888', font: {{ family: 'Inter', size: 11 }} }}
                }}
            }},
            scales: {{
                x: {{
                    grid: {{ color: 'rgba(255,255,255,0.04)' }},
                    ticks: {{ color: '#666', font: {{ size: 10 }}, maxTicksLimit: 10 }}
                }},
                y: {{
                    grid: {{ color: 'rgba(255,255,255,0.04)' }},
                    ticks: {{ color: '#666', font: {{ size: 10 }} }}
                }}
            }}
        }};
        
        // Portfolio Value Chart
        new Chart(document.getElementById('pvChart'), {{
            type: 'line',
            data: {{
                labels: steps,
                datasets: [{{
                    label: 'Portfolio Value ($)',
                    data: portfolioValues,
                    borderColor: '#00d2ff',
                    backgroundColor: 'rgba(0,210,255,0.1)',
                    fill: true,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.1,
                }}]
            }},
            options: chartDefaults
        }});
        
        // Price Chart with Trade Markers
        new Chart(document.getElementById('priceChart'), {{
            type: 'line',
            data: {{
                labels: steps,
                datasets: [
                    {{
                        label: 'Mid Price',
                        data: midPrices,
                        borderColor: '#7b2ff7',
                        borderWidth: 1,
                        pointRadius: 0,
                        tension: 0.1,
                        order: 2,
                    }},
                    {{
                        label: 'Buy',
                        data: buyTrades.map(t => ({{ x: t.x, y: t.y }})),
                        type: 'scatter',
                        backgroundColor: '#00e676',
                        borderColor: '#00e676',
                        pointRadius: 3,
                        pointStyle: 'triangle',
                        order: 1,
                    }},
                    {{
                        label: 'Sell',
                        data: sellTrades.map(t => ({{ x: t.x, y: t.y }})),
                        type: 'scatter',
                        backgroundColor: '#ff5252',
                        borderColor: '#ff5252',
                        pointRadius: 3,
                        pointStyle: 'triangleDown', 
                        order: 1,
                    }}
                ]
            }},
            options: {{
                ...chartDefaults,
                scales: {{
                    ...chartDefaults.scales,
                    x: {{
                        ...chartDefaults.scales.x,
                        type: 'linear',
                        position: 'bottom',
                    }}
                }}
            }}
        }});
        
        // Inventory Chart
        new Chart(document.getElementById('invChart'), {{
            type: 'line',
            data: {{
                labels: steps,
                datasets: [{{
                    label: 'Inventory',
                    data: inventoryData,
                    borderColor: '#ff9800',
                    backgroundColor: 'rgba(255,152,0,0.1)',
                    fill: true,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    stepped: true,
                }}]
            }},
            options: chartDefaults
        }});
        
        // Cumulative Reward Chart
        new Chart(document.getElementById('rewardChart'), {{
            type: 'line',
            data: {{
                labels: steps,
                datasets: [{{
                    label: 'Cumulative Reward',
                    data: cumRewards,
                    borderColor: '#00e676',
                    backgroundColor: 'rgba(0,230,118,0.1)',
                    fill: true,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.1,
                }}]
            }},
            options: chartDefaults
        }});
    </script>
</body>
</html>"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"Dashboard saved to {output_path}")
    print(f"Open in browser: file:///{output_path.replace(os.sep, '/')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate backtest dashboard")
    parser.add_argument("--data", default="aapl", choices=list(DATA_PATHS.keys()))
    parser.add_argument("--model", default=None, help="Model path (without .zip)")
    parser.add_argument("--max-ticks", type=int, default=None, help="Limit evaluation ticks")
    parser.add_argument("--output", default=None, help="Output HTML path")
    args = parser.parse_args()
    
    data_path = DATA_PATHS[args.data]
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        sys.exit(1)
    
    model_path = args.model or os.path.join(MODELS_DIR, "ppo_cvml_final")
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found: {model_path}.zip")
        print("Run train.py first to train a model.")
        sys.exit(1)
    
    output_path = args.output or os.path.join(BASE_DIR, "notebooks", "backtest_dashboard.html")
    macro_path = MACRO_VECTORS_PATH if os.path.exists(MACRO_VECTORS_PATH) else None
    
    print(f"Loading model: {model_path}")
    env = LOBReplayEnv(
        data_path=data_path,
        use_macro_vector=True,
        macro_vectors_path=macro_path,
        reward_type="log_return",
    )
    model = PPO.load(model_path, env=env)
    
    print(f"Running evaluation on {args.data.upper()}...")
    ticks, stats, trades = collect_episode_data(model, env, max_ticks=args.max_ticks)
    
    print(f"\nEpisode Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    generate_html(ticks, stats, trades, output_path,
                  model_name="PPO+CVML", dataset_name=args.data.upper())
