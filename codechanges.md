### micro/env.py

#### Note: Modify the environment to enforce long-only trading, add a maximum inventory limit, and penalize invalid actions.
#### Modify the `__init__` method to include `max_inventory` and `invalid_action_penalty` parameters.

```python
class TradingEnvironment(gym.Env):
    def __init__(self, max_inventory=50, invalid_action_penalty=0.01):
        super().__init__()
        self.max_inventory = max_inventory
        self.invalid_action_penalty = invalid_action_penalty
        # ... other initialization ...
```

#### Modify the `step` method to enforce long-only trading, add a maximum inventory limit, and penalize invalid actions.

```python
    def step(self, action):
        # ... other logic ...
        if action == 1:  # Market Buy
            if self.cash >= self.execution_price and self.inventory < self.max_inventory:
                self.cash -= self.execution_price
                self.inventory += 1
                trade_executed = True
                trade_side = "buy"
            else:
                invalid_action = True
        elif action == 2:  # Market Sell
            if self.inventory > 0:
                execution_price = self.best_bid * (1 - self.taker_fee)
                self.cash += execution_price
                self.inventory -= 1
                trade_executed = True
                trade_side = "sell"
            else:
                invalid_action = True
        elif action == 3:  # Limit Buy
            if self.cash >= self.execution_price and self.inventory < self.max_inventory:
                fill_prob = min(0.5, float(self.limit_fill.get("ask_size_1", 500)) / 1000.0)
                if self.np_random.random() < fill_prob:
                    self.cash -= self.execution_price
                    self.inventory += 1
                    trade_executed = True
                    trade_side = "limit_buy"
                else:
                    invalid_action = True
            else:
                invalid_action = True
        elif action == 4:  # Limit Sell
            if self.inventory > 0:
                fill_prob = min(0.5, float(self.limit_fill.get("ask_size_1", 500)) / 1000.0)
                execution_price = self.best_ask * (1 - self.maker_fee)
                if self.np_random.random() < fill_prob:
                    self.cash += execution_price
                    self.inventory -= 1
                    trade_executed = True
                    trade_side = "limit_sell"
                else:
                    invalid_action = True
            else:
                invalid_action = True
        # ... other logic ...

        if invalid_action:
            reward -= self.invalid_action_penalty

        # ... other logic ...

        return self.observation, reward, done, info
```

#### Modify the `reset` method to include terminal liquidation.

```python
    def reset(self):
        # ... other logic ...
        if self.done:
            terminal_value = self.cash + self.inventory * self.best_bid * (1 - self.taker_fee)
            self.terminal_value = terminal_value
            self.observation = {
                "cash": self.cash,
                "inventory": self.inventory,
                "terminal_value": self.terminal_value,
            }
            return self.observation
        else:
            # ... other logic ...
```

### train.py

#### Note: Modify the evaluation loop to include terminal liquidation.

```python
def evaluate_policy(model, env, num_episodes=100):
    rewards = []
    terminal_values = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = model.predict(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        rewards.append(episode_reward)
        terminal_value = env.terminal_value
        terminal_values.append(terminal_value)
    return rewards, terminal_values
```

### build_notebook.py

#### Note: Mirror the changes made to `micro/env.py` and `train.py` in this file.

```python
# ... mirror changes ...
```
