"""
PPO Planning Agent - 激进成长型策略

核心特征:
1. 追求最大化收益,不惧波动
2. 积极调仓,高换手率(3-8%)
3. 接近满仓运作,现金<5%
4. 集中持仓,押注强势股
5. 对交易成本不敏感

预期表现:
- 收益: 40-60%
- 波动: 15-20%
- 换手: 3-8%/天
- 现金: 0-5%
- 回撤: 8-15%
"""

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf

# ==================== 配置 ====================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
PREDICTION_BASE_DIR = os.path.join(BACKEND_DIR, "predictions")
MODEL_OUT_DIR = os.path.join(BACKEND_DIR, "rl_models")
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

MODEL_OUT_PATH = os.path.join(MODEL_OUT_DIR, "ppo_planning_agent.pth")

TRAIN_TICKERS_POOL = [
    ['AAPL', 'MSFT', 'GOOGL'],
    ['AMZN', 'TSLA', 'NVDA'],
    ['META', 'NFLX', 'AMD'],
    ['JPM', 'BAC', 'WFC'],
    ['GS', 'MS', 'C'],
    ['JNJ', 'PFE', 'UNH'],
    ['ABBV', 'MRK', 'TMO'],
    ['XOM', 'CVX', 'COP'],
    ['PG', 'KO', 'PEP'],
    ['MCD', 'SBUX', 'NKE'],
]

TRAIN_START = "2018-01-01"
TRAIN_END = "2023-01-01"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 PPO Planning Agent Training - AGGRESSIVE GROWTH")
print(f"📍 Device: {DEVICE}")
print("=" * 60)

# ==================== 工具函数 ====================

def load_model_predictions(model_dir: str, tickers: List[str]) -> Dict[str, Dict[str, float]]:
    preds: Dict[str, Dict[str, float]] = {}
    if not os.path.isdir(model_dir):
        return preds
    for t in tickers:
        path = os.path.join(model_dir, f"{t}_predictions.pkl")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                preds[t] = {
                    str(k): float(v)
                    for k, v in data.items()
                    if isinstance(v, (int, float, np.floating))
                }
        except Exception as e:
            print(f"[load_model_predictions] error for {t}: {e}")
    return preds

def load_all_model_predictions(tickers: List[str]):
    mapping = {"lstm": "LSTM", "xgb": "XGBoost"}
    all_preds = {}
    for key, folder in mapping.items():
        model_dir = os.path.join(PREDICTION_BASE_DIR, folder)
        m = load_model_predictions(model_dir, tickers)
        if m:
            all_preds[key] = m
    return all_preds

def extract_signals(model_predict, symbols, date_str) -> np.ndarray:
    if not isinstance(model_predict, dict):
        return np.zeros(2 * len(symbols), dtype=np.float32)
    sigs: List[float] = []
    for m_name in ["lstm", "xgb"]:
        m_dict = model_predict.get(m_name, {})
        if not isinstance(m_dict, dict):
            sigs.extend([0.0] * len(symbols))
            continue
        for s in symbols:
            v = 0.0
            if s in m_dict:
                v = float(m_dict[s].get(date_str, 0.0) or 0.0)
            sigs.append(v)
    if len(sigs) != 2 * len(symbols):
        return np.zeros(2 * len(symbols), dtype=np.float32)
    return np.array(sigs, dtype=np.float32)

# ==================== 环境 ====================

@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: dict

class AggressivePPOEnv:
    """激进成长型环境"""
    def __init__(
        self,
        tickers: List[str],
        price_df: pd.DataFrame,
        model_predict,
        min_episode_length: int = 60,
        max_episode_length: int = 180,
        transaction_cost: float = 0.0003,  # 极低交易成本
        risk_free_rate: float = 0.02,
    ):
        self.tickers = tickers
        self.n = len(tickers)
        self.price_df = price_df
        self.model_predict = model_predict
        self.min_ep_len = min_episode_length
        self.max_ep_len = max_episode_length
        self.tc = transaction_cost
        self.rf_rate = risk_free_rate / 252
        self.dates = list(self.price_df.index)
        self.max_start = len(self.dates) - self.max_ep_len - 1
        self.reset()

    def reset(self) -> np.ndarray:
        self.episode_length = np.random.randint(self.min_ep_len, self.max_ep_len + 1)
        if self.max_start <= 0:
            self.start_idx = 0
        else:
            self.start_idx = np.random.randint(0, max(1, len(self.dates) - self.episode_length))
        self.t = 0
        self.cash = 100000.0
        self.positions = np.zeros(self.n, dtype=np.float32)
        self.value_history = [self.cash]
        self.return_history = []
        self.price_mean = None
        self.price_std = None
        return self._get_state()

    def _get_prices(self, idx: int) -> np.ndarray:
        d = self.dates[idx]
        prices = self.price_df.loc[d, self.tickers].values.astype(np.float32)
        if self.price_mean is None:
            self.price_mean = prices.copy()
            self.price_std = np.ones_like(prices)
        return prices

    def _portfolio_value(self, idx: int) -> float:
        prices = self._get_prices(idx)
        return float(self.cash + np.sum(self.positions * prices))

    def _get_state(self) -> np.ndarray:
        idx = self.start_idx + self.t
        date = self.dates[idx]
        date_str = date.strftime("%Y-%m-%d")
        prices = self._get_prices(idx)
        prices_norm = (prices - self.price_mean) / (self.price_std + 1e-8)
        prices_norm = np.clip(prices_norm, -5, 5)
        total_value = self._portfolio_value(idx)
        if total_value <= 0:
            w_assets = np.zeros(self.n, dtype=np.float32)
            cash_w = 1.0
        else:
            w_assets = (self.positions * prices) / total_value
            cash_w = self.cash / total_value
        signals = extract_signals(self.model_predict, self.tickers, date_str)
        remaining = (self.episode_length - self.t) / self.episode_length
        state = np.concatenate([
            prices_norm,
            signals,
            w_assets,
            np.array([cash_w], dtype=np.float32),
            np.array([remaining], dtype=np.float32),
        ])
        return state.astype(np.float32)

    @property
    def state_dim(self):
        return 14

    @property
    def action_dim(self):
        return self.n + 1

    def step(self, action: np.ndarray) -> StepResult:
        """激进型交易逻辑"""
        idx = self.start_idx + self.t
        next_idx = idx + 1
        if next_idx >= len(self.dates) or self.t >= self.episode_length:
            return StepResult(self._get_state(), 0.0, True, {})

        scores = np.array(action, dtype=np.float32)
        scores = scores - scores.max()
        exp_scores = np.exp(scores)
        weights = exp_scores / (exp_scores.sum() + 1e-8)

        # 🔥 激进策略：压低现金比例
        if weights[-1] > 0.05:  # 现金不超过5%
            excess_cash = weights[-1] - 0.05
            weights[-1] = 0.05
            # 将多余现金分配到股票（优先分配给权重最大的）
            stock_weights = weights[:self.n]
            if stock_weights.sum() > 0:
                weights[:self.n] = stock_weights / stock_weights.sum() * (1 - 0.05)

        # 🔥 鼓励集中持仓：允许单股最高50%
        # (移除单股持仓限制)

        # 重新归一化
        weights = weights / (weights.sum() + 1e-8)

        prices_now = self._get_prices(idx)
        prices_next = self._get_prices(next_idx)
        prev_value = self._portfolio_value(idx)

        target_stock_value = weights[:self.n] * prev_value
        target_cash = weights[-1] * prev_value
        target_positions = target_stock_value / (prices_now + 1e-8)

        trade_volume = np.abs(target_positions - self.positions) * prices_now
        cost = self.tc * trade_volume.sum()

        self.positions = target_positions
        self.cash = target_cash - cost
        self.t += 1
        new_value = self._portfolio_value(next_idx)

        # 🔥 激进型奖励函数
        daily_return = (new_value - prev_value) / (prev_value + 1e-8)
        
        # 1. 高收益权重 (15x)
        reward = daily_return * 15
        
        # 2. 惩罚持有过多现金
        cash_ratio = self.cash / new_value if new_value > 0 else 1.0
        if cash_ratio > 0.10:
            reward -= 0.3 * (cash_ratio - 0.10)
        
        # 3. 鼓励集中持仓
        if new_value > 0:
            stock_values = self.positions * prices_next
            max_position_ratio = np.max(stock_values) / new_value if new_value > 0 else 0
            if max_position_ratio > 0.35:  # 集中度>35%有奖励
                reward += 0.2
        
        # 4. 鼓励积极交易 (3-8%换手)
        turnover_ratio = trade_volume.sum() / (prev_value + 1e-8)
        if 0.03 < turnover_ratio < 0.08:
            reward += 0.3
        elif turnover_ratio < 0.01:  # 惩罚过低换手
            reward -= 0.2
        
        # 5. 轻微交易成本惩罚（让agent知道有成本但不过分限制）
        reward -= cost / (prev_value + 1e-8) * 0.5

        self.value_history.append(new_value)
        self.return_history.append(daily_return)
        self.price_mean = 0.99 * self.price_mean + 0.01 * prices_next
        price_diff = (prices_next - self.price_mean) ** 2
        self.price_std = np.sqrt(0.99 * (self.price_std ** 2) + 0.01 * price_diff)

        done = self.t >= self.episode_length
        
        # 计算统计信息
        concentration = max_position_ratio if 'max_position_ratio' in locals() else 0
        
        info = {
            "new_value": new_value,
            "turnover": turnover_ratio,
            "cash_ratio": cash_ratio,
            "concentration": concentration,
        }
        
        return StepResult(self._get_state(), float(reward), done, info)

# ==================== Actor-Critic网络 ====================

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        
    def forward(self, x):
        shared_out = self.shared(x)
        logits = self.actor_head(shared_out)
        value = self.critic_head(shared_out)
        return logits, value

# ==================== PPO Agent ====================

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, lam=0.95,
                 clip_eps=0.2, ent_coef=0.03, vf_coef=0.5, device='cpu'):
        self.ac = PPOActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.device = device

    def select_action(self, state: np.ndarray):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.ac(state_t)
        logits_np = logits.cpu().numpy()[0]
        value_np = value.cpu().item()
        
        # Softmax to get probabilities
        scores = logits_np - logits_np.max()
        exp_scores = np.exp(scores)
        probs = exp_scores / (exp_scores.sum() + 1e-8)
        
        # Sample action
        action_idx = np.random.choice(len(probs), p=probs)
        log_prob = np.log(probs[action_idx] + 1e-8)
        
        return logits_np, log_prob, value_np

    def update(self, states, actions, old_log_probs, rewards, dones, values):
        if len(states) == 0:
            return
        
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        
        # Compute advantages
        advantages = []
        returns = []
        gae = 0
        next_value = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            next_value = values[t]
        
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # PPO update
        logits, values_pred = self.ac(states_t)
        
        # Recompute log probs for current policy
        # We need to compute log prob of the actions that were taken
        # For continuous softmax distribution over actions
        scores = logits - logits.max(dim=1, keepdim=True)[0]
        exp_scores = torch.exp(scores)
        probs = exp_scores / (exp_scores.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute log prob of taken actions
        # This is an approximation - we compute the prob of the logits matching the action
        log_probs = torch.log(probs + 1e-8)
        # Use the max logit action as proxy
        action_indices = actions_t.argmax(dim=1)
        new_log_probs = log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # Policy loss
        ratio = torch.exp(new_log_probs - old_log_probs_t)
        surr1 = ratio * advantages_t
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_t
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = ((values_pred.squeeze() - returns_t) ** 2).mean()
        
        # Entropy bonus (encourage exploration)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        
        # Total loss
        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
        self.optimizer.step()

# ==================== 数据准备 ====================

def prepare_price_data(tickers: List[str]) -> pd.DataFrame:
    df = yf.download(tickers, start=TRAIN_START, end=TRAIN_END, progress=False)
    if df.empty:
        raise RuntimeError("No price data downloaded.")
    close = df["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame()
    close = close.dropna(how="all")
    close = close[tickers]
    close = close.dropna()
    return close

# ==================== 训练循环 ====================

def train():
    print("\n🔄 Loading price data...")
    all_price_data = {}
    for i, tickers in enumerate(TRAIN_TICKERS_POOL):
        print(f"  Group {i+1}: {tickers}")
        try:
            price_df = prepare_price_data(tickers)
            model_pred = load_all_model_predictions(tickers)
            all_price_data[i] = (tickers, price_df, model_pred)
            print(f"    ✓ Loaded {len(price_df)} days")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
    
    if not all_price_data:
        raise RuntimeError("No valid price data loaded!")
    
    print(f"\n✓ Successfully loaded {len(all_price_data)} groups")
    
    agent = PPOAgent(state_dim=14, action_dim=4, lr=3e-4, gamma=0.99, lam=0.95,
                     clip_eps=0.2, ent_coef=0.03, vf_coef=0.5, device=DEVICE)

    num_episodes = 2000
    steps_per_update = 200

    print(f"\n🎯 Starting AGGRESSIVE training for {num_episodes} episodes...")
    print("=" * 60)
    
    all_rewards = []
    all_final_values = []
    all_turnovers = []
    all_concentrations = []
    all_cash_ratios = []

    for ep in range(1, num_episodes + 1):
        group_idx = ep % len(all_price_data)
        tickers, price_df, model_pred = all_price_data[group_idx]
        
        env = AggressivePPOEnv(tickers=tickers, price_df=price_df, model_predict=model_pred,
                              min_episode_length=60, max_episode_length=180)
        
        state = env.reset()
        ep_rewards = []
        ep_turnovers = []
        ep_concentrations = []
        ep_cash_ratios = []
        
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

        while True:
            logits, logp, value = agent.select_action(state)
            step_res = env.step(logits)

            states.append(state)
            actions.append(logits)
            log_probs.append(logp)
            rewards.append(step_res.reward)
            dones.append(float(step_res.done))
            values.append(value)

            ep_rewards.append(step_res.reward)
            ep_turnovers.append(step_res.info.get('turnover', 0))
            ep_concentrations.append(step_res.info.get('concentration', 0))
            ep_cash_ratios.append(step_res.info.get('cash_ratio', 0))
            state = step_res.state

            if len(states) >= steps_per_update or step_res.done:
                agent.update(states, actions, log_probs, rewards, dones, values)
                states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

            if step_res.done:
                final_value = step_res.info["new_value"]
                all_final_values.append(final_value)
                break

        ep_ret = float(np.sum(ep_rewards))
        all_rewards.append(ep_ret)
        all_turnovers.append(np.mean(ep_turnovers))
        all_concentrations.append(np.mean(ep_concentrations))
        all_cash_ratios.append(np.mean(ep_cash_ratios))

        if ep % 50 == 0:
            avg_reward = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)
            avg_final = np.mean(all_final_values[-100:]) if len(all_final_values) >= 100 else np.mean(all_final_values)
            avg_turnover = np.mean(all_turnovers[-100:]) if len(all_turnovers) >= 100 else np.mean(all_turnovers)
            avg_concentration = np.mean(all_concentrations[-100:]) if len(all_concentrations) >= 100 else np.mean(all_concentrations)
            avg_cash = np.mean(all_cash_ratios[-100:]) if len(all_cash_ratios) >= 100 else np.mean(all_cash_ratios)
            roi = ((avg_final - 100000) / 100000) * 100
            print(
                f"Ep {ep:4d}/{num_episodes} | "
                f"Ret: {ep_ret:7.2f} | Avg: {avg_reward:7.2f} | "
                f"Val: ${avg_final:10.2f} | ROI: {roi:+6.2f}% | "
                f"Turn: {avg_turnover*100:5.1f}% | Cash: {avg_cash*100:4.1f}% | Conc: {avg_concentration*100:5.1f}%"
            )

    ckpt = {"state_dim": 14, "action_dim": 4, "model_state_dict": agent.ac.state_dict()}
    torch.save(ckpt, MODEL_OUT_PATH)
    print(f"\n✓ Saved model to {MODEL_OUT_PATH}")
    
    # 绘图
    try:
        plt.figure(figsize=(18, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(all_rewards, alpha=0.3, label='Episode Return', color='blue')
        if len(all_rewards) >= 100:
            ma = pd.Series(all_rewards).rolling(100).mean()
            plt.plot(ma, label='100-ep MA', linewidth=2, color='red')
        plt.xlabel('Episode')
        plt.ylabel('Episode Return')
        plt.title('PPO Aggressive: Training Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.plot(all_final_values, alpha=0.3, label='Final Value', color='green')
        if len(all_final_values) >= 100:
            ma = pd.Series(all_final_values).rolling(100).mean()
            plt.plot(ma, label='100-ep MA', linewidth=2, color='red')
        plt.axhline(y=100000, color='black', linestyle='--', label='Initial', alpha=0.7)
        plt.xlabel('Episode')
        plt.ylabel('Final Portfolio Value ($)')
        plt.title('Final Portfolio Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        roi_values = [(v - 100000) / 100000 * 100 for v in all_final_values]
        plt.plot(roi_values, alpha=0.3, label='ROI', color='purple')
        if len(roi_values) >= 100:
            ma = pd.Series(roi_values).rolling(100).mean()
            plt.plot(ma, label='100-ep MA', linewidth=2, color='red')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.xlabel('Episode')
        plt.ylabel('ROI (%)')
        plt.title('Return on Investment')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 4)
        plt.plot([t*100 for t in all_turnovers], alpha=0.3, label='Turnover', color='orange')
        if len(all_turnovers) >= 100:
            ma = pd.Series([t*100 for t in all_turnovers]).rolling(100).mean()
            plt.plot(ma, label='100-ep MA', linewidth=2, color='red')
        plt.xlabel('Episode')
        plt.ylabel('Turnover Rate (%)')
        plt.title('Trading Activity (AGGRESSIVE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 5)
        plt.plot([c*100 for c in all_cash_ratios], alpha=0.3, label='Cash Ratio', color='green')
        if len(all_cash_ratios) >= 100:
            ma = pd.Series([c*100 for c in all_cash_ratios]).rolling(100).mean()
            plt.plot(ma, label='100-ep MA', linewidth=2, color='red')
        plt.axhline(y=5, color='red', linestyle='--', label='Target 5%', alpha=0.7)
        plt.xlabel('Episode')
        plt.ylabel('Cash Ratio (%)')
        plt.title('Cash Allocation (Low)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 6)
        plt.plot([c*100 for c in all_concentrations], alpha=0.3, label='Concentration', color='brown')
        if len(all_concentrations) >= 100:
            ma = pd.Series([c*100 for c in all_concentrations]).rolling(100).mean()
            plt.plot(ma, label='100-ep MA', linewidth=2, color='red')
        plt.axhline(y=35, color='red', linestyle='--', label='Target >35%', alpha=0.7)
        plt.xlabel('Episode')
        plt.ylabel('Max Position (%)')
        plt.title('Position Concentration (High)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(MODEL_OUT_DIR, "ppo_planning_training_curves.png")
        plt.savefig(plot_path, dpi=150)
        print(f"✓ Saved training curves to {plot_path}")
    except Exception as e:
        print(f"Warning: Could not save plot: {e}")

    print("\n" + "=" * 60)
    print("🎉 PPO Aggressive Growth Training Complete!")
    print("=" * 60)
    final_100_avg = np.mean(all_final_values[-100:]) if len(all_final_values) >= 100 else np.mean(all_final_values)
    final_roi = ((final_100_avg - 100000) / 100000) * 100
    final_turnover = np.mean(all_turnovers[-100:]) if len(all_turnovers) >= 100 else np.mean(all_turnovers)
    final_concentration = np.mean(all_concentrations[-100:]) if len(all_concentrations) >= 100 else np.mean(all_concentrations)
    final_cash = np.mean(all_cash_ratios[-100:]) if len(all_cash_ratios) >= 100 else np.mean(all_cash_ratios)
    print(f"Final 100-episode metrics:")
    print(f"  Average value: ${final_100_avg:.2f}")
    print(f"  Average ROI: {final_roi:+.2f}%")
    print(f"  Average turnover: {final_turnover*100:.1f}%")
    print(f"  Average cash ratio: {final_cash*100:.1f}%")
    print(f"  Average concentration: {final_concentration*100:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    train()