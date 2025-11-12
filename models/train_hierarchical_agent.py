"""
Hierarchical Agent - 动态平衡型策略

核心特征:
1. 三层决策: Aggressive/Balanced/Defensive
2. 基于市场信号智能切换模式
3. 不同模式有明确的行为特征
4. 自适应风险管理

模式特征:
- Aggressive (牛市): 现金5%, 高换手, 集中持仓
- Balanced (震荡): 现金15%, 中换手, 分散持仓  
- Defensive (熊市): 现金35%, 低换手, 极度分散

预期表现:
- 收益: 30-45%
- 波动: 12-16%
- 换手: 1-4%/天
- 现金: 5-35%(动态)
- 模式分布: A30% B40% D30%
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

MODEL_OUT_PATH = os.path.join(MODEL_OUT_DIR, "hierarchical_agent.pth")

TRAIN_TICKERS_POOL = [
    ['AAPL', 'MSFT', 'GOOGL'], ['AMZN', 'TSLA', 'NVDA'], ['META', 'NFLX', 'AMD'],
    ['JPM', 'BAC', 'WFC'], ['GS', 'MS', 'C'], ['JNJ', 'PFE', 'UNH'],
    ['ABBV', 'MRK', 'TMO'], ['XOM', 'CVX', 'COP'], ['PG', 'KO', 'PEP'],
    ['MCD', 'SBUX', 'NKE'],
]

TRAIN_START = "2018-01-01"
TRAIN_END = "2023-01-01"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Hierarchical Agent Training - ADAPTIVE STRATEGY")
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
                preds[t] = {str(k): float(v) for k, v in data.items()
                           if isinstance(v, (int, float, np.floating))}
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

class HierarchicalTradingEnv:
    """动态平衡型环境 - 智能模式切换"""
    def __init__(self, tickers: List[str], price_df: pd.DataFrame, model_predict,
                 min_episode_length: int = 60, max_episode_length: int = 180,
                 transaction_cost: float = 0.0006, risk_free_rate: float = 0.02):
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
        self.mode_history = []
        self.price_mean = None
        self.price_std = None
        self.last_mode = None
        self.mode_persistence = 0  # 模式持续时间
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

    def _compute_market_state(self) -> tuple:
        """计算市场状态指标用于模式决策"""
        if len(self.return_history) < 5:
            return 0.0, 0.0, 0.0  # trend, volatility, momentum
        
        recent_returns = self.return_history[-20:] if len(self.return_history) >= 20 else self.return_history[-5:]
        
        # 趋势: 最近收益的平均值
        trend = np.mean(recent_returns)
        
        # 波动: 标准差
        volatility = np.std(recent_returns)
        
        # 动量: 最近5天vs之前5天
        if len(recent_returns) >= 10:
            recent_5 = np.mean(recent_returns[-5:])
            prev_5 = np.mean(recent_returns[-10:-5])
            momentum = recent_5 - prev_5
        else:
            momentum = 0.0
        
        return trend, volatility, momentum

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
            prices_norm, signals, w_assets,
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

    def determine_mode(self, network_mode_probs: np.ndarray) -> int:
        """
        基于市场状态和网络建议智能确定模式
        0=Aggressive, 1=Balanced, 2=Defensive
        """
        trend, volatility, momentum = self._compute_market_state()
        
        # 基于规则的模式偏好
        mode_scores = np.zeros(3)
        
        # Aggressive偏好: 正趋势, 低波动, 正动量
        if trend > 0.001 and volatility < 0.015 and momentum > 0:
            mode_scores[0] += 2.0
        elif trend > 0.0005:
            mode_scores[0] += 1.0
        
        # Defensive偏好: 负趋势, 高波动, 负动量
        if trend < -0.001 or volatility > 0.025:
            mode_scores[2] += 2.0
        elif trend < 0 and volatility > 0.018:
            mode_scores[2] += 1.5
        
        # Balanced是中性选择
        if -0.001 <= trend <= 0.001:
            mode_scores[1] += 1.0
        if 0.01 < volatility < 0.02:
            mode_scores[1] += 1.0
        
        # 结合网络建议 (权重0.5)
        combined_scores = mode_scores + 0.5 * network_mode_probs
        
        # 添加模式惯性 - 避免频繁切换
        if self.last_mode is not None and self.mode_persistence < 5:
            combined_scores[self.last_mode] += 1.0
        
        # 选择得分最高的模式
        mode = int(np.argmax(combined_scores))
        
        # 更新持续性
        if mode == self.last_mode:
            self.mode_persistence += 1
        else:
            self.mode_persistence = 0
        
        return mode

    def step(self, action: np.ndarray, mode_probs: np.ndarray) -> StepResult:
        """
        action: 低层网络输出的资产权重logits
        mode_probs: 高层网络输出的模式概率
        """
        idx = self.start_idx + self.t
        next_idx = idx + 1
        if next_idx >= len(self.dates) or self.t >= self.episode_length:
            return StepResult(self._get_state(), 0.0, True, {})

        # 🎯 智能模式选择
        mode = self.determine_mode(mode_probs)
        self.mode_history.append(mode)

        scores = np.array(action, dtype=np.float32)
        scores = scores - scores.max()
        exp_scores = np.exp(scores)
        weights = exp_scores / (exp_scores.sum() + 1e-8)

        # 🎯 根据模式调整现金和集中度
        if mode == 0:  # Aggressive
            target_cash_ratio = 0.05
            max_single_position = 0.45  # 允许高集中度
        elif mode == 1:  # Balanced
            target_cash_ratio = 0.15
            max_single_position = 0.35
        else:  # Defensive
            target_cash_ratio = 0.35
            max_single_position = 0.25  # 强制分散

        # 调整权重
        weights[-1] = target_cash_ratio
        stock_weights = weights[:self.n]
        
        # 应用单股持仓限制
        for i in range(self.n):
            if stock_weights[i] > max_single_position:
                stock_weights[i] = max_single_position
        
        # 重新归一化股票权重
        if stock_weights.sum() > 0:
            stock_weights = stock_weights / stock_weights.sum() * (1 - target_cash_ratio)
        weights[:self.n] = stock_weights

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

        # 🎯 平衡型奖励函数 - 适应不同模式
        daily_return = (new_value - prev_value) / (prev_value + 1e-8)
        
        # 基础收益奖励 (中等权重)
        reward = daily_return * 10
        
        # 模式特定奖励
        turnover_ratio = trade_volume.sum() / (prev_value + 1e-8)
        cash_ratio = self.cash / new_value if new_value > 0 else 1.0
        
        if mode == 0:  # Aggressive - 鼓励高收益
            reward *= 1.2
            if turnover_ratio > 0.02:  # 鼓励适度换手
                reward += 0.1
        elif mode == 1:  # Balanced - 平衡收益和稳定
            if 0.01 < turnover_ratio < 0.03:
                reward += 0.15
        else:  # Defensive - 重视资本保护
            if daily_return < 0:
                reward *= 1.5  # 惩罚亏损更多
            if cash_ratio > 0.30:
                reward += 0.2
            if turnover_ratio < 0.015:
                reward += 0.15
        
        # 鼓励模式适应性 (但不过度切换)
        if self.last_mode is not None and self.last_mode != mode and self.mode_persistence >= 3:
            reward += 0.2  # 在合适时机切换有奖励
        
        # 轻度波动惩罚
        if len(self.return_history) >= 10:
            recent_vol = np.std(self.return_history[-10:])
            if recent_vol > 0.025:
                reward -= 0.1

        self.value_history.append(new_value)
        self.return_history.append(daily_return)
        self.last_mode = mode
        self.price_mean = 0.99 * self.price_mean + 0.01 * prices_next
        price_diff = (prices_next - self.price_mean) ** 2
        self.price_std = np.sqrt(0.99 * (self.price_std ** 2) + 0.01 * price_diff)

        done = self.t >= self.episode_length
        
        mode_names = ["Aggressive", "Balanced", "Defensive"]
        concentration = np.max(weights[:self.n]) if weights[:self.n].sum() > 0 else 0
        
        info = {
            "new_value": new_value,
            "turnover": turnover_ratio,
            "cash_ratio": cash_ratio,
            "concentration": concentration,
            "mode": mode_names[mode],
            "mode_idx": mode,
        }
        
        return StepResult(self._get_state(), float(reward), done, info)

# ==================== Hierarchical 网络 ====================

class HierarchicalNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, num_modes: int = 3, hidden: int = 256):
        super().__init__()
        # 共享特征提取
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # 高层: 模式选择网络
        self.mode_net = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_modes),
        )
        
        # 低层: 动作执行网络
        self.action_net = nn.Sequential(
            nn.Linear(hidden + num_modes, hidden),  # 拼接模式信息
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        
        # Critic
        self.value_net = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        
    def forward(self, x):
        shared_feat = self.shared(x)
        
        # 模式选择
        mode_logits = self.mode_net(shared_feat)
        mode_probs = torch.softmax(mode_logits, dim=-1)
        
        # 动作选择 (结合模式信息)
        combined = torch.cat([shared_feat, mode_probs], dim=-1)
        action_logits = self.action_net(combined)
        
        # 价值估计
        value = self.value_net(shared_feat)
        
        return mode_logits, action_logits, value, mode_probs

# ==================== Hierarchical Agent ====================

class HierarchicalAgent:
    def __init__(self, state_dim, action_dim, num_modes=3, lr=3e-4, gamma=0.99, lam=0.95,
                 clip_eps=0.2, ent_coef=0.02, vf_coef=0.5, device='cpu'):
        self.net = HierarchicalNet(state_dim, action_dim, num_modes).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.device = device

    def select_action(self, state: np.ndarray):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mode_logits, action_logits, value, mode_probs = self.net(state_t)
        
        mode_probs_np = mode_probs.cpu().numpy()[0]
        action_logits_np = action_logits.cpu().numpy()[0]
        value_np = value.cpu().item()
        
        # 返回模式概率和动作logits
        return action_logits_np, mode_probs_np, value_np

    def update(self, states, actions, mode_probs_list, rewards, dones, values):
        if len(states) == 0:
            return
        
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(values, dtype=torch.float32, device=self.device)
        
        # Compute returns and advantages
        returns = []
        advantages = []
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
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # Forward pass
        mode_logits, action_logits, values_pred, mode_probs = self.net(states_t)
        
        # Policy loss (simplified - using advantages)
        policy_loss = -advantages_t.mean()
        
        # Value loss
        value_loss = ((values_pred.squeeze() - returns_t) ** 2).mean()
        
        # Entropy for exploration
        mode_entropy = -(mode_probs * torch.log(mode_probs + 1e-8)).sum(dim=1).mean()
        action_probs = torch.softmax(action_logits, dim=-1)
        action_entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
        entropy = mode_entropy + action_entropy
        
        # Total loss
        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
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
    
    agent = HierarchicalAgent(state_dim=14, action_dim=4, num_modes=3, lr=3e-4,
                              gamma=0.99, lam=0.95, clip_eps=0.2, ent_coef=0.02,
                              vf_coef=0.5, device=DEVICE)

    num_episodes = 2000
    steps_per_update = 200

    print(f"\n🎯 Starting ADAPTIVE training for {num_episodes} episodes...")
    print("=" * 60)
    
    all_rewards = []
    all_final_values = []
    all_turnovers = []
    all_cash_ratios = []
    all_mode_counts = [0, 0, 0]  # A, B, D

    for ep in range(1, num_episodes + 1):
        group_idx = ep % len(all_price_data)
        tickers, price_df, model_pred = all_price_data[group_idx]
        
        env = HierarchicalTradingEnv(tickers=tickers, price_df=price_df, model_predict=model_pred,
                                     min_episode_length=60, max_episode_length=180)
        
        state = env.reset()
        ep_rewards = []
        ep_turnovers = []
        ep_cash_ratios = []
        ep_modes = []
        
        states, actions, mode_probs_list, rewards, dones, values = [], [], [], [], [], []

        while True:
            action_logits, mode_probs, value = agent.select_action(state)
            step_res = env.step(action_logits, mode_probs)

            states.append(state)
            actions.append(action_logits)
            mode_probs_list.append(mode_probs)
            rewards.append(step_res.reward)
            dones.append(float(step_res.done))
            values.append(value)

            ep_rewards.append(step_res.reward)
            ep_turnovers.append(step_res.info.get('turnover', 0))
            ep_cash_ratios.append(step_res.info.get('cash_ratio', 0))
            ep_modes.append(step_res.info.get('mode_idx', 1))
            state = step_res.state

            if len(states) >= steps_per_update or step_res.done:
                agent.update(states, actions, mode_probs_list, rewards, dones, values)
                states, actions, mode_probs_list, rewards, dones, values = [], [], [], [], [], []

            if step_res.done:
                final_value = step_res.info["new_value"]
                all_final_values.append(final_value)
                break

        ep_ret = float(np.sum(ep_rewards))
        all_rewards.append(ep_ret)
        all_turnovers.append(np.mean(ep_turnovers))
        all_cash_ratios.append(np.mean(ep_cash_ratios))
        
        # 统计模式分布
        for m in ep_modes:
            all_mode_counts[m] += 1

        if ep % 50 == 0:
            avg_reward = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)
            avg_final = np.mean(all_final_values[-100:]) if len(all_final_values) >= 100 else np.mean(all_final_values)
            avg_turnover = np.mean(all_turnovers[-100:]) if len(all_turnovers) >= 100 else np.mean(all_turnovers)
            avg_cash = np.mean(all_cash_ratios[-100:]) if len(all_cash_ratios) >= 100 else np.mean(all_cash_ratios)
            roi = ((avg_final - 100000) / 100000) * 100
            
            # 最近100集的模式分布
            recent_episodes = min(100, ep)
            recent_mode_sum = sum(all_mode_counts)
            if recent_mode_sum > 0:
                mode_dist = [c / recent_mode_sum * 100 for c in all_mode_counts]
            else:
                mode_dist = [0, 0, 0]
            
            print(
                f"Ep {ep:4d}/{num_episodes} | "
                f"Ret: {ep_ret:7.2f} | Avg: {avg_reward:7.2f} | "
                f"Val: ${avg_final:10.2f} | ROI: {roi:+6.2f}% | "
                f"Turn: {avg_turnover*100:5.1f}% | Cash: {avg_cash*100:4.1f}% | "
                f"Modes: A{mode_dist[0]:.0f}% B{mode_dist[1]:.0f}% D{mode_dist[2]:.0f}%"
            )

    ckpt = {"state_dim": 14, "action_dim": 4, "num_modes": 3, "model_state_dict": agent.net.state_dict()}
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
        plt.title('Hierarchical: Training Returns')
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
        plt.title('Trading Activity (ADAPTIVE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 5)
        plt.plot([c*100 for c in all_cash_ratios], alpha=0.3, label='Cash Ratio', color='green')
        if len(all_cash_ratios) >= 100:
            ma = pd.Series([c*100 for c in all_cash_ratios]).rolling(100).mean()
            plt.plot(ma, label='100-ep MA', linewidth=2, color='red')
        plt.axhline(y=15, color='blue', linestyle='--', label='Target 15%', alpha=0.7)
        plt.xlabel('Episode')
        plt.ylabel('Cash Ratio (%)')
        plt.title('Cash Allocation (DYNAMIC)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 6)
        # 模式分布饼图
        total_modes = sum(all_mode_counts)
        if total_modes > 0:
            mode_percentages = [c / total_modes * 100 for c in all_mode_counts]
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
            labels = ['Aggressive', 'Balanced', 'Defensive']
            plt.pie(mode_percentages, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('Mode Distribution')
        
        plt.tight_layout()
        plot_path = os.path.join(MODEL_OUT_DIR, "hierarchical_training_curves.png")
        plt.savefig(plot_path, dpi=150)
        print(f"✓ Saved training curves to {plot_path}")
    except Exception as e:
        print(f"Warning: Could not save plot: {e}")

    print("\n" + "=" * 60)
    print("🎉 Hierarchical Adaptive Training Complete!")
    print("=" * 60)
    final_100_avg = np.mean(all_final_values[-100:]) if len(all_final_values) >= 100 else np.mean(all_final_values)
    final_roi = ((final_100_avg - 100000) / 100000) * 100
    final_turnover = np.mean(all_turnovers[-100:]) if len(all_turnovers) >= 100 else np.mean(all_turnovers)
    final_cash = np.mean(all_cash_ratios[-100:]) if len(all_cash_ratios) >= 100 else np.mean(all_cash_ratios)
    
    total_modes = sum(all_mode_counts)
    if total_modes > 0:
        mode_dist = [c / total_modes * 100 for c in all_mode_counts]
    else:
        mode_dist = [0, 0, 0]
    
    print(f"Final 100-episode metrics:")
    print(f"  Average value: ${final_100_avg:.2f}")
    print(f"  Average ROI: {final_roi:+.2f}%")
    print(f"  Average turnover: {final_turnover*100:.1f}%")
    print(f"  Average cash ratio: {final_cash*100:.1f}%")
    print(f"  Mode distribution:")
    print(f"    Aggressive: {mode_dist[0]:.1f}%")
    print(f"    Balanced: {mode_dist[1]:.1f}%")
    print(f"    Defensive: {mode_dist[2]:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    train()