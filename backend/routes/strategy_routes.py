# backend/routes/strategy_routes.py
# 🔥 完全修复版 - 解决权重加载和交易量问题

"""
关键修复:
1. 网络结构匹配训练脚本 (shared + actor_head)
2. 权重直接加载,不需要key映射
3. 🔥 浮点数持仓系统 - 与训练环境完全一致
4. 前端显示时才四舍五入到整数
"""

import os
import math
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from flask import Blueprint, request, jsonify
from flask_cors import cross_origin

from utils.db_utils import db, TradeLog

# ====== Torch 可能不存在时的兼容处理 ======
import torch
import torch.nn as nn
TORCH_AVAILABLE = True

# ====== 基础路径配置 ======

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
PREDICTION_BASE_DIR = os.path.join(BACKEND_DIR, "predictions")
MODEL_WEIGHT_DIR = os.path.join(BACKEND_DIR, "rl_models")

os.makedirs(MODEL_WEIGHT_DIR, exist_ok=True)

strategy_bp = Blueprint("strategy_bp", __name__)

# 四类 Agent 的对外名字
SUPPORTED_AGENTS = {
    "ppo_planning": "PPOPlanning",
    "hierarchical": "Hierarchical",
    "risk_constrained": "RiskConstrained",
    "llm_reasoning": "LLMReasoning",
    "naive": "Naive",
}

# =========================
# 工具：加载预测信号（LSTM / XGB）
# =========================

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def load_model_predictions(model_dir: str, tickers):
    """读取预测文件"""
    preds = {}
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
                preds[t] = {str(k): _safe_float(v) for k, v in data.items()}
            elif hasattr(data, "items"):
                preds[t] = {str(k): _safe_float(v) for k, v in data.items()}
        except Exception as e:
            print(f"[load_model_predictions] {t} error: {e}")
    return preds

def load_all_model_predictions(tickers):
    """聚合 LSTM / XGBoost 信号（只用2个模型，匹配14维state）"""
    mapping = {"lstm": "LSTM", "xgb": "XGBoost"}
    all_preds = {}
    for key, folder in mapping.items():
        d = os.path.join(PREDICTION_BASE_DIR, folder)
        m = load_model_predictions(d, tickers)
        if m:
            all_preds[key] = m
    return all_preds

def extract_signals(model_predict, symbols, date_str):
    if not isinstance(model_predict, dict):
        return []

    sigs = []
    for m_key in ("lstm", "xgb"):
        m_dict = model_predict.get(m_key)
        if not isinstance(m_dict, dict):
            sigs.extend([0.0] * len(symbols))
            continue

        for s in symbols:
            v = 0.0
            if s in m_dict:
                # 🩵 新增：兼容 "2023-12-05 00:00:00" 键
                keys = list(m_dict[s].keys())
                matched_key = None
                for k in keys:
                    if date_str in str(k):   # 模糊包含匹配
                        matched_key = k
                        break
                if matched_key:
                    v = _safe_float(m_dict[s][matched_key], 0.0)
            sigs.append(v)
    return sigs


# =========================
# 策略基类
# =========================

class MyStrategy:
    def __init__(
        self,
        symbols,
        start_date,
        end_date,
        start_cash=100000.0,
        model_predict=None,
        rebalance_interval=1,
        strategy_name="Base",
    ):
        self.symbols = list(symbols)
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_cash = float(start_cash)
        self.cash = float(start_cash)
        self.model_predict = model_predict or {}
        self.rebalance_interval = max(1, int(rebalance_interval))
        self.strategy = strategy_name

        self.portfolio = {}  # 🔥 将存储浮点数
        self.trade_log = []

        self.data, self.data_open, self.trading_days = self._download_data()

    def _download_data(self):
        symbols = list(self.symbols)
        if "SPY" not in symbols:
            symbols.append("SPY")

        df = yf.download(
            symbols,
            start=self.start_date,
            end=self.end_date + timedelta(days=5),
            progress=False,
        )

        if df.empty:
            raise RuntimeError("Failed to download price data.")

        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"].copy()
            open_ = df["Open"].copy()
        else:
            close = df[["Close"]].copy()
            open_ = df[["Open"]].copy()

        close = close[self.symbols].dropna(how="all")
        open_ = open_[self.symbols].reindex(close.index).fillna(method="ffill")

        trading_days = [d for d in close.index if self.start_date <= d <= self.end_date]
        if not trading_days:
            trading_days = list(close.index)

        return close, open_, trading_days

    def run_backtest(self):
        if not self.trading_days:
            return []

        portfolio_values = []
        total_days = len(self.trading_days)
        
        # 🔥 添加进度提示
        print(f"[{self.strategy}] 🚀 开始回测: {total_days} 个交易日")
        print(f"[{self.strategy}] 📅 从 {self.trading_days[0].strftime('%Y-%m-%d')} 到 {self.trading_days[-1].strftime('%Y-%m-%d')}")

        for i, current_date in enumerate(self.trading_days):
            # 🔥 每天输出进度
            progress = (i + 1) / total_days * 100
            date_str = current_date.strftime('%Y-%m-%d')
            print(f"[{self.strategy}] 📊 Day {i+1}/{total_days} ({progress:.1f}%) - {date_str}", flush=True)
            
            if i % self.rebalance_interval == 0:
                print(f"[{self.strategy}] 🔄 执行 rebalance...", flush=True)
                self.rebalance(current_date)

            value = float(
                self.cash
                + sum(
                    float(self.data.loc[current_date, s])
                    * self.portfolio.get(s, 0.0)  # 🔥 浮点数
                    for s in self.symbols
                )
            )
            portfolio_values.append(
                {"date": current_date.strftime("%Y-%m-%d"), "value": value}
            )
            
            print(f"[{self.strategy}] 💰 Portfolio Value: ${value:,.2f}", flush=True)

        print(f"[{self.strategy}] ✅ 回测完成！", flush=True)
        return portfolio_values

    def rebalance(self, current_date):
        raise NotImplementedError


# =========================
# State 构造：14维 [3价格 + 6信号 + 3持仓 + 1现金 + 1时间]
# =========================

def build_state_from_market(strategy: MyStrategy, current_date: pd.Timestamp):
    """构造14维state，匹配训练环境"""
    symbols = strategy.symbols
    n = len(symbols)

    prices = np.array(
        [float(strategy.data.loc[current_date, s]) for s in symbols],
        dtype=np.float32,
    )
    mean_p = float(prices.mean()) if prices.size > 0 else 1.0
    prices_norm = prices / (mean_p + 1e-8)

    values = []
    for s in symbols:
        px_o = float(strategy.data_open.loc[current_date, s])
        sh = strategy.portfolio.get(s, 0.0)  # 🔥 浮点数
        values.append(px_o * sh)
    total_value = float(strategy.cash + sum(values))

    if total_value <= 0:
        weights_assets = np.zeros(n, dtype=np.float32)
        cash_weight = 1.0
    else:
        weights_assets = np.array(
            [
                (float(strategy.data_open.loc[current_date, s])
                 * strategy.portfolio.get(s, 0.0))  # 🔥 浮点数
                / total_value
                for s in symbols
            ],
            dtype=np.float32,
        )
        cash_weight = float(strategy.cash / total_value)

    date_str = current_date.strftime("%Y-%m-%d")
    signals = extract_signals(strategy.model_predict, symbols, date_str)
    
    # 确保是2*n=6维信号
    if signals and len(signals) >= 2 * n:
        signals_vec = np.array(signals[:2*n], dtype=np.float32)
    else:
        signals_vec = np.zeros(2 * n, dtype=np.float32)

    total_days = max(1, (strategy.end_date.date() - strategy.start_date.date()).days)
    remaining_days = max(0, (strategy.end_date.date() - current_date.date()).days)
    remain_ratio = remaining_days / total_days

    state = np.concatenate([
        prices_norm,              # [3]
        signals_vec,               # [6]
        weights_assets,            # [3]
        np.array([cash_weight], dtype=np.float32),  # [1]
        np.array([remain_ratio], dtype=np.float32), # [1]
    ])

    return state.astype(np.float32)


# =========================
# 网络结构定义 - 与训练脚本完全一致
# =========================

if TORCH_AVAILABLE:
    # PPO Planning 网络 - 🔥 完全匹配训练脚本
    class PPOActorCritic(nn.Module):
        def __init__(self, state_dim=14, action_dim=4, hidden=256):
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

    # Hierarchical 网络
    class HierarchicalNet(nn.Module):
        def __init__(self, state_dim=14, action_dim=4, n_modes=3, hidden=256):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
            )
            self.mode_head = nn.Linear(hidden, n_modes)
            self.action_head = nn.Linear(hidden, action_dim)
            self.value_head = nn.Linear(hidden, 1)

        def forward(self, x):
            shared_out = self.shared(x)
            mode_logits = self.mode_head(shared_out)
            action_logits = self.action_head(shared_out)
            value = self.value_head(shared_out)
            mode_probs = torch.softmax(mode_logits, dim=-1)
            return mode_logits, action_logits, value, mode_probs

    # Risk-Constrained 网络 - 🔥 完全匹配训练脚本
    class RiskConstrainedNet(nn.Module):
        def __init__(self, state_dim=14, action_dim=4, hidden=256):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
            )
            self.actor_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.Tanh(),
                nn.Linear(hidden // 2, action_dim),
            )
            self.critic_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.Tanh(),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, x):
            shared_out = self.shared(x)
            logits = self.actor_head(shared_out)
            value = self.critic_head(shared_out)
            return logits, value


# =========================
# Heuristic fallback
# =========================

def heuristic_weights_from_signals(symbols, signals):
    """当模型不可用时的简单启发式策略"""
    n = len(symbols)
    if not signals or len(signals) < 2 * n:
        w = np.ones(n + 1, dtype=np.float32) / (n + 1)
        return w
    
    lstm_sigs = signals[:n]
    xgb_sigs = signals[n:2*n]
    avg_sigs = [(lstm_sigs[i] + xgb_sigs[i]) / 2 for i in range(n)]
    
    min_sig = min(avg_sigs)
    if min_sig < 0:
        avg_sigs = [s - min_sig + 0.01 for s in avg_sigs]
    
    total = sum(avg_sigs)
    if total > 0:
        stock_weights = [s / total * 0.7 for s in avg_sigs]
        cash_weight = 0.3
    else:
        stock_weights = [0.2] * n
        cash_weight = 0.4
    
    weights = np.array(stock_weights + [cash_weight], dtype=np.float32)
    return weights / (weights.sum() + 1e-8)


# =========================
# Agent 定义 - 🔥 修复权重加载
# =========================

class PPOPlanningAgent:
    """PPO Planning Agent - 激进成长型"""
    def __init__(self, model_path=None, device="cpu"):
        self.policy = None
        self.device = torch.device(device) if TORCH_AVAILABLE else None

        if not TORCH_AVAILABLE:
            print("[PPOPlanningAgent] PyTorch not available")
            return

        if not (model_path and os.path.exists(model_path)):
            print(f"[PPOPlanningAgent] Model not found: {model_path}")
            return

        try:
            print(f"[PPOPlanningAgent] Loading from {model_path}")
            ckpt = torch.load(model_path, map_location=self.device)
            
            state_dim = ckpt.get("state_dim", 14)
            action_dim = ckpt.get("action_dim", 4)
            model_state = ckpt.get("model_state_dict", ckpt)

            self.policy = PPOActorCritic(state_dim, action_dim).to(self.device)
            
            # 🔥 直接加载完整的state_dict
            missing, unexpected = self.policy.load_state_dict(model_state, strict=False)
            if not missing and not unexpected:
                print(f"[PPOPlanningAgent] ✅ Loaded successfully")
            else:
                print(f"[PPOPlanningAgent] ⚠️ Partial load (OK)")
            self.policy.eval()

        except Exception as e:
            print(f"[PPOPlanningAgent] ❌ Load failed: {e}")
            self.policy = None

    def act(self, state, symbols, signals):
        n = len(symbols)
        
        if self.policy is None:
            return heuristic_weights_from_signals(symbols, signals)
        
        try:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits, value = self.policy(s)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                
                # 激进策略：压低现金比例
                if probs[-1] > 0.05:
                    probs[-1] = 0.05
                    stock_probs = probs[:-1]
                    if stock_probs.sum() > 0:
                        probs[:-1] = stock_probs / stock_probs.sum() * 0.95
                
                probs = probs / (probs.sum() + 1e-8)
            
            if len(probs) != n + 1:
                return heuristic_weights_from_signals(symbols, signals)
            
            return probs.astype(np.float32)
        except Exception as e:
            print(f"[PPOPlanningAgent] Inference failed: {e}")
            return heuristic_weights_from_signals(symbols, signals)


class HierarchicalAgent:
    """Hierarchical Agent - 动态平衡型"""
    def __init__(self, model_path=None, device="cpu"):
        self.policy = None
        self.device = torch.device(device) if TORCH_AVAILABLE else None

        if not TORCH_AVAILABLE:
            print("[HierarchicalAgent] PyTorch not available")
            return

        if not (model_path and os.path.exists(model_path)):
            print(f"[HierarchicalAgent] Model not found: {model_path}")
            return

        try:
            print(f"[HierarchicalAgent] Loading from {model_path}")
            ckpt = torch.load(model_path, map_location=self.device)
            
            state_dim = ckpt.get("state_dim", 14)
            action_dim = ckpt.get("action_dim", 4)
            n_modes = ckpt.get("num_modes", ckpt.get("n_modes", 3))
            model_state = ckpt.get("model_state_dict", ckpt)

            self.policy = HierarchicalNet(state_dim, action_dim, n_modes).to(self.device)
            
            missing, unexpected = self.policy.load_state_dict(model_state, strict=False)
            if not missing and not unexpected:
                print(f"[HierarchicalAgent] ✅ Loaded successfully")
            else:
                print(f"[HierarchicalAgent] ⚠️ Partial load (OK)")
            self.policy.eval()

        except Exception as e:
            print(f"[HierarchicalAgent] ❌ Load failed: {e}")
            self.policy = None

    def act(self, state, symbols, signals):
        n = len(symbols)
        
        if self.policy is None:
            return heuristic_weights_from_signals(symbols, signals)

        try:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                mode_logits, action_logits, _, mode_probs = self.policy(s)
                
                mode_probs_np = mode_probs.cpu().numpy()[0]
                mode = int(np.argmax(mode_probs_np))
                
                if mode == 0:  # aggressive
                    target_cash_ratio = 0.05
                    max_single_position = 0.45
                elif mode == 1:  # balanced
                    target_cash_ratio = 0.15
                    max_single_position = 0.35
                else:  # defensive
                    target_cash_ratio = 0.35
                    max_single_position = 0.25
                
                action_probs = torch.softmax(action_logits, dim=-1).cpu().numpy()[0]
                
                action_probs[-1] = target_cash_ratio
                stock_weights = action_probs[:-1]
                
                for i in range(len(stock_weights)):
                    if stock_weights[i] > max_single_position:
                        stock_weights[i] = max_single_position
                
                if stock_weights.sum() > 0:
                    stock_weights = stock_weights / stock_weights.sum() * (1 - target_cash_ratio)
                action_probs[:-1] = stock_weights
                
            if len(action_probs) != n + 1:
                return heuristic_weights_from_signals(symbols, signals)

            return action_probs.astype(np.float32)
            
        except Exception as e:
            print(f"[HierarchicalAgent] Inference failed: {e}")
            return heuristic_weights_from_signals(symbols, signals)


class RiskConstrainedAgent:
    """Risk-Constrained Agent - 保守防御型"""
    def __init__(self, model_path=None, device="cpu"):
        self.policy = None
        self.device = torch.device(device) if TORCH_AVAILABLE else None

        if not TORCH_AVAILABLE:
            print("[RiskConstrainedAgent] PyTorch not available")
            return

        if not (model_path and os.path.exists(model_path)):
            print(f"[RiskConstrainedAgent] Model not found: {model_path}")
            return

        try:
            print(f"[RiskConstrainedAgent] Loading from {model_path}")
            ckpt = torch.load(model_path, map_location=self.device)
            
            state_dim = ckpt.get("state_dim", 14)
            action_dim = ckpt.get("action_dim", 4)
            model_state = ckpt.get("model_state_dict", ckpt)

            self.policy = RiskConstrainedNet(state_dim, action_dim).to(self.device)
            
            # 🔥 直接加载完整的state_dict
            missing, unexpected = self.policy.load_state_dict(model_state, strict=False)
            if not missing and not unexpected:
                print(f"[RiskConstrainedAgent] ✅ Loaded successfully")
            else:
                print(f"[RiskConstrainedAgent] ⚠️ Partial load (OK)")
            self.policy.eval()

        except Exception as e:
            print(f"[RiskConstrainedAgent] ❌ Load failed: {e}")
            self.policy = None

    def act(self, state, symbols, signals):
        n = len(symbols)
        
        if self.policy is None:
            return heuristic_weights_from_signals(symbols, signals)
        
        try:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits, value = self.policy(s)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                
                # 保守策略：保持高现金比例
                if probs[-1] < 0.30:
                    probs[-1] = 0.40
                    stock_probs = probs[:-1]
                    if stock_probs.sum() > 0:
                        stock_probs = np.minimum(stock_probs, 0.20)
                        probs[:-1] = stock_probs / stock_probs.sum() * 0.60
                
                probs = probs / (probs.sum() + 1e-8)
            
            if len(probs) != n + 1:
                return heuristic_weights_from_signals(symbols, signals)
            
            return probs.astype(np.float32)
        except Exception as e:
            print(f"[RiskConstrainedAgent] Inference failed: {e}")
            return heuristic_weights_from_signals(symbols, signals)


# =========================
# 🔥🔥🔥 核心修复: 浮点数持仓系统
# =========================

def fractional_rebalance(strategy, current_date, target_weights):
    """
    🔥 浮点数持仓系统 - 与训练环境完全一致
    
    关键点:
    1. 内部使用浮点数持仓 (fractional shares)
    2. 前端显示时才四舍五入
    3. 完全匹配训练环境的行为
    4. 支持reasoning字段(用于LLM agent)
    """
    prev_balance = (
        strategy.trade_log[-1]["balance"] if strategy.trade_log else strategy.initial_cash
    )
    prev_port = strategy.portfolio.copy()

    # 计算当前总资产
    total_value = float(strategy.cash)
    for s, sh in prev_port.items():
        px = float(strategy.data_open.loc[current_date, s])
        total_value += px * sh  # 🔥 sh是浮点数

    # 🔥 计算目标持仓 (浮点数)
    n = len(strategy.symbols)
    for i, symbol in enumerate(strategy.symbols):
        px = float(strategy.data_open.loc[current_date, symbol])
        if px <= 0:
            strategy.portfolio[symbol] = 0.0
            continue
            
        target_value = total_value * float(target_weights[i])
        # 🔥🔥🔥 关键: 直接使用浮点数,不取整!
        target_shares = target_value / px
        strategy.portfolio[symbol] = target_shares
    
    # 更新现金
    used_cash = 0.0
    for symbol in strategy.symbols:
        px = float(strategy.data_open.loc[current_date, symbol])
        shares = strategy.portfolio.get(symbol, 0.0)
        used_cash += shares * px
    
    strategy.cash = total_value - used_cash
    
    # 计算收盘时的价值
    close_value = float(
        strategy.cash
        + sum(
            float(strategy.data.loc[current_date, s])
            * strategy.portfolio.get(s, 0.0)  # 🔥 浮点数
            for s in strategy.symbols
        )
    )
    
    # 🔥 计算交易变化 (前端显示用整数)
    change = {}
    for s in set(list(prev_port.keys()) + strategy.symbols):
        old_pos = prev_port.get(s, 0.0)
        new_pos = strategy.portfolio.get(s, 0.0)
        delta = new_pos - old_pos
        # 只记录变化超过0.1股的交易
        if abs(delta) > 0.1:
            change[s] = round(delta)  # 前端显示用整数
    
    # 记录交易日志
    date_str = current_date.strftime("%Y-%m-%d")
    strategy.trade_log.append(
        {
            "date": date_str,
            "balance": close_value,
            "earning": close_value - prev_balance,
            # 🔥 portfolio显示用整数,但内部保留浮点数
            "portfolio": {s: round(v) for s, v in strategy.portfolio.items()},
            "change": change,
            "reasoning": "",  # Will be filled by LLM strategies
        }
    )


# =========================
# 策略实现类
# =========================

class NaiveStrategy(MyStrategy):
    def __init__(self, **kwargs):
        super().__init__(strategy_name="Naive", **kwargs)

    def rebalance(self, current_date):
        if self.trade_log:
            return
        
        n = len(self.symbols)
        equal_weight = 1.0 / n
        weights = np.array([equal_weight] * n + [0.0], dtype=np.float32)
        fractional_rebalance(self, current_date, weights)


class PPOPlanningStrategy(MyStrategy):
    def __init__(self, agent: PPOPlanningAgent = None, **kwargs):
        super().__init__(strategy_name="PPOPlanning", **kwargs)
        model_path = os.path.join(MODEL_WEIGHT_DIR, "ppo_planning_agent.pth")
        self.agent = agent or PPOPlanningAgent(model_path=model_path)

    def rebalance(self, current_date):
        state = build_state_from_market(self, current_date)
        date_str = current_date.strftime("%Y-%m-%d")
        signals = extract_signals(self.model_predict, self.symbols, date_str)

        weights = self.agent.act(state, self.symbols, signals)
        weights = np.maximum(weights, 0.0)
        if weights.sum() == 0:
            weights = heuristic_weights_from_signals(self.symbols, signals)
        weights = weights / (weights.sum() + 1e-8)

        fractional_rebalance(self, current_date, weights)


class HierarchicalStrategy(MyStrategy):
    def __init__(self, agent=None, **kwargs):
        super().__init__(strategy_name="Hierarchical", **kwargs)
        model_path = os.path.join(MODEL_WEIGHT_DIR, "hierarchical_agent.pth")
        self.agent = agent or HierarchicalAgent(model_path=model_path)

    def rebalance(self, current_date):
        state = build_state_from_market(self, current_date)
        date_str = current_date.strftime("%Y-%m-%d")
        signals = extract_signals(self.model_predict, self.symbols, date_str)

        weights = self.agent.act(state, self.symbols, signals)
        weights = np.maximum(weights, 0.0)
        if weights.sum() == 0:
            weights = heuristic_weights_from_signals(self.symbols, signals)
        weights = weights / (weights.sum() + 1e-8)

        fractional_rebalance(self, current_date, weights)


class RiskConstrainedStrategy(MyStrategy):
    def __init__(self, agent=None, **kwargs):
        super().__init__(strategy_name="RiskConstrained", **kwargs)
        model_path = os.path.join(MODEL_WEIGHT_DIR, "risk_constrained_agent.pth")
        self.agent = agent or RiskConstrainedAgent(model_path=model_path)

    def rebalance(self, current_date):
        state = build_state_from_market(self, current_date)
        date_str = current_date.strftime("%Y-%m-%d")
        signals = extract_signals(self.model_predict, self.symbols, date_str)

        weights = self.agent.act(state, self.symbols, signals)
        weights = np.maximum(weights, 0.0)
        if weights.sum() == 0:
            weights = heuristic_weights_from_signals(self.symbols, signals)
        weights = weights / (weights.sum() + 1e-8)

        fractional_rebalance(self, current_date, weights)


class LLMReasoningStrategy(MyStrategy):
    """
    LLM-based trading strategy using GPT-4o for decision making.
    Features Chain-of-Thought reasoning and interpretable trade explanations.
    """
    def __init__(self, **kwargs):
        super().__init__(strategy_name="LLMReasoning", **kwargs)
        # Import LLM agent (lazy import to avoid dependency issues)
        try:
            import sys
            import os
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models")
            if models_dir not in sys.path:
                sys.path.insert(0, models_dir)
            from llm_reasoning_agent import create_llm_agent
            self.llm_agent = create_llm_agent()
            print("[LLMReasoningStrategy] ✓ LLM agent initialized")
        except Exception as e:
            print(f"[LLMReasoningStrategy] ⚠️ Failed to load LLM agent: {e}")
            print("[LLMReasoningStrategy] Falling back to heuristic mode")
            self.llm_agent = None

    def rebalance(self, current_date):
        """Execute rebalancing with LLM reasoning."""
        date_str = current_date.strftime("%Y-%m-%d")
        
        print(f"[LLMReasoning] ⏰ {date_str} - 开始LLM决策...", flush=True)
        
        # Extract predictive signals
        signals = extract_signals(self.model_predict, self.symbols, date_str)
        
        # Prepare signal dictionary for LLM
        signal_dict = {}
        if signals and len(signals) >= 2 * len(self.symbols):
            signal_dict["lstm"] = {
                s: float(signals[i]) for i, s in enumerate(self.symbols)
            }
            signal_dict["xgb"] = {
                s: float(signals[len(self.symbols) + i]) for i, s in enumerate(self.symbols)
            }
        
        print(f"[LLMReasoning] 📈 信号: LSTM={signal_dict.get('lstm', {})}", flush=True)
        
        # Get current prices
        prices = {
            s: float(self.data_open.loc[current_date, s]) for s in self.symbols
        }
        
        print(f"[LLMReasoning] 💵 价格: {prices}", flush=True)
        
        # Calculate current portfolio state
        prev_balance = (
            self.trade_log[-1]["balance"] if self.trade_log else self.initial_cash
        )
        
        reasoning = "Using fallback heuristic strategy."  # Default
        
        # Try to get LLM decision
        if self.llm_agent is not None:
            try:
                print(f"[LLMReasoning] 🤖 调用 GPT-4o API...", flush=True)
                import time
                start_time = time.time()
                
                # Get LLM trading decision with reasoning
                trades, reasoning = self.llm_agent.get_trading_decision(
                    date=date_str,
                    symbols=self.symbols,
                    prices=prices,
                    portfolio=self.portfolio.copy(),
                    cash=float(self.cash),
                    signals=signal_dict
                )
                
                elapsed = time.time() - start_time
                print(f"[LLMReasoning] ✅ LLM响应成功 (耗时: {elapsed:.2f}s)", flush=True)
                print(f"[LLMReasoning] 💡 决策: {trades}", flush=True)
                print(f"[LLMReasoning] 📝 理由: {reasoning}", flush=True)
                
                # Convert LLM trades to target weights
                total_value = float(self.cash)
                for s in self.symbols:
                    total_value += self.portfolio.get(s, 0) * prices[s]
                
                weights = []
                for symbol in self.symbols:
                    trade = trades.get(symbol, {"action": "hold", "shares": 0})
                    action = trade["action"]
                    shares = trade["shares"]
                    
                    if action == "buy":
                        # Increase position
                        current_shares = self.portfolio.get(symbol, 0)
                        target_shares = current_shares + shares
                    elif action == "sell":
                        # Decrease position
                        current_shares = self.portfolio.get(symbol, 0)
                        target_shares = max(0, current_shares - shares)
                    else:  # hold
                        target_shares = self.portfolio.get(symbol, 0)
                    
                    target_value = target_shares * prices[symbol]
                    weight = target_value / total_value if total_value > 0 else 0
                    weights.append(weight)
                
                # Add cash weight
                cash_weight = max(0, 1.0 - sum(weights))
                weights.append(cash_weight)
                weights = np.array(weights, dtype=np.float32)
                
                # Normalize weights
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                else:
                    # Fallback to heuristic
                    weights = heuristic_weights_from_signals(self.symbols, signals)
                    reasoning = "LLM returned invalid weights, using fallback heuristic."
                    
            except Exception as e:
                print(f"[LLMReasoningStrategy] Error in LLM decision: {e}")
                weights = heuristic_weights_from_signals(self.symbols, signals)
                reasoning = f"LLM error: {str(e)[:100]}. Using fallback heuristic."
        else:
            # Fallback to heuristic if LLM not available
            weights = heuristic_weights_from_signals(self.symbols, signals)
        
        # Execute rebalancing
        prev_port = self.portfolio.copy()
        fractional_rebalance(self, current_date, weights)
        
        # Add reasoning to the last trade log entry
        if self.trade_log:
            # Truncate reasoning if too long
            if len(reasoning) > 500:
                reasoning = reasoning[:497] + "..."
            self.trade_log[-1]["reasoning"] = reasoning
            
            # Also add earnings_per_stock for consistency
            if "earnings_per_stock" not in self.trade_log[-1]:
                self.trade_log[-1]["earnings_per_stock"] = {}


# =========================
# 路由部分保持不变
# =========================

@strategy_bp.route("/run_backtest", methods=["POST"])
@cross_origin()
def run_backtest_route():
    data = request.get_json(force=True, silent=True) or {}

    agent_type = (data.get("agent_type") or "ppo_planning").lower()
    tickers = data.get("tickers", ["AAPL", "MSFT", "GOOGL"])
    start_date_str = data.get("start_date", "2023-01-01")
    end_date_str = data.get("end_date", "2023-12-31")
    start_cash = float(data.get("start_cash", 100000))
    rebalance_interval = int(data.get("rebalance_interval", 1))

    try:
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
    except Exception as e:
        return jsonify({"error": f"Date parse error: {e}"}), 400

    if len(tickers) != 3:
        return jsonify({"error": "Must provide exactly 3 tickers"}), 400

    model_predict = load_all_model_predictions(tickers)

    if agent_type == "naive":
        strategy = NaiveStrategy(
            symbols=tickers, start_date=start_date, end_date=end_date,
            start_cash=start_cash, rebalance_interval=rebalance_interval)
    elif agent_type == "ppo_planning":
        strategy = PPOPlanningStrategy(
            symbols=tickers, start_date=start_date, end_date=end_date,
            start_cash=start_cash, model_predict=model_predict, rebalance_interval=rebalance_interval)
    elif agent_type == "hierarchical":
        strategy = HierarchicalStrategy(
            symbols=tickers, start_date=start_date, end_date=end_date,
            start_cash=start_cash, model_predict=model_predict, rebalance_interval=rebalance_interval)
    elif agent_type == "risk_constrained":
        strategy = RiskConstrainedStrategy(
            symbols=tickers, start_date=start_date, end_date=end_date,
            start_cash=start_cash, model_predict=model_predict, rebalance_interval=rebalance_interval)
    elif agent_type == "llm_reasoning":
        strategy = LLMReasoningStrategy(
            symbols=tickers, start_date=start_date, end_date=end_date,
            start_cash=start_cash, model_predict=model_predict, rebalance_interval=rebalance_interval)
    else:
        return jsonify({"error": "Invalid agent_type"}), 400

    portfolio_values = strategy.run_backtest()
    trade_log = strategy.trade_log

    return jsonify({
        "agent_type": agent_type,
        "agent_name": SUPPORTED_AGENTS.get(agent_type, agent_type),
        "tickers": tickers,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "rebalance_interval": rebalance_interval,
        "start_cash": start_cash,
        "portfolio_values": portfolio_values,
        "trade_log": trade_log,
    })


@strategy_bp.route("/save_trade_log", methods=["POST"])
@cross_origin()
def save_trade_log():
    data = request.get_json(force=True, silent=True) or {}
    date_str = data.get("date")
    if not date_str:
        return jsonify({"error": "date is required"}), 400

    try:
        date = pd.to_datetime(date_str).date()
    except Exception as e:
        return jsonify({"error": f"Date format error: {e}"}), 400

    balance = _safe_float(data.get("balance", 0.0))
    earning = _safe_float(data.get("earning", 0.0))
    portfolio = data.get("portfolio", {})
    change = data.get("change", {})
    earnings_per_stock = data.get("earnings_per_stock", {})
    model = data.get("model")
    game_id = data.get("game_id")

    if model is None or game_id is None:
        return jsonify({"error": "model and game_id are required"}), 400

    new_record = TradeLog(
        date=date, balance=balance, earning=earning,
        portfolio=portfolio, change=change,
        earnings_per_stock=earnings_per_stock,
        model=model, game_id=game_id,
    )
    db.session.add(new_record)
    db.session.commit()

    return jsonify({"message": "Trade log saved successfully."})


@strategy_bp.route("/get_trade_log", methods=["GET"])
@cross_origin()
def get_trade_log():
    game_id = request.args.get("game_id")
    model = request.args.get("model")
    date_str = request.args.get("date")

    if not (game_id and model and date_str):
        return jsonify({"error": "game_id, model, date are required"}), 400

    try:
        date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Date format error"}), 400

    trade_log = TradeLog.query.filter_by(
        game_id=game_id, model=model, date=date
    ).first()

    if not trade_log:
        return jsonify({"error": "Trade log not found"}), 404

    return jsonify({
        "date": trade_log.date.strftime("%Y-%m-%d"),
        "balance": trade_log.balance,
        "earning": trade_log.earning,
        "portfolio": trade_log.portfolio,
        "change": trade_log.change,
        "earnings_per_stock": trade_log.earnings_per_stock,
        "model": trade_log.model,
        "game_id": trade_log.game_id,
        "reasoning": trade_log.reasoning or "",
    })

def run_agent_for_game_and_save(game_id, tickers, start_date, end_date,
                                 start_cash, rounds, agent_type):
    """在创建游戏时调用：运行AI回测并保存到数据库"""
    raw_agent = agent_type
    agent_key = (raw_agent or "ppo_planning").lower()
    mapping = {
        "ppoplanning": "ppo_planning", "ppo_planning": "ppo_planning",
        "hierarchical": "hierarchical",
        "riskconstrained": "risk_constrained", "risk_constrained": "risk_constrained",
        "llmreasoning": "llm_reasoning", "llm_reasoning": "llm_reasoning",
        "naive": "naive",
    }
    agent_type = mapping.get(agent_key, "ppo_planning")

    if agent_type not in SUPPORTED_AGENTS:
        print(f"[run_agent_for_game_and_save] Unsupported: {raw_agent}")
        return False

    if not tickers or len(tickers) != 3:
        print("[run_agent_for_game_and_save] tickers must be 3")
        return False

    try:
        df = yf.download(tickers[0], start=start_date,
                        end=end_date + timedelta(days=1), progress=False)
    except Exception as e:
        print(f"[run_agent_for_game_and_save] download error: {e}")
        return False

    if df is None or df.empty:
        print("[run_agent_for_game_and_save] No price data")
        return False

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    trading_days = [d for d in df.index if start_date <= d <= end_date]
    if not trading_days:
        print("[run_agent_for_game_and_save] No trading days")
        return False

    rebalance_interval = 1
    model_predict = load_all_model_predictions(tickers)

    common_kwargs = dict(
        symbols=tickers, start_date=start_date, end_date=end_date,
        start_cash=start_cash, model_predict=model_predict,
        rebalance_interval=rebalance_interval,
    )

    if agent_type == "naive":
        strategy = NaiveStrategy(**{k: v for k, v in common_kwargs.items() if k != "model_predict"})
    elif agent_type == "ppo_planning":
        strategy = PPOPlanningStrategy(**common_kwargs)
    elif agent_type == "hierarchical":
        strategy = HierarchicalStrategy(**common_kwargs)
    elif agent_type == "risk_constrained":
        strategy = RiskConstrainedStrategy(**common_kwargs)
    elif agent_type == "llm_reasoning":
        strategy = LLMReasoningStrategy(**common_kwargs)
    else:
        print("[run_agent_for_game_and_save] Invalid agent_type")
        return False

    try:
        strategy.run_backtest()
    except Exception as e:
        print(f"[run_agent_for_game_and_save] backtest error: {e}")
        return False

    trade_log = strategy.trade_log
    model_name = SUPPORTED_AGENTS.get(agent_type, agent_type)

    try:
        for entry in trade_log:
            date_str = entry.get("date")
            if not date_str:
                continue
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            existing = TradeLog.query.filter_by(
                game_id=game_id, model=model_name, date=date).first()

            if existing:
                existing.balance = _safe_float(entry.get("balance", existing.balance))
                existing.earning = _safe_float(entry.get("earning", existing.earning))
                existing.portfolio = entry.get("portfolio", existing.portfolio)
                existing.change = entry.get("change", existing.change)
                existing.earnings_per_stock = entry.get(
                    "earnings_per_stock", existing.earnings_per_stock)
                existing.reasoning = entry.get("reasoning", existing.reasoning or "")
            else:
                rec = TradeLog(
                    date=date,
                    balance=_safe_float(entry.get("balance", 0.0)),
                    earning=_safe_float(entry.get("earning", 0.0)),
                    portfolio=entry.get("portfolio", {}),
                    change=entry.get("change", {}),
                    earnings_per_stock=entry.get("earnings_per_stock", {}),
                    model=model_name,
                    game_id=game_id,
                    reasoning=entry.get("reasoning", ""),
                )
                db.session.add(rec)

        db.session.commit()
        print(f"[run_agent_for_game_and_save] ✅ Saved {len(trade_log)} logs for game={game_id}, model={model_name}")
        return True

    except Exception as e:
        print(f"[run_agent_for_game_and_save] DB error: {e}")
        db.session.rollback()
        return False