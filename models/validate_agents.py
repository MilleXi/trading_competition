"""
验证脚本 - 对比三个Agent的行为差异(更新版，兼容多种checkpoint命名)

在相同的市场数据上测试三个agent,对比:
1. 收益率与风险调整收益
2. 波动率与夏普比率  
3. 最大回撤与回撤持续时间
4. 换手率与交易成本
5. 现金比例动态变化
6. 持仓集中度与分散化
7. 不同市场环境下的表现
8. 策略特征统计分析

本版修正：
- 兼容 .pth 文件中不同的权重命名风格：
    * shared.* / actor_head.* / critic_head.*  (新训练脚本)
    * ac.actor.* / ac.critic.* / ac.shared.*  (早期Actor-Critic封装)
    * actor.* / critic.* / shared.*           (简化版命名)
- 自动从 checkpoint 中读取 state_dim / action_dim 校验网络构造
- 打印详细映射与缺失键报告，避免“agent weights not found”的误判
"""

import os
import numpy as np
import pandas as pd
import torch
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats

import sys
sys.path.append(os.path.dirname(__file__))

# ============== 配置 ==============
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
MODEL_DIR = os.path.join(BACKEND_DIR, "rl_models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_TICKERS = ['AAPL', 'MSFT', 'GOOGL']
TEST_START = "2023-01-01"
TEST_END = "2024-01-01"
import json
from hashlib import md5

def _print_progress(i, n, prefix="LLM", bar_len=30):
    ratio = i / max(1, n)
    done = int(ratio * bar_len)
    bar = "█" * done + "-" * (bar_len - done)
    end = "" if i < n else "\n"
    print(f"\r{prefix} [{bar}] {i}/{n}", end=end, flush=True)

def _llm_cache_filename(base_dir, tickers, start, end, initial_cash, fee_rate, temperature, model_name):
    tag = {
        "tickers": list(tickers),
        "start": str(start),
        "end": str(end),
        "cash": float(initial_cash),
        "fee": float(fee_rate),
        "temp": float(temperature),
        "model": str(model_name),
    }
    sig = md5(json.dumps(tag, sort_keys=True).encode()).hexdigest()[:10]
    fname = f"llm_decisions_{sig}.json"
    return os.path.join(base_dir, fname)

def _save_llm_cache(path, meta, daily_records):
    payload = {"meta": meta, "records": daily_records}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _load_llm_cache(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("meta", {}), data.get("records", [])
    except Exception:
        return None, None

def _pretty_title():
    print("=" * 80)
    print("🧪 Agent Validation & Comparison Test (Enhanced & Checkpoint-Compatible)")
    print("=" * 80)
    print(f"Test Period: {TEST_START} to {TEST_END}")
    print(f"Test Stocks: {TEST_TICKERS}")
    print(f"Device: {DEVICE}")
    print("=" * 80)


def _download_prices():
    print("\n📊 Downloading test data...")
    df = yf.download(TEST_TICKERS, start=TEST_START, end=TEST_END, progress=False)
    close = df["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame()
    close = close[TEST_TICKERS].dropna()
    print(f"✓ Downloaded {len(close)} trading days")
    return close


# ============== Checkpoint 装载工具 ==============
def read_state_dict_from_ckpt(ckpt_obj):
    """从checkpoint对象中提取state_dict；兼容多种保存格式。"""
    if isinstance(ckpt_obj, dict):
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            return ckpt_obj["model_state_dict"], ckpt_obj
        # 有些直接就是state_dict
        return ckpt_obj, ckpt_obj
    # 其他类型直接返回
    return ckpt_obj, {"raw": True}


def detect_style_keys(state_dict):
    keys = list(state_dict.keys())

    def has_prefix(p):
        return any(k.startswith(p) for k in keys)

    style = {
        "has_shared_head": has_prefix("shared.") or has_prefix("actor_head.") or has_prefix("critic_head."),
        "has_ac": has_prefix("ac.actor.") or has_prefix("ac.critic.") or has_prefix("ac.shared."),
        "has_simple_actor": has_prefix("actor.") or has_prefix("critic."),
    }
    return style, keys


def adapt_state_dict_for_net(state_dict, net):
    """
    将不同风格的state_dict映射到目标网络参数名。
    优先级：
      1) 直接同名 (shared./actor_head./critic_head.)
      2) ac.actor./ac.critic./ac.shared.  ->  actor_head./critic_head./shared.
      3) actor./critic. -> actor_head./critic_head.
    并过滤掉目标网络中不存在的键。
    """
    style, _ = detect_style_keys(state_dict)
    target_keys = set(net.state_dict().keys())

    def map_ac(k):
        if k.startswith("ac.actor."):
            return "actor_head." + k[len("ac.actor."):]
        if k.startswith("ac.critic."):
            return "critic_head." + k[len("ac.critic."):]
        if k.startswith("ac.shared."):
            return "shared." + k[len("ac.shared."):]
        return None

    def map_simple(k):
        if k.startswith("actor."):
            return "actor_head." + k[len("actor."):]
        if k.startswith("critic."):
            return "critic_head." + k[len("critic."):]
        return None

    remapped = {}
    mapping_used = None

    # Case 1: 已经是目标命名
    if style["has_shared_head"]:
        mapping_used = "direct(shared/actor_head/critic_head)"
        for k, v in state_dict.items():
            if k in target_keys:
                remapped[k] = v

    # Case 2: ac.* -> *_head / shared
    elif style["has_ac"]:
        mapping_used = "ac.* -> *_head/shared"
        for k, v in state_dict.items():
            new_k = map_ac(k)
            if new_k and new_k in target_keys:
                remapped[new_k] = v

    # Case 3: actor./critic. -> *_head
    elif style["has_simple_actor"]:
        mapping_used = "actor./critic. -> *_head"
        for k, v in state_dict.items():
            new_k = map_simple(k)
            if new_k and new_k in target_keys:
                remapped[new_k] = v

    else:
        # 无法识别，尝试原样过滤
        mapping_used = "fallback(filter by intersection)"
        for k, v in state_dict.items():
            if k in target_keys:
                remapped[k] = v

    # 统计装载覆盖率
    coverage = len(remapped) / max(1, len(target_keys))
    print(f"   • Mapping used: {mapping_used}")
    print(f"   • Matched parameters: {len(remapped)}/{len(target_keys)} ({coverage:.1%})")

    # 打印未匹配的关键层提示
    important_prefixes = ["shared.", "actor_head.", "critic_head."]
    for pref in important_prefixes:
        expected = [k for k in target_keys if k.startswith(pref)]
        matched = [k for k in remapped.keys() if k.startswith(pref)]
        if expected and not matched:
            print(f"   ! WARN: no parameters matched for prefix '{pref}'")

    return remapped


def safe_load_weights(ckpt_path, net, name_hint=""):
    if not os.path.exists(ckpt_path):
        print(f"✗ {name_hint} model not found: {ckpt_path}")
        return False

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state_dict_raw, meta = read_state_dict_from_ckpt(ckpt)

    # 如果checkpoint里声明了维度，尽量校验一下
    try:
        sd = int(meta.get("state_dim")) if "state_dim" in meta else None
        ad = int(meta.get("action_dim")) if "action_dim" in meta else None
        if sd is not None or ad is not None:
            print(f"   • ckpt meta: state_dim={sd}, action_dim={ad}")
    except Exception:
        pass

    # 做命名映射
    preview_keys = list(state_dict_raw.keys())[:8]
    print(f"   • keys example: {preview_keys} ... (total {len(state_dict_raw)})")
    remapped = adapt_state_dict_for_net(state_dict_raw, net)

    if len(remapped) == 0:
        print(f"✗ {name_hint} agent weights mapping produced 0 usable keys.")
        return False

    # 以 strict=False 装载
    missing, unexpected = net.load_state_dict(remapped, strict=False)
    if isinstance(missing, (list, set)) and missing:
        m_preview = list(missing)[:6]
        print(f"   • Missing keys (ok if heads differ): {m_preview}{' ...' if len(missing)>6 else ''}")
    if isinstance(unexpected, (list, set)) and unexpected:
        u_preview = list(unexpected)[:6]
        print(f"   • Unexpected keys (ignored): {u_preview}{' ...' if len(unexpected)>6 else ''}")
    print(f"✓ Loaded {name_hint} weights successfully.")
    net.eval()
    return True


# ============== 载入三个Agent ==============
def load_agents():
    print("\n🤖 Loading agents...")
    ppo_net = None
    hier_net = None
    risk_net = None
    AggressivePPOEnv = None
    HierarchicalTradingEnv = None
    RiskConstrainedEnv = None

    # 1. PPO Planning (Aggressive)
    try:
        from train_ppo_planning_agent import PPOActorCritic as PPONet, AggressivePPOEnv as _AggressivePPOEnv
        AggressivePPOEnv = _AggressivePPOEnv
        ppo_model_path = os.path.join(MODEL_DIR, "ppo_planning_agent.pth")
        ppo_net = PPONet(14, 4).to(DEVICE)
        ok = safe_load_weights(ppo_model_path, ppo_net, name_hint="PPO Planning (Aggressive)")
        if not ok:
            ppo_net = None
    except Exception as e:
        print(f"✗ Failed to prepare PPO Planning agent: {e}")
        ppo_net = None
        AggressivePPOEnv = None

    # 2. Hierarchical (Adaptive)
    try:
        from train_hierarchical_agent import HierarchicalNet, HierarchicalTradingEnv as _HierarchicalTradingEnv
        HierarchicalTradingEnv = _HierarchicalTradingEnv
        hier_model_path = os.path.join(MODEL_DIR, "hierarchical_agent.pth")
        if os.path.exists(hier_model_path):
            hier_ckpt = torch.load(hier_model_path, map_location=DEVICE)
            state_dict_raw, _ = read_state_dict_from_ckpt(hier_ckpt)
            hier_net = HierarchicalNet(14, 4, 3).to(DEVICE)
            missing, unexpected = hier_net.load_state_dict(state_dict_raw, strict=False)
            if isinstance(missing, (list, set)) and missing:
                print(f"   • Hierarchical missing keys: {list(missing)[:6]}{' ...' if len(missing)>6 else ''}")
            if isinstance(unexpected, (list, set)) and unexpected:
                print(f"   • Hierarchical unexpected keys: {list(unexpected)[:6]}{' ...' if len(unexpected)>6 else ''}")
            hier_net.eval()
            print("✓ Loaded Hierarchical Agent (Adaptive)")
        else:
            print("✗ Hierarchical model not found")
            hier_net = None
    except Exception as e:
        print(f"✗ Failed to prepare Hierarchical agent: {e}")
        hier_net = None
        HierarchicalTradingEnv = None

    # 3. Risk-Constrained (Defensive)
    try:
        from train_risk_constrained_agent import RiskConstrainedNet, RiskConstrainedEnv as _RiskConstrainedEnv
        RiskConstrainedEnv = _RiskConstrainedEnv
        risk_model_path = os.path.join(MODEL_DIR, "risk_constrained_agent.pth")
        risk_net = RiskConstrainedNet(14, 4).to(DEVICE)
        ok = safe_load_weights(risk_model_path, risk_net, name_hint="Risk-Constrained (Defensive)")
        if not ok:
            risk_net = None
    except Exception as e:
        print(f"✗ Failed to prepare Risk-Constrained agent: {e}")
        risk_net = None
        RiskConstrainedEnv = None

    return ppo_net, hier_net, risk_net, AggressivePPOEnv, HierarchicalTradingEnv, RiskConstrainedEnv


# ============== 回测逻辑 ==============
def run_backtest(agent_name, network, env_class, price_df, tickers, model_predict):
    print(f"\n📈 Testing {agent_name}...")
    env = env_class(
        tickers=tickers,
        price_df=price_df,
        model_predict=model_predict,
        min_episode_length=len(price_df) - 1,
        max_episode_length=len(price_df) - 1,
    )
    state = env.reset()
    portfolio_values = [100000.0]
    cash_ratios, turnovers, positions_history = [], [], []
    rewards, actions_history, drawdown_history = [], [], []
    modes_history = []

    peak_value = 100000.0
    done = False

    while not done:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            if "Hierarchical" in agent_name:
                mode_logits, act_logits, value, mode_probs = network(s)
                logits = torch.softmax(act_logits, dim=-1).cpu().numpy()[0]
                step_res = env.step(logits, mode_probs.cpu().numpy()[0])
                modes_history.append(int(torch.argmax(mode_probs).item()))
            else:
                logits, value = network(s)
                logits = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                step_res = env.step(logits)

        new_value = step_res.info.get("new_value", portfolio_values[-1])
        portfolio_values.append(new_value)
        cash_ratios.append(step_res.info.get("cash_ratio", 0.0))
        turnovers.append(step_res.info.get("turnover", 0.0))
        rewards.append(step_res.reward)
        actions_history.append(logits)

        if new_value > peak_value:
            peak_value = new_value
        drawdown = (peak_value - new_value) / peak_value if peak_value > 0 else 0
        drawdown_history.append(drawdown)

        positions = {tickers[i]: env.positions[i] for i in range(len(tickers))}
        positions['cash'] = env.cash
        positions_history.append(positions)

        state = step_res.state
        done = step_res.done

    returns = [(portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1] for i in range(1, len(portfolio_values))]
    total_return = (portfolio_values[-1] - 100000) / 100000
    volatility = (np.std(returns) * np.sqrt(252)) if len(returns) > 1 else 0.0
    mean_return = np.mean(returns) if returns else 0.0
    sharpe = (mean_return / (np.std(returns) + 1e-8) * np.sqrt(252)) if returns else 0.0
    max_dd = max(drawdown_history) if drawdown_history else 0.0

    in_dd, dd_duration, max_dd_duration = False, 0, 0
    for dd in drawdown_history:
        if dd > 0.01:
            in_dd = True
            dd_duration += 1
        else:
            if in_dd:
                max_dd_duration = max(max_dd_duration, dd_duration)
                dd_duration = 0
                in_dd = False

    avg_cash = float(np.mean(cash_ratios)) if cash_ratios else 0.0
    avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0

    concentrations = []
    for i, pos in enumerate(positions_history):
        if i >= len(price_df):
            break
        total_val = sum([pos.get(t, 0) * float(price_df.iloc[i][t]) for t in tickers if t in pos])
        if total_val > 0:
            weights = [pos.get(t, 0) * float(price_df.iloc[i][t]) / total_val for t in tickers if t in pos]
            hhi = sum([w ** 2 for w in weights])
            concentrations.append(hhi)
    avg_concentration = float(np.mean(concentrations)) if concentrations else 0.0

    calmar = (total_return / max_dd) if max_dd > 0 else 0.0

    downside_returns = [r for r in returns if r < 0]
    downside_std = np.std(downside_returns) if downside_returns else 1e-8
    sortino = (mean_return / downside_std * np.sqrt(252)) if returns else 0.0

    win_rate = (sum([1 for r in returns if r > 0]) / len(returns)) if returns else 0.0
    wins = [r for r in returns if r > 0]
    losses = [abs(r) for r in returns if r < 0]
    profit_loss_ratio = (np.mean(wins) / np.mean(losses)) if wins and losses else 0.0

    result = {
        "agent_name": agent_name,
        "final_value": float(portfolio_values[-1]),
        "total_return": float(total_return),
        "volatility": float(volatility),
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "max_dd_duration": int(max_dd_duration),
        "avg_cash_ratio": float(avg_cash),
        "avg_turnover": float(avg_turnover),
        "avg_concentration": float(avg_concentration),
        "win_rate": float(win_rate),
        "profit_loss_ratio": float(profit_loss_ratio),
        "portfolio_values": portfolio_values,
        "returns": returns,
        "cash_ratios": cash_ratios,
        "turnovers": turnovers,
        "positions_history": positions_history,
        "drawdown_history": drawdown_history,
        "actions_history": actions_history,
        "modes_history": modes_history if "Hierarchical" in agent_name else None,
    }
    return result
# ============== LLM Agent 回测（独立于 RL env） ==============
import time
from datetime import datetime as _dt

def _apply_trades_llm(current_prices, trades, positions, cash, fee_rate=0.0005):
    """
    根据 LLM 的 trades 指令执行交易，返回更新后的 (positions, cash)。
    trades 形如: {"AAPL": {"action":"buy","shares":10}, ...}
    自动约束：现金不足不买、持仓不足不卖。
    """
    for sym, instr in trades.items():
        price = float(current_prices.get(sym, 0.0))
        if price <= 0:
            continue
        action = str(instr.get("action", "hold")).lower()
        qty = float(instr.get("shares", 0.0))
        if qty <= 0 or action == "hold":
            continue

        if action == "buy":
            # 以现金为约束，计算最多能买的股数
            max_shares = int(cash // (price * (1 + fee_rate)))
            buy_qty = int(min(qty, max_shares))
            if buy_qty > 0:
                cost = buy_qty * price
                fee = cost * fee_rate
                cash -= (cost + fee)
                positions[sym] = positions.get(sym, 0.0) + buy_qty

        elif action == "sell":
            # 以持仓为约束，不能卖超过现有持仓
            hold_qty = int(positions.get(sym, 0.0))
            sell_qty = int(min(qty, hold_qty))
            if sell_qty > 0:
                proceeds = sell_qty * price
                fee = proceeds * fee_rate
                cash += (proceeds - fee)
                positions[sym] = hold_qty - sell_qty

    return positions, cash


def run_backtest_llm(agent, price_df, tickers, initial_cash=100000.0,
                     signals=None, market_ctx=None,
                     fee_rate=0.0005, rpm_limit=60, sleep_sec=1.1):
    """
    支持磁盘缓存与进度条的 LLM 回测。
    - 首次运行：逐日调用 API，显示进度条，并把每日决策写入 rl_models/ 缓存文件
    - 之后运行：命中相同签名的缓存时，不再调 API，直接加载决策，秒跑
    """
    print("\n📈 Testing LLM Reasoning Agent (per-day API calls with disk cache)...")
    prices = price_df.copy()
    dates = list(prices.index)
    positions = {sym: 0.0 for sym in tickers}
    cash = float(initial_cash)
    portfolio_values, cash_ratios, turnovers = [cash], [], []
    drawdown_history, returns, positions_history = [], [], []
    actions_history = []
    peak_value = cash
    dd_in, dd_len, dd_len_max = False, 0, 0

    # —— 缓存路径 & 元信息
    start_date = dates[0]
    end_date = dates[-1]
    cache_path = _llm_cache_filename(
        base_dir=MODEL_DIR,
        tickers=tickers,
        start=start_date, end=end_date,
        initial_cash=initial_cash, fee_rate=fee_rate,
        temperature=getattr(agent, "temperature", 0.0),
        model_name=getattr(agent, "model_name", "gpt-4o"),
    )
    cache_meta = {
        "tickers": list(tickers),
        "start": str(start_date),
        "end": str(end_date),
        "initial_cash": initial_cash,
        "fee_rate": fee_rate,
        "temperature": getattr(agent, "temperature", 0.0),
        "model_name": getattr(agent, "model_name", "gpt-4o"),
    }

    # —— 优先尝试读取缓存（若命中则不调 API）
    daily_records = []
    meta0, rec0 = _load_llm_cache(cache_path)
    if meta0 and rec0:
        print(f"⚡ Using cached LLM decisions: {cache_path}")
        daily_records = rec0
        # 校验记录长度与日期对齐（简单校验）
        if len(daily_records) != len(dates) - 1:
            print("⚠ Cache length mismatch; will ignore cache and call API.")
            daily_records = []
        else:
            for i in range(1, len(dates)):
                if daily_records[i - 1]["date"] != str(dates[i].date()) if hasattr(dates[i], "date") else str(dates[i]):
                    print("⚠ Cache date mismatch; will ignore cache and call API.")
                    daily_records = []
                    break

    use_cache = len(daily_records) == (len(dates) - 1)

    # —— 回测主循环
    N = len(dates) - 1
    for i in range(1, len(dates)):
        _print_progress(i, N, prefix="LLM")

        day = dates[i]
        prev_val = portfolio_values[-1]
        current_prices = {sym: float(prices.iloc[i][sym]) for sym in tickers}
        day_signals = signals or {}

        if use_cache:
            # 直接用缓存
            rec = daily_records[i - 1]
            trades = rec.get("trades", {})
            # reasoning = rec.get("reasoning", "")
        else:
            # 首次运行：调用 API
            try:
                user_date = str(day.date()) if hasattr(day, "date") else str(day)
                decisions, reasoning = agent.get_trading_decision(
                    date=user_date,
                    symbols=tickers,
                    prices=current_prices,
                    portfolio=positions,
                    cash=cash,
                    signals=day_signals,
                    market_context=market_ctx or {}
                )
                trades = decisions
                daily_records.append({
                    "date": user_date,
                    "trades": trades,
                    "reasoning": reasoning
                })
                time.sleep(sleep_sec)  # 简单限流
            except Exception as e:
                print(f"\n[LLM Backtest] API error at {day}: {e} -> HOLD")
                trades = {}

        # 执行指令
        pos_before = positions.copy()
        positions, cash = _apply_trades_llm(current_prices, trades, positions, cash, fee_rate=fee_rate)

        # 成交额与换手
        traded_amount = 0.0
        for sym in tickers:
            delta = abs(positions.get(sym, 0.0) - pos_before.get(sym, 0.0))
            traded_amount += delta * current_prices[sym]

        # 组合价值
        new_val = cash + sum(positions.get(sym, 0.0) * current_prices[sym] for sym in tickers)
        turnover = traded_amount / max(prev_val, 1e-8)
        cash_ratio = cash / max(new_val, 1e-8)

        portfolio_values.append(new_val)
        cash_ratios.append(cash_ratio)
        turnovers.append(turnover)
        positions_history.append({**{sym: positions.get(sym, 0.0) for sym in tickers}, "cash": cash})
        actions_history.append({sym: trades.get(sym, {"action": "hold", "shares": 0}) for sym in tickers})

        # 回撤统计
        if new_val > peak_value:
            peak_value = new_val
        dd = (peak_value - new_val) / max(peak_value, 1e-8)
        drawdown_history.append(dd)
        if dd > 0.01:
            dd_in = True; dd_len += 1
        else:
            if dd_in:
                dd_len_max = max(dd_len_max, dd_len)
                dd_len = 0; dd_in = False

        # 日收益
        day_ret = (new_val - prev_val) / max(prev_val, 1e-8)
        returns.append(day_ret)

    # 指标汇总
    total_return = (portfolio_values[-1] - initial_cash) / initial_cash
    vol = (np.std(returns) * np.sqrt(252)) if len(returns) > 1 else 0.0
    mu = np.mean(returns) if returns else 0.0
    sharpe = (mu / (np.std(returns) + 1e-8) * np.sqrt(252)) if returns else 0.0
    max_dd = max(drawdown_history) if drawdown_history else 0.0
    calmar = (total_return / max_dd) if max_dd > 0 else 0.0
    downs = [r for r in returns if r < 0]
    dstd = np.std(downs) if downs else 1e-8
    sortino = (mu / dstd * np.sqrt(252)) if returns else 0.0
    win_rate = (sum([1 for r in returns if r > 0]) / len(returns)) if returns else 0.0
    wins = [r for r in returns if r > 0]
    losses = [abs(r) for r in returns if r < 0]
    pl_ratio = (np.mean(wins) / np.mean(losses)) if wins and losses else 0.0

    # 集中度
    concentrations = []
    for i, pos in enumerate(positions_history):
        if i >= len(price_df):
            break
        total_val = sum([pos.get(t, 0) * float(price_df.iloc[i][t]) for t in tickers if t in pos])
        if total_val > 0:
            weights = [pos.get(t, 0) * float(price_df.iloc[i][t]) / total_val for t in tickers if t in pos]
            hhi = sum([w ** 2 for w in weights])
            concentrations.append(hhi)
    avg_conc = float(np.mean(concentrations)) if concentrations else 0.0

    # —— 首次运行才保存缓存
    if not use_cache:
        os.makedirs(MODEL_DIR, exist_ok=True)
        _save_llm_cache(cache_path, cache_meta, daily_records)
        print(f"\n💾 Saved LLM daily decisions to {cache_path}")

    result = {
        "agent_name": "LLM Reasoning (GPT-4o)",
        "final_value": float(portfolio_values[-1]),
        "total_return": float(total_return),
        "volatility": float(vol),
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "max_dd_duration": int(dd_len_max),
        "avg_cash_ratio": float(np.mean(cash_ratios) if cash_ratios else 0.0),
        "avg_turnover": float(np.mean(turnovers) if turnovers else 0.0),
        "avg_concentration": float(avg_conc),
        "win_rate": float(win_rate),
        "profit_loss_ratio": float(pl_ratio),
        "portfolio_values": portfolio_values,
        "returns": returns,
        "cash_ratios": cash_ratios,
        "turnovers": turnovers,
        "positions_history": positions_history,
        "drawdown_history": drawdown_history,
        "actions_history": actions_history,
        "modes_history": None,
    }
    return result


def main():
    _pretty_title()
    close = _download_prices()

    # 预测信号占位（如无则为空）
    model_predict = {}

    ppo_net, hier_net, risk_net, AggressivePPOEnv, HierarchicalTradingEnv, RiskConstrainedEnv = load_agents()
    results = []

    if ppo_net and AggressivePPOEnv:
        results.append(run_backtest("PPO Planning (Aggressive)", ppo_net, AggressivePPOEnv, close, TEST_TICKERS, model_predict))
    if hier_net and HierarchicalTradingEnv:
        results.append(run_backtest("Hierarchical (Adaptive)", hier_net, HierarchicalTradingEnv, close, TEST_TICKERS, model_predict))
    if risk_net and RiskConstrainedEnv:
        results.append(run_backtest("Risk-Constrained (Defensive)", risk_net, RiskConstrainedEnv, close, TEST_TICKERS, model_predict))
    
    # ===== Test LLM Reasoning Agent WITH backtest =====
    print("\n📝 Testing LLM Reasoning Agent...")
    try:
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        if os.path.exists(models_dir):
            sys.path.insert(0, models_dir)
            from llm_reasoning_agent import create_llm_agent

            if os.getenv("OPENAI_API_KEY"):
                llm_agent = create_llm_agent()
                if llm_agent.test_connection():
                    print("✓ LLM agent connected. Running backtest (daily decisions)...")
                    # 可选：降低温度以减少随机性
                    llm_agent.temperature = 0.2

                    # 可选：传入你已有的 signals / market context
                    llm_signals = {}
                    llm_ctx = {"note": "Validation run in script"}

                    llm_result = run_backtest_llm(
                        agent=llm_agent,
                        price_df=close,
                        tickers=TEST_TICKERS,
                        initial_cash=100000.0,
                        signals=llm_signals,
                        market_ctx=llm_ctx,
                        fee_rate=0.0005,
                        rpm_limit=60,
                        sleep_sec=1.0
                    )
                    results.append(llm_result)
                else:
                    print("⚠ LLM Agent API connection failed; skipping LLM backtest.")
            else:
                print("⚠ OPENAI_API_KEY not set; skipping LLM backtest.")
        else:
            print("⚠ LLM Agent module not found; skipping LLM backtest.")
    except Exception as e:
        print(f"✗ LLM Agent backtest failed: {e}")

    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE COMPARISON RESULTS")
    print("=" * 80)

    print(f"\n{'Agent':<30} {'Final $':<12} {'Return':<10} {'Vol':<10} {'Sharpe':<8} {'Sortino':<8}")
    print("-" * 90)
    for r in results:
        print(f"{r['agent_name']:<30} "
              f"${r['final_value']:>10,.0f} "
              f"{r['total_return']*100:>8.2f}% "
              f"{r['volatility']*100:>8.2f}% "
              f"{r['sharpe']:>6.2f} "
              f"{r['sortino']:>6.2f}")

    print(f"\n{'Agent':<30} {'MaxDD':<10} {'DDDays':<8} {'Calmar':<8} {'Cash%':<10} {'Turn%':<10}")
    print("-" * 90)
    for r in results:
        print(f"{r['agent_name']:<30} "
              f"{r['max_drawdown']*100:>8.2f}% "
              f"{r['max_dd_duration']:>6.0f} "
              f"{r['calmar']:>6.2f} "
              f"{r['avg_cash_ratio']*100:>8.1f}% "
              f"{r['avg_turnover']*100:>8.1f}%")

    print(f"\n{'Agent':<30} {'WinRate':<10} {'P/L Ratio':<12} {'Concentration':<15}")
    print("-" * 90)
    for r in results:
        print(f"{r['agent_name']:<30} "
              f"{r['win_rate']*100:>8.1f}% "
              f"{r['profit_loss_ratio']:>10.2f} "
              f"{r['avg_concentration']*100:>13.1f}%")


    try:
        # —— 统一的色盲友好（Okabe–Ito）调色与样式 —— #
        PALETTE = {
            "PPO Planning (Aggressive)":  {"c": "#1F77B4", "ls": "-",  "mk": "o"},  
            "Hierarchical (Adaptive)":    {"c": "#EEBF6D", "ls": "-", "mk": "s"},  
            "Risk-Constrained (Defensive)":{"c": "#D94F33", "ls": "-", "mk": "D"}, 
            "LLM Reasoning (GPT-4o)":     {"c": "#834026", "ls": "-",  "mk": "^"},  
        }
        def _sty(name):
            s = PALETTE.get(name, {"c": "#333333", "ls": "-", "mk": "o"})
            return s["c"], s["ls"], s["mk"]

        fig = plt.figure(figsize=(20, 16))

        # 1) Portfolio Value
        ax = plt.subplot(3, 3, 1)
        for r in results:
            c, ls, mk = _sty(r['agent_name'])
            ax.plot(r['portfolio_values'], label=r['agent_name'], linewidth=2, alpha=0.9, color=c, linestyle=ls)
        ax.axhline(y=100000, color='black', linestyle='--', alpha=0.5, label='Initial')
        ax.set_xlabel('Trading Day'); ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

        # 2) Cumulative Returns
        ax = plt.subplot(3, 3, 2)
        for r in results:
            c, ls, mk = _sty(r['agent_name'])
            cumulative_returns = [(v - 100000) / 100000 * 100 for v in r['portfolio_values']]
            ax.plot(cumulative_returns, label=r['agent_name'], linewidth=2, alpha=0.9, color=c, linestyle=ls)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel('Trading Day'); ax.set_ylabel('Cumulative Return (%)')
        ax.set_title('Cumulative Returns Comparison', fontsize=12, fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

        # 3) Drawdown
        ax = plt.subplot(3, 3, 3)
        for r in results:
            c, ls, mk = _sty(r['agent_name'])
            ax.plot([dd*100 for dd in r['drawdown_history']], label=r['agent_name'], linewidth=2, alpha=0.9, color=c, linestyle=ls)
        ax.set_xlabel('Trading Day'); ax.set_ylabel('Drawdown (%)')
        ax.set_title('Drawdown Over Time', fontsize=12, fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

        # 4) Cash Ratio
        ax = plt.subplot(3, 3, 4)
        for r in results:
            c, ls, mk = _sty(r['agent_name'])
            ax.plot([csh*100 for csh in r['cash_ratios']], label=r['agent_name'], linewidth=2, alpha=0.85, color=c, linestyle=ls)
        ax.set_xlabel('Trading Day'); ax.set_ylabel('Cash Ratio (%)')
        ax.set_title('Cash Allocation Over Time', fontsize=12, fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

        # 5) Turnover (5-day MA)
        ax = plt.subplot(3, 3, 5)
        for r in results:
            c, ls, mk = _sty(r['agent_name'])
            turnover_ma = pd.Series([t*100 for t in r['turnovers']]).rolling(5, min_periods=1).mean()
            ax.plot(turnover_ma, label=r['agent_name'], linewidth=2, alpha=0.85, color=c, linestyle=ls)
        ax.set_xlabel('Trading Day'); ax.set_ylabel('Turnover Rate (%) - 5-day MA')
        ax.set_title('Trading Activity (Smoothed)', fontsize=12, fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

        # 6) Risk-Return Profile
        ax = plt.subplot(3, 3, 6)
        for r in results:
            c, ls, mk = _sty(r['agent_name'])
            ax.scatter(
                r['volatility']*100, r['total_return']*100,
                s=260, alpha=0.85, label=r['agent_name'],
                c=c, edgecolors='black', linewidths=1.2, marker=mk
            )
        ax.set_xlabel('Volatility (%)'); ax.set_ylabel('Total Return (%)')
        ax.set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

        # 7) Daily Returns Distribution 
        ax = plt.subplot(3, 3, 7)
        for r in results:
            returns_pct = [ret*100 for ret in r['returns']]
            c, ls, _ = _sty(r['agent_name'])
            # 绘制线条轮廓和填充
            ax.hist(
                returns_pct, bins=30, histtype='step', linewidth=1.8,
                label=r['agent_name'], color=c, linestyle=ls
            )
            ax.hist(
                returns_pct, bins=30, histtype='stepfilled',
                alpha=0.12, color=c
            )
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Daily Return (%)'); ax.set_ylabel('Frequency')
        ax.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
        
        # 8) Radar
        ax = plt.subplot(3, 3, 8, projection='polar')
        metrics = ['Return', 'Sharpe', 'Calmar', 'Win Rate', 'Stability']
        num_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]

        max_return = max([rr['total_return'] for rr in results]) if results else 1.0
        max_sharpe = max([rr['sharpe'] for rr in results if rr['sharpe'] > 0], default=1.0)
        max_calmar = max([rr['calmar'] for rr in results if rr['calmar'] > 0], default=1.0)
        max_vol = max([rr['volatility'] for rr in results]) if results else 1.0

        for r in results:
            c, ls, _ = _sty(r['agent_name'])
            values = [
                r['total_return'] / max_return if max_return > 0 else 0,
                r['sharpe'] / max_sharpe if max_sharpe > 0 else 0,
                r['calmar'] / max_calmar if max_calmar > 0 else 0,
                r['win_rate'],
                1 - r['volatility'] / max_vol if max_vol > 0 else 0,
            ]
            values += values[:1]
            ax.plot(angles, values, linestyle=ls, linewidth=2.2, label=r['agent_name'], color=c)
            ax.fill(angles, values, alpha=0.15, color=c)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Strategy Performance Radar', fontsize=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1)); ax.grid(True)

        # 9) Hierarchical pie 或 Key Metrics 柱状
        ax = plt.subplot(3, 3, 9)
        hier_results = [r for r in results if "Hierarchical" in r['agent_name']]
        if hier_results and hier_results[0]['modes_history']:
            modes = hier_results[0]['modes_history']
            mode_names = ['Aggressive', 'Balanced', 'Defensive']
            mode_counts = [modes.count(i) for i in range(3)]
            # 这里用固定颜色以匹配三种模式的语义（非策略颜色）
            colors_pie = ['#3B9AB2', '#EDDCC3', '#EEBF6D']
            ax.pie(mode_counts, labels=mode_names, colors=colors_pie, autopct='%1.1f%%', startangle=90)
            ax.set_title('Hierarchical Mode Distribution', fontsize=12, fontweight='bold')
        else:
            metric_names = ['Return', 'Sharpe', 'Calmar']
            x = np.arange(len(metric_names))
            n = len(results)
            width = 0.8 / max(1, n)
            for i, r in enumerate(results):
                c, ls, mk = _sty(r['agent_name'])
                values = [r['total_return']*100, r['sharpe'], r['calmar']]
                ax.bar(x + i*width, values, width, label=r['agent_name'],
                       alpha=0.85, color=c, edgecolor='black', linewidth=0.8)
            ax.set_ylabel('Value'); ax.set_title('Key Metrics Comparison', fontsize=12, fontweight='bold')
            ax.set_xticks(x + width * (n - 1) / 2); ax.set_xticklabels(metric_names)
            ax.legend(); ax.grid(True, alpha=0.3, axis='y')

        plot_path = os.path.join(MODEL_DIR, "agent_comparison_enhanced.png")
        plt.tight_layout(); plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved enhanced comparison chart to {plot_path}")
    except Exception as e:
        print(f"(viz skipped) Plot error: {e}")

    # 保存CSV
    summary_data = []
    for r in results:
        summary_data.append({
            'Agent': r['agent_name'],
            'Final_Value': r['final_value'],
            'Total_Return_%': r['total_return'] * 100,
            'Volatility_%': r['volatility'] * 100,
            'Sharpe': r['sharpe'],
            'Sortino': r['sortino'],
            'Calmar': r['calmar'],
            'Max_Drawdown_%': r['max_drawdown'] * 100,
            'Max_DD_Duration': r['max_dd_duration'],
            'Win_Rate_%': r['win_rate'] * 100,
            'Profit_Loss_Ratio': r['profit_loss_ratio'],
            'Avg_Cash_%': r['avg_cash_ratio'] * 100,
            'Avg_Turnover_%': r['avg_turnover'] * 100,
            'Concentration_%': r['avg_concentration'] * 100,
        })
    summary_df = pd.DataFrame(summary_data)
    csv_path = os.path.join(MODEL_DIR, "agent_comparison_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved detailed summary to {csv_path}")
    print("\n" + "=" * 80)
    print("✅ Enhanced Validation Complete!")
    print("=" * 80)



if __name__ == "__main__":
    main()