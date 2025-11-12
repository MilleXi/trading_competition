# backend/routes/game_routes.py
# 简化版本:只预计算用户选择的agent

import uuid
import os
import pickle
import math
import threading
from datetime import datetime, timedelta
from typing import Dict, List

from flask import Blueprint, request, jsonify, current_app
from flask_cors import cross_origin
import yfinance as yf
import pandas as pd
import numpy as np
from .strategy_routes import run_agent_for_game_and_save
from utils.db_utils import db, TradeLog

game_bp = Blueprint("game", __name__)

# In-memory game registry
GAMES = {}
GAME_INFO = {}

# 游戏状态跟踪
GAME_STATUS = {}

# Agent specifications
AGENT_SPECS = [
    {
        "id": "ppo_planning",
        "name": "RL Planning Agent (PPO)",
        "description": "Optimizes long-horizon returns under transaction costs using PPO.",
    },
    {
        "id": "hierarchical",
        "name": "Hierarchical Decision Agent",
        "description": "High-level portfolio allocation with low-level execution (Hierarchical RL).",
    },
    {
        "id": "risk_constrained",
        "name": "Risk-Constrained Agent",
        "description": "Actor—Critic with CVaR/MDD-style risk constraints for conservative planning.",
    },
    {
        "id": "llm_reasoning",
        "name": "LLM Reasoning Agent",
        "description": "Uses ensemble signals with interpretable rationales for its decisions.",
    },
]

# ============== Helper Functions ==============

def _parse_date(date_str, default=None):
    if not date_str:
        return default
    text = str(date_str).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", ""))
    except ValueError:
        pass
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return default


def _default_start_date():
    return datetime(2019, 1, 1)


def _default_end_date():
    return datetime(2024, 1, 1)


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


# ============== AI Agent Runner (Background) ==============

def run_agent_in_background(
    app,
    game_id: str,
    tickers: List[str],
    start_date: datetime,
    end_date: datetime,
    start_cash: float,
    rounds: int,
    agent_type: str,
):
    """
    在后台线程中运行用户选择的AI agent,更新GAME_STATUS
    
    参数:
    - agent_type: 用户选择的agent类型
    """
    print(f"[Background] Starting AI agent '{agent_type}' for game {game_id}")
    
    # 更新状态为 running
    GAME_STATUS[game_id] = {
        "status": "running",
        "progress": 0,
        "error": "",
        "started_at": datetime.utcnow().isoformat() + "Z"
    }
    
    try:
        with app.app_context():
            # 更新进度到 10%
            GAME_STATUS[game_id]["progress"] = 10
            
            # 运行agent
            success = run_agent_for_game_and_save(
                game_id=game_id,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                start_cash=start_cash,
                rounds=rounds,
                agent_type=agent_type,
            )
            
            if success:
                # 完成
                GAME_STATUS[game_id] = {
                    "status": "completed",
                    "progress": 100,
                    "error": "",
                    "completed_at": datetime.utcnow().isoformat() + "Z"
                }
                print(f"[Background] AI agent '{agent_type}' completed successfully for game {game_id}")
            else:
                # 失败
                GAME_STATUS[game_id] = {
                    "status": "failed",
                    "progress": 0,
                    "error": "Agent execution failed. Check server logs.",
                    "failed_at": datetime.utcnow().isoformat() + "Z"
                }
                print(f"[Background] AI agent '{agent_type}' failed for game {game_id}")
                
    except Exception as e:
        # 异常
        GAME_STATUS[game_id] = {
            "status": "failed",
            "progress": 0,
            "error": str(e),
            "failed_at": datetime.utcnow().isoformat() + "Z"
        }
        print(f"[Background] AI agent '{agent_type}' error for game {game_id}: {e}")


# ============== Routes ==============

@game_bp.route("/game/agents", methods=["GET"])
@cross_origin()
def list_agents():
    """返回可用的agent列表"""
    return jsonify(AGENT_SPECS)


@game_bp.route("/game/create", methods=["POST"])
@cross_origin()
def create_game():
    """
    创建一局 Trading Wars 游戏,并在后台运行用户选择的AI agent
    """
    payload = request.get_json(force=True, silent=True) or {}

    player_name = payload.get("player_name", "Player")
    tickers = payload.get("tickers") or []
    rounds_ = int(payload.get("rounds", 10))
    start_cash = float(payload.get("start_cash", 100000))
    agent_type = payload.get("agent_type", "ppo_planning")

    if len(tickers) != 3:
        return jsonify({"error": "Please choose exactly 3 tickers."}), 400

    valid_agent_ids = {a["id"] for a in AGENT_SPECS}
    if agent_type not in valid_agent_ids:
        return jsonify({"error": f"Unknown agent_type '{agent_type}'."}), 400

    start_date = _parse_date(payload.get("start_date"), _default_start_date())
    end_date = _parse_date(payload.get("end_date"), _default_end_date())
    if not start_date or not end_date or start_date >= end_date:
        return jsonify({"error": "Invalid start_date/end_date."}), 400

    game_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat() + "Z"

    info = {
        "id": game_id,
        "player_name": player_name,
        "tickers": tickers,
        "rounds": rounds_,
        "start_cash": start_cash,
        "agent_type": agent_type,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "created_at": created_at,
        "current_round": 0,
    }
    GAMES[game_id] = info

    # 初始化状态为pending
    GAME_STATUS[game_id] = {
        "status": "pending",
        "progress": 0,
        "error": ""
    }

    # 启动后台线程运行选中的agent
    print(f"[Game Create] Starting background AI agent '{agent_type}' for game {game_id}...")
    thread = threading.Thread(
        target=run_agent_in_background,
        args=(
            current_app._get_current_object(),
            game_id, 
            tickers, 
            start_date, 
            end_date, 
            start_cash, 
            rounds_, 
            agent_type
        ),
        daemon=True
    )
    thread.start()

    # 立即返回游戏信息(包含状态)
    response = info.copy()
    response["status"] = GAME_STATUS[game_id]["status"]
    response["progress"] = GAME_STATUS[game_id]["progress"]
    
    return jsonify(response)


@game_bp.route("/game/<game_id>", methods=["GET"])
@cross_origin()
def get_game(game_id):
    """
    查询某一局游戏的配置和状态
    """
    info = GAMES.get(game_id)
    if not info:
        return jsonify({"error": "Game not found"}), 404
    
    # 添加状态信息
    response = info.copy()
    status_info = GAME_STATUS.get(game_id, {
        "status": "unknown", 
        "progress": 0, 
        "error": ""
    })
    response.update(status_info)
    
    return jsonify(response)


@game_bp.route("/game/<game_id>/status", methods=["GET"])
@cross_origin()
def get_game_status(game_id):
    """
    查询游戏的AI处理状态(供前端轮询)
    """
    if game_id not in GAMES:
        return jsonify({"error": "Game not found"}), 404
    
    status_info = GAME_STATUS.get(game_id, {
        "status": "unknown",
        "progress": 0,
        "error": "No status information available"
    })
    
    return jsonify(status_info)


@game_bp.route("/game/last_trading_day", methods=["GET"])
@cross_origin()
def get_last_trading_day():
    """
    基于 SPY 获取往前第 n 个交易日
    """
    current_date_str = request.args.get("current_date")
    n = int(request.args.get("n", 1))

    current_date = _parse_date(current_date_str, datetime.utcnow())

    start_cal = (current_date - timedelta(days=365)).strftime("%Y-%m-%d")
    end_cal = current_date.strftime("%Y-%m-%d")
    df = yf.download("SPY", start=start_cal, end=end_cal, progress=False)

    if df is None or df.empty:
        return jsonify({"error": "Failed to fetch trading calendar."}), 500

    trading_days = df.index[df.index < current_date]
    if len(trading_days) < n:
        return jsonify({"error": "Not enough trading days."}), 400

    last_trading_day = trading_days[-n]
    return jsonify({"last_trading_day": last_trading_day.strftime("%Y-%m-%d")})


@game_bp.route("/next_trading_day", methods=["POST", "OPTIONS"])
@cross_origin()
def next_trading_day():
    """
    获取当前日期之后的第 n 个交易日
    """
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    data = request.get_json(force=True, silent=True) or {}
    current_date_str = data.get("current_date")
    n = int(data.get("n", 1))

    if not current_date_str:
        return jsonify({"error": "Missing current_date parameter"}), 400

    current_date = _parse_date(current_date_str)
    if current_date is None:
        return jsonify({"error": f"Invalid current_date format: {current_date_str}"}), 400

    start_cal = current_date.strftime("%Y-%m-%d")
    end_cal = (current_date + timedelta(days=365)).strftime("%Y-%m-%d")
    
    try:
        df = yf.download("SPY", start=start_cal, end=end_cal, progress=False)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch trading calendar: {str(e)}"}), 500

    if df is None or df.empty:
        return jsonify({"error": "No trading days available"}), 500

    trading_days = [d for d in df.index if d > current_date]
    
    if len(trading_days) < n:
        return jsonify({"error": f"Not enough future trading days. Requested: {n}, Available: {len(trading_days)}"}), 400

    next_trading_day = trading_days[n - 1]
    
    return jsonify({"next_trading_day": next_trading_day.strftime("%Y-%m-%d")})


@game_bp.route("/last_trading_day", methods=["POST", "OPTIONS"])
@cross_origin()
def last_trading_day():
    """
    获取当前日期之前的第 n 个交易日
    """
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    data = request.get_json(force=True, silent=True) or {}
    current_date_str = data.get("current_date")
    n = int(data.get("n", 1))

    if not current_date_str:
        return jsonify({"error": "Missing current_date parameter"}), 400

    current_date = _parse_date(current_date_str)
    if current_date is None:
        return jsonify({"error": f"Invalid current_date format: {current_date_str}"}), 400

    start_cal = (current_date - timedelta(days=365)).strftime("%Y-%m-%d")
    end_cal = current_date.strftime("%Y-%m-%d")
    
    try:
        df = yf.download("SPY", start=start_cal, end=end_cal, progress=False)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch trading calendar: {str(e)}"}), 500

    if df is None or df.empty:
        return jsonify({"error": "No trading days available"}), 500

    previous_trading_days = [d for d in df.index if d < current_date]
    
    if len(previous_trading_days) < n:
        return jsonify({"error": f"Not enough past trading days. Requested: {n}, Available: {len(previous_trading_days)}"}), 400

    last_trading_day = previous_trading_days[-n]
    
    return jsonify({"last_trading_day": last_trading_day.strftime("%Y-%m-%d")})


# ========== Game Info 接口 ==========

@game_bp.route("/game_info", methods=["POST"])
@cross_origin()
def upsert_game_info():
    data = request.get_json() or {}
    key = (data.get("game_id"), str(data.get("user_id")))
    GAME_INFO[key] = data
    return jsonify({"message": "ok"})


@game_bp.route("/game_info", methods=["GET"])
@cross_origin()
def get_game_info():
    game_id = request.args.get("game_id")
    user_id = request.args.get("user_id")
    key = (game_id, str(user_id))
    info = GAME_INFO.get(key)
    if not info:
        return jsonify([]), 200
    return jsonify([info])