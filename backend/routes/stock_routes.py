import sys
import os
from datetime import datetime, timedelta

from flask import Blueprint, jsonify, request
from flask_cors import cross_origin
import yfinance as yf
import pandas as pd
import numpy as np

# 允许从项目根导入 utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from utils.db_utils import db, StockData  # type: ignore
except ImportError:
    db = None
    StockData = None

stock_bp = Blueprint("stock", __name__)

# 候选股票池（游戏选择用）
DEFAULT_STOCK_POOL = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "NFLX", "AMD", "INTC",
    "JPM", "BAC", "C", "GS", "WFC",
    "V", "MA", "DIS", "KO", "PEP",
    "PFE", "MRK", "JNJ", "XOM", "CVX",
    "BABA", "TCEHY", "SHOP", "ADBE", "ORCL",
]


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


# ========= 技术指标计算（供 DB 初始化和 yfinance 回退时复用）=========

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入: df 包含列 ['Open','High','Low','Close','Volume']
    输出: 增加一堆技术指标列（全小写，和 StockData / 前端约定一致）
    """

    if df.empty:
        return df

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # 移动均线
    df["ma5"] = close.rolling(window=5).mean()
    df["ma10"] = close.rolling(window=10).mean()
    df["ma20"] = close.rolling(window=20).mean()

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26

    # VWAP
    df["vwap"] = (close * volume).cumsum() / (volume.replace(0, np.nan).cumsum())

    # 简单均线（这里用 20 日）
    df["sma"] = close.rolling(window=20).mean()

    # 波动率/布林带
    df["std_dev"] = close.rolling(window=20).std()
    df["upper_band"] = df["sma"] + 2 * df["std_dev"]
    df["lower_band"] = df["sma"] - 2 * df["std_dev"]

    # ATR（简化版：用 high-low）
    tr = (high - low).abs()
    df["atr"] = tr.rolling(window=14).mean()

    # Sharpe Ratio（滚动 252 日，年化，简单实现）
    ret = close.pct_change()
    rolling_mean = ret.rolling(window=252, min_periods=60).mean()
    rolling_std = ret.rolling(window=252, min_periods=60).std()
    df["sharpe_ratio"] = (rolling_mean / (rolling_std + 1e-9)) * np.sqrt(252)

    return df


# ========= 接口：候选股票 =========

@stock_bp.route("/stocks/candidates", methods=["GET"])
@cross_origin()
def get_stock_candidates():
    data = [{"symbol": s, "name": s} for s in DEFAULT_STOCK_POOL]
    return jsonify(data)


# ========= 接口：历史数据（新）=========

@stock_bp.route("/stocks/history", methods=["GET"])
@cross_origin()
def get_stock_history():
    symbols_param = request.args.get("symbols", "").strip()
    if not symbols_param:
        return jsonify({"error": "Missing 'symbols' query parameter"}), 400

    symbols = [s.strip().upper() for s in symbols_param.split(",") if s.strip()]
    if not symbols:
        return jsonify({"error": "No valid symbols provided"}), 400

    start_date = _parse_date(request.args.get("start_date"))
    end_date = _parse_date(request.args.get("end_date"))
    if end_date is None:
        end_date = datetime.utcnow()
    if start_date is None:
        start_date = end_date - timedelta(days=365 * 2)

    use_db_flag = request.args.get("use_db", "").strip() == "1"

    result = {}

    for symbol in symbols:
        rows = []
        used_db = False

        # 优先从数据库取
        if StockData is not None and db is not None:
            try:
                q = (
                    StockData.query.filter(StockData.symbol == symbol)
                    .filter(StockData.date >= start_date.date())
                    .filter(StockData.date <= end_date.date())
                    .order_by(StockData.date.asc())
                )
                db_rows = q.all()
                if db_rows:
                    used_db = True
                    for r in db_rows:
                        item = {
                            "date": r.date.strftime("%Y-%m-%d"),
                            "open": float(getattr(r, "open", getattr(r, "open_price", 0.0))),
                            "high": float(getattr(r, "high", 0.0)),
                            "low": float(getattr(r, "low", 0.0)),
                            "close": float(getattr(r, "close", getattr(r, "close_price", 0.0))),
                            "volume": float(getattr(r, "volume", 0.0)),
                        }

                        # 补充技术指标字段（如果 DB 中有值）
                        for field in [
                            "ma5",
                            "ma10",
                            "ma20",
                            "rsi",
                            "macd",
                            "vwap",
                            "sma",
                            "std_dev",
                            "upper_band",
                            "lower_band",
                            "atr",
                            "sharpe_ratio",
                        ]:
                            val = getattr(r, field, None)
                            if val is not None:
                                item[field] = float(val)

                        rows.append(item)
            except Exception as e:
                print(f"[stocks/history] DB error {symbol}: {e}")

        # 回退到 yfinance：顺便计算指标
        if (not used_db) and (not use_db_flag):
            try:
                df = yf.download(
                    symbol,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                    progress=False,
                ).dropna()

                if not df.empty:
                    df = calculate_indicators(df)
                    for idx, r in df.iterrows():
                        item = {
                            "date": idx.strftime("%Y-%m-%d"),
                            "open": float(r["Open"]),
                            "high": float(r["High"]),
                            "low": float(r["Low"]),
                            "close": float(r["Close"]),
                            "volume": float(r["Volume"]),
                        }
                        for field in [
                            "ma5",
                            "ma10",
                            "ma20",
                            "rsi",
                            "macd",
                            "vwap",
                            "sma",
                            "std_dev",
                            "upper_band",
                            "lower_band",
                            "atr",
                            "sharpe_ratio",
                        ]:
                            val = r.get(field, np.nan)
                            if pd.notna(val):
                                item[field] = float(val)
                        rows.append(item)

            except Exception as e:
                print(f"[stocks/history] yfinance error {symbol}: {e}")

        result[symbol] = {"data": rows}

    return jsonify(result)


# ========= 接口：单只股票 snapshot =========

@stock_bp.route("/stocks/snapshot", methods=["GET"])
@cross_origin()
def get_stock_snapshot():
    symbol = request.args.get("symbol", "").strip().upper()
    if not symbol:
        return jsonify({"error": "Missing 'symbol' parameter"}), 400

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)

    # 尝试 DB
    if StockData is not None and db is not None:
        try:
            row = (
                StockData.query.filter(StockData.symbol == symbol)
                .filter(StockData.date <= end_date.date())
                .order_by(StockData.date.desc())
                .first()
            )
            if row:
                return jsonify(
                    {
                        "symbol": symbol,
                        "date": row.date.strftime("%Y-%m-%d"),
                        "close": float(getattr(row, "close", getattr(row, "close_price", 0.0))),
                    }
                )
        except Exception as e:
            print(f"[stocks/snapshot] DB error: {e}")

    # 回退 yfinance
    try:
        df = yf.download(
            symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
        ).dropna()
        if df.empty:
            return jsonify({"error": f"No data for {symbol}"}), 404
        last_idx = df.index[-1]
        return jsonify(
            {
                "symbol": symbol,
                "date": last_idx.strftime("%Y-%m-%d"),
                "close": float(df.loc[last_idx, "Close"]),
            }
        )
    except Exception as e:
        print(f"[stocks/snapshot] yfinance error: {e}")
        return jsonify({"error": "Failed to fetch price"}), 500



# ========= 兼容老前端的 stored_stock_data =========

@stock_bp.route("/stored_stock_data", methods=["GET"])
@cross_origin()
def stored_stock_data_legacy():
    symbol = request.args.get("symbol", "").strip().upper()
    start_date = _parse_date(request.args.get("start_date"))
    end_date = _parse_date(request.args.get("end_date"))

    if not symbol:
        return jsonify([])

    if end_date is None:
        end_date = datetime.utcnow()
    if start_date is None:
        start_date = end_date - timedelta(days=365)

    rows = []

    # 优先 DB
    if StockData is not None and db is not None:
        try:
            q = (
                StockData.query.filter(StockData.symbol == symbol)
                .filter(StockData.date >= start_date.date())
                .filter(StockData.date <= end_date.date())
                .order_by(StockData.date.asc())
            )
            for r in q.all():
                item = {
                    "date": r.date.strftime("%Y-%m-%d"),
                    "open": float(getattr(r, "open", getattr(r, "open_price", 0.0))),
                    "high": float(getattr(r, "high", 0.0)),
                    "low": float(getattr(r, "low", 0.0)),
                    "close": float(getattr(r, "close", getattr(r, "close_price", 0.0))),
                    "volume": float(getattr(r, "volume", 0.0)),
                }

                for field in [
                    "ma5",
                    "ma10",
                    "ma20",
                    "rsi",
                    "macd",
                    "vwap",
                    "sma",
                    "std_dev",
                    "upper_band",
                    "lower_band",
                    "atr",
                    "sharpe_ratio",
                ]:
                    val = getattr(r, field, None)
                    if val is not None:
                        item[field] = float(val)

                rows.append(item)
        except Exception as e:
            print(f"[stored_stock_data] DB error {symbol}: {e}")

    # 如没 DB 或没数据，用 yfinance 回退 + 计算指标
    if not rows:
        try:
            df = yf.download(
                symbol,
                start=start_date.strftime("%Y-%m-%d"),
                end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                progress=False,
            ).dropna()

            if not df.empty:
                df = calculate_indicators(df)
                for idx, r in df.iterrows():
                    item = {
                        "date": idx.strftime("%Y-%m-%d"),
                        "open": float(r["Open"]),
                        "high": float(r["High"]),
                        "low": float(r["Low"]),
                        "close": float(r["Close"]),
                        "volume": float(r["Volume"]),
                    }
                    for field in [
                        "ma5",
                        "ma10",
                        "ma20",
                        "rsi",
                        "macd",
                        "vwap",
                        "sma",
                        "std_dev",
                        "upper_band",
                        "lower_band",
                        "atr",
                        "sharpe_ratio",
                    ]:
                        val = r.get(field, np.nan)
                        if pd.notna(val):
                            item[field] = float(val)
                    rows.append(item)
        except Exception as e:
            print(f"[stored_stock_data] yfinance error {symbol}: {e}")

    return jsonify(rows)


# ========= 提供给初始化脚本使用：下载数据入库（含指标）=========

def download_stock_data():
    if db is None or StockData is None:
        print("DB/StockData not available; skip download_stock_data.")
        return

    start_date = datetime(2019, 1, 1)
    end_date = datetime.utcnow()

    for symbol in DEFAULT_STOCK_POOL:
        try:
            df = yf.download(
                symbol,
                start=start_date.strftime("%Y-%m-%d"),
                end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                progress=False,
            ).dropna()
        except Exception as e:
            print(f"[download_stock_data] yfinance error {symbol}: {e}")
            continue

        if df.empty:
            continue

        df = calculate_indicators(df)

        for idx, r in df.iterrows():
            date_val = idx.date()
            exists = StockData.query.filter_by(symbol=symbol, date=date_val).first()
            if exists:
                # 可按需更新已有记录的指标，这里简单跳过
                continue

            row = StockData(
                symbol=symbol,
                date=date_val,
                open=float(r["Open"]),
                high=float(r["High"]),
                low=float(r["Low"]),
                close=float(r["Close"]),
                volume=int(r["Volume"]),
                ma5=float(r["ma5"]) if not pd.isna(r.get("ma5")) else None,
                ma10=float(r["ma10"]) if not pd.isna(r.get("ma10")) else None,
                ma20=float(r["ma20"]) if not pd.isna(r.get("ma20")) else None,
                rsi=float(r["rsi"]) if not pd.isna(r.get("rsi")) else None,
                macd=float(r["macd"]) if not pd.isna(r.get("macd")) else None,
                vwap=float(r["vwap"]) if not pd.isna(r.get("vwap")) else None,
                sma=float(r["sma"]) if not pd.isna(r.get("sma")) else None,
                std_dev=float(r["std_dev"]) if not pd.isna(r.get("std_dev")) else None,
                upper_band=float(r["upper_band"]) if not pd.isna(r.get("upper_band")) else None,
                lower_band=float(r["lower_band"]) if not pd.isna(r.get("lower_band")) else None,
                atr=float(r["atr"]) if not pd.isna(r.get("atr")) else None,
                sharpe_ratio=float(r["sharpe_ratio"]) if not pd.isna(r.get("sharpe_ratio")) else None,
            )
            db.session.add(row)

        db.session.commit()

    print("download_stock_data: completed with indicators.")
