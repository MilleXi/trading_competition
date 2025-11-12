import sys
import os
from collections import defaultdict
from datetime import datetime

from flask import Blueprint, jsonify, request
from flask_cors import cross_origin

from utils.db_utils import db, Transaction

# 只定义 blueprint，不要在这里创建 Flask app
transaction_bp = Blueprint("transaction_bp", __name__)

# ---------- 获取交易记录 ----------

@transaction_bp.route("/transactions", methods=["GET"])
@cross_origin()
def get_transactions():
    """
    查询交易记录，支持按 user_id, game_id, stock_symbols, start_date, end_date 过滤。
    返回结构按日期和股票分组，方便前端展示。
    """
    user_id = request.args.get("user_id")
    game_id = request.args.get("game_id")
    stock_symbols = request.args.get("stock_symbols")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    query = Transaction.query

    if user_id:
        query = query.filter_by(user_id=user_id)

    if game_id:
        query = query.filter_by(game_id=game_id)

    if stock_symbols:
        symbols = [s.strip() for s in stock_symbols.split(",") if s.strip()]
        if symbols:
            query = query.filter(Transaction.stock_symbol.in_(symbols))

    if start_date:
        # 前端传的是 'YYYY-MM-DD'
        query = query.filter(Transaction.date >= start_date)

    if end_date:
        query = query.filter(Transaction.date <= end_date)

    transactions = query.all()

    result = {
        "user_id": user_id,
        "game_id": game_id,
        "transactions_by_date": defaultdict(lambda: defaultdict(list)),
    }

    for t in transactions:
        date_str = t.date.strftime("%Y-%m-%d")
        result["transactions_by_date"][date_str][t.stock_symbol].append(
            {
                "id": t.id,
                "transaction_type": t.transaction_type,
                "amount": float(t.amount),
            }
        )

    # 日期倒序
    sorted_data = {
        "user_id": user_id,
        "game_id": game_id,
        "transactions_by_date": dict(
            sorted(
                result["transactions_by_date"].items(),
                key=lambda x: datetime.strptime(x[0], "%Y-%m-%d"),
                reverse=True,
            )
        ),
    }

    return jsonify(sorted_data)


# ---------- 创建交易记录 ----------

@transaction_bp.route("/transactions", methods=["POST"])
@cross_origin()
def create_transaction():
    """
    前端传入:
    {
        "game_id": "...",
        "user_id": ... 或 "ai",
        "stock_symbol": "AAPL",
        "transaction_type": "buy" / "sell" / "hold" / "init",
        "amount": 10,
        "price": 123.45,  # 可选，目前只存 amount + 时间，价格前端自己展示也行
        "date": "2023-01-05T00:00:00.000Z"
    }
    """
    data = request.get_json() or {}

    raw_date = data.get("date")
    if not raw_date:
        return jsonify({"error": "Missing 'date' field"}), 400

    # 兼容 ISO 带 Z 的格式
    try:
        if raw_date.endswith("Z"):
            date = datetime.strptime(raw_date, "%Y-%m-%dT%H:%M:%S.%fZ")
        else:
            # 再尝试通用解析
            date = datetime.fromisoformat(raw_date)
    except Exception:
        return jsonify({"error": f"Incorrect date format: {raw_date}"}), 400

    try:
        new_tx = Transaction(
            game_id=data.get("game_id"),
            user_id=str(data.get("user_id")),
            stock_symbol=data.get("stock_symbol", ""),
            transaction_type=data.get("transaction_type", "hold"),
            amount=float(data.get("amount", 0)),
            date=date,
        )
        db.session.add(new_tx)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to create transaction: {e}"}), 500

    return jsonify({"message": "Transaction created successfully"}), 201


# ---------- 更新交易记录（可选） ----------

@transaction_bp.route("/transactions/<int:transaction_id>", methods=["PUT"])
@cross_origin()
def update_transaction(transaction_id):
    data = request.get_json() or {}
    tx = Transaction.query.get_or_404(transaction_id)

    raw_date = data.get("date")
    if raw_date:
        try:
            if raw_date.endswith("Z"):
                tx.date = datetime.strptime(raw_date, "%Y-%m-%dT%H:%M:%S.%fZ")
            else:
                tx.date = datetime.fromisoformat(raw_date)
        except Exception:
            return jsonify({"error": f"Incorrect date format: {raw_date}"}), 400

    tx.game_id = data.get("game_id", tx.game_id)
    tx.user_id = str(data.get("user_id", tx.user_id))
    tx.stock_symbol = data.get("stock_symbol", tx.stock_symbol)
    tx.transaction_type = data.get("transaction_type", tx.transaction_type)
    tx.amount = float(data.get("amount", tx.amount))

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to update transaction: {e}"}), 500

    return jsonify({"message": "Transaction updated successfully"})


# ---------- 删除交易记录（可选） ----------

@transaction_bp.route("/transactions/<int:transaction_id>", methods=["DELETE"])
@cross_origin()
def delete_transaction(transaction_id):
    tx = Transaction.query.get_or_404(transaction_id)

    try:
        db.session.delete(tx)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to delete transaction: {e}"}), 500

    return jsonify({"message": "Transaction deleted successfully"})
