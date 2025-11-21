#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从 LLM 决策 json (如 llm_decisions_b4143e72f1.json) 中读取 trades，
截取前 0–60 天，画带符号交易强度热力图：

- x 轴：时间（按记录顺序，前 0~59 天）
- y 轴：tickers（如 AAPL / MSFT / GOOGL）
- 颜色：带符号的交易强度（+shares=买入，-shares=卖出，0=观望）

用法：
    python plot_llm_heatmap.py llm_decisions_b4143e72f1.json
"""

import sys
import os
import json
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt


# 想改截取范围就改这里
SLICE_START = 0
SLICE_END = 250  # [0, 60) → 最多 60 天


def load_llm_decisions(json_path: str):
    """读取 LLM 决策 json，返回 tickers, dates, trade_matrix（已按日期排序）."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("meta", {})
    records = data.get("records", [])

    tickers: List[str] = meta.get("tickers", [])
    if not tickers:
        raise ValueError("meta.tickers 为空，无法绘图")

    # 按 date 排序（保险起见）
    records_sorted = sorted(records, key=lambda r: r.get("date", ""))

    # 先整体排序，再做时间截取（按 index，而不是日期字符串）
    total_T = len(records_sorted)
    s = max(0, SLICE_START)
    e = min(total_T, SLICE_END)
    if e - s < 1:
        raise ValueError(f"记录数太少，无法截取区间 {SLICE_START}-{SLICE_END}（总记录数={total_T}）")

    records_slice = records_sorted[s:e]

    dates: List[str] = []
    trade_matrix = []  # 将来 shape: [n_tickers, T_slice]

    for rec in records_slice:
        date_str = rec.get("date", "")
        trades: Dict[str, Dict] = rec.get("trades", {})
        dates.append(date_str)

        row = []
        for sym in tickers:
            instr = trades.get(sym, {"action": "hold", "shares": 0.0})
            action = str(instr.get("action", "hold")).lower()
            shares = float(instr.get("shares", 0.0))

            # 带符号交易量：买=+shares，卖=-shares，hold=0
            if action == "buy":
                val = shares
            elif action == "sell":
                val = -shares
            else:
                val = 0.0

            row.append(val)

        trade_matrix.append(row)

    trade_matrix = np.array(trade_matrix, dtype=float).T  # shape: [n_tickers, T_slice]

    return tickers, dates, trade_matrix, s, e


def plot_heatmap(tickers: List[str], dates: List[str], mat: np.ndarray,
                 out_path: str, s_idx: int, e_idx: int):
    """
    绘制带符号交易强度热力图：
    - mat: shape [n_tickers, T_slice]
    - s_idx, e_idx: 原始 records 的 index 范围，用来标注标题
    """
    n_tickers, T = mat.shape

    vals = mat.flatten()
    non_zero = vals[vals != 0]

    if non_zero.size > 0:
        # 用 abs 的 95 分位数做上界，对称色阶，避免极端单日太大
        vmax = float(np.percentile(np.abs(non_zero), 95))
        if vmax <= 0:
            vmax = float(np.max(np.abs(non_zero)))
        vmax = max(vmax, 1e-6)
        vmin = -vmax
    else:
        vmax = 1.0
        vmin = -1.0

    fig, ax = plt.subplots(figsize=(max(8, T * 0.25), 3 + 0.3 * n_tickers))

    im = ax.imshow(
        mat,
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap="coolwarm",  # 红=净买入，蓝=净卖出
    )

    ax.set_yticks(range(n_tickers))
    ax.set_yticklabels(tickers)

    # x 轴刻度：只标少量
    if T <= 15:
        xticks = list(range(T))
    else:
        step = max(1, T // 8)
        xticks = list(range(0, T, step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([dates[i] for i in xticks], rotation=45, ha="right")

    ax.set_xlabel(f"Date (records index {s_idx}–{e_idx})")
    ax.set_ylabel("Ticker")
    ax.set_title("LLM Trade Intensity Heatmap (first 0–60 records)\n"
                 "+shares = buy, -shares = sell, 0 = hold")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Signed trade size (shares)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Heatmap saved to: {out_path}")


def main():
    if len(sys.argv) < 2:
        print("用法: python plot_llm_heatmap.py llm_decisions_xxx.json")
        sys.exit(1)

    json_path = sys.argv[1]
    if not os.path.exists(json_path):
        print(f"文件不存在: {json_path}")
        sys.exit(1)

    tickers, dates, mat, s_idx, e_idx = load_llm_decisions(json_path)

    base_dir = os.path.dirname(os.path.abspath(json_path))
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    out_path = os.path.join(base_dir, f"{base_name}_heatmap_{SLICE_START}_{SLICE_END}.png")

    print(f"[+] Loaded {len(dates)} records (sliced {s_idx}–{e_idx}) for tickers: {tickers}")
    plot_heatmap(tickers, dates, mat, out_path, s_idx, e_idx)


if __name__ == "__main__":
    main()
