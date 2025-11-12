"""
LLM Reasoning Agent - GPT-4o powered trading agent with concise rationale
位置: /models/llm_reasoning_agent.py

This agent uses GPT-4o to make trading decisions with interpretable rationales.
It demonstrates:
1. Step-by-step INTERNAL reasoning (not revealed)
2. Lightweight predictive signals analysis
3. Interpretable, concise trading explanations
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
from openai import OpenAI, AuthenticationError, RateLimitError, APIError

# ⚠️ 不要在代码里硬编码 API Key；使用环境变量 OPENAI_API_KEY
# 用户需要设置: export OPENAI_API_KEY="your-key-here"
client = OpenAI()  # 读取环境变量中的 OPENAI_API_KEY


class LLMReasoningAgent:
    """
    LLM-based trading agent using GPT-4o for decision making.

    Features:
    - INTERNAL step-by-step reasoning (only concise rationale is returned)
    - Interprets market signals and technical indicators
    - Provides human-readable explanations for each trade
    - Considers portfolio state and risk management
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        max_tokens: int = 500,
        temperature: float = 0.7
    ):
        """
        Initialize the LLM agent.

        Args:
            model_name: OpenAI model to use (default: gpt-4o)
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature (0-1)
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Validate API key presence via env var
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it before using LLM Reasoning Agent."
            )

    def _build_system_prompt(self) -> str:
        return """You are an expert quantitative trading advisor.

    MANDATES (be decisive):
    - Prefer proposing trades over holding everything.
    - Default cash weight between 10% and 30% unless there is a strong risk reason.
    - If signals are weak or mixed, still propose small exploratory positions rather than doing nothing.
    - Each rebalance should change at least a small notional per chosen stock (e.g., >= $1,000) when you decide buy/sell.

    INPUTS:
    - You will see LSTM and XGBoost predictions per stock (they may be small numbers).
    - You will see current holdings, cash, and prices.

    WHAT TO OUTPUT (JSON). You may EITHER:
    A) Preferred: target portfolio weights that sum to <= 1 (last component is cash), or
    B) Per-stock trades in shares.

    EXACT OUTPUT FORMAT (JSON):
    {
    "reasoning": "2-3 sentences; concise; no chain-of-thought.",
    "risk_assessment": "brief risk summary",
    "confidence": 0.0,

    "target_weights": {
        "AAPL": 0.00,
        "MSFT": 0.00,
        "GOOGL": 0.00,
        "CASH": 0.20
    },

    "trades": {
        "AAPL": {"action": "buy|sell|hold", "shares": 0},
        "MSFT": {"action": "buy|sell|hold", "shares": 0},
        "GOOGL": {"action": "buy|sell|hold", "shares": 0}
    }
    }

    POLICY:
    - Prefer filling 'target_weights'. If you fill it, ensure CASH weight keeps total <= 1.
    - If you choose 'trades', avoid zero-share trades when action is buy/sell; size them realistically given cash and prices.
    - Avoid 'hold' for all three unless you explicitly justify extreme risk in 'risk_assessment'.
    """

    def _build_user_prompt(
        self,
        date: str,
        symbols: List[str],
        prices: Dict[str, float],
        portfolio: Dict[str, float],
        cash: float,
        signals: Dict[str, Dict[str, float]],
        market_context: Dict = None
    ) -> str:
        """
        Build the user prompt with current market state and signals.
        """
        # Calculate portfolio value
        portfolio_value = sum(portfolio.get(s, 0) * prices.get(s, 0) for s in symbols)
        total_value = cash + portfolio_value

        # Format signals
        signal_text = ""
        for model, preds in signals.items():
            signal_text += f"\n{model.upper()} predictions:\n"
            for sym in symbols:
                pred = preds.get(sym, 0.0)
                signal_text += f"  {sym}: {pred:+.4f}\n"

        # Build portfolio summary
        portfolio_text = ""
        for sym in symbols:
            shares = portfolio.get(sym, 0)
            value = shares * prices.get(sym, 0)
            weight = (value / total_value * 100) if total_value > 0 else 0
            portfolio_text += f"  {sym}: {shares:.2f} shares @ ${prices.get(sym, 0):.2f} = ${value:.2f} ({weight:.1f}%)\n"

        prompt = f"""Trading Decision Request for {date}

CURRENT PORTFOLIO STATE:
Cash: ${cash:.2f}
Portfolio Value: ${portfolio_value:.2f}
Total Assets: ${total_value:.2f}

HOLDINGS:
{portfolio_text}

CURRENT PRICES:
"""
        for sym in symbols:
            prompt += f"  {sym}: ${prices.get(sym, 0):.2f}\n"

        prompt += f"\nPREDICTIVE SIGNALS:{signal_text}"

        if market_context:
            prompt += f"\nADDITIONAL CONTEXT:\n{json.dumps(market_context, indent=2)}\n"

        prompt += (
            "\nPlease analyze this information and provide:\n"
            "1) A concise rationale (2–3 sentences, no chain-of-thought)\n"
            "2) Specific trade recommendations for EACH stock\n"
            "3) Risk assessment and confidence level\n\n"
            "Remember: Output must be valid JSON matching the specified format."
        )

        return prompt

    def get_trading_decision(
        self,
        date: str,
        symbols: List[str],
        prices: Dict[str, float],
        portfolio: Dict[str, float],
        cash: float,
        signals: Dict[str, Dict[str, float]],
        market_context: Dict = None
    ) -> Tuple[Dict[str, Dict[str, float]], str]:
        """
        Returns:
            (result_dict, reasoning_text)

        result_dict keys:
            - "trades": {sym: {"action": ..., "shares": ...}}
            - "target_weights": Optional[{sym: w, ..., "CASH": w}]
        """
        try:
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(
                date, symbols, prices, portfolio, cash, signals, market_context
            )

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            reasoning = result.get("reasoning", "No reasoning provided")
            risk = result.get("risk_assessment", "")
            confidence = result.get("confidence", 0.5)
            full_reasoning = f"{reasoning}"
            if risk:
                full_reasoning += f" Risk: {risk}."
            if confidence is not None:
                try:
                    full_reasoning += f" (Confidence: {float(confidence):.0%})"
                except Exception:
                    pass

            # ---- Parse target_weights if present
            tw = result.get("target_weights") or {}
            target_weights = {}
            if isinstance(tw, dict) and len(tw) > 0:
                # Keep only known symbols + CASH (case-insensitive)
                cash_key = None
                for k, v in tw.items():
                    kk = str(k).upper()
                    if kk == "CASH":
                        cash_key = k
                    elif k in symbols:
                        target_weights[k] = max(0.0, float(v))
                if cash_key is not None:
                    target_weights["CASH"] = max(0.0, float(tw[cash_key]))

            # ---- Parse trades as fallback
            trades_in = result.get("trades", {}) or {}
            trades = {}
            for sym in symbols:
                t = trades_in.get(sym, {})
                action = str(t.get("action", "hold")).lower()
                try:
                    shares = float(t.get("shares", 0.0))
                except Exception:
                    shares = 0.0
                trades[sym] = {"action": action, "shares": shares}

            return {"target_weights": target_weights, "trades": trades}, full_reasoning

        except AuthenticationError:
            raise ValueError("OpenAI API authentication failed. Please check your OPENAI_API_KEY.")
        except RateLimitError:
            print("[LLM Agent] Rate limit hit, using conservative default...")
            return {"trades": {s: {"action":"hold","shares":0.0} for s in symbols}}, "Rate limit - holding"
        except APIError as e:
            print(f"[LLM Agent] OpenAI API error: {e}")
            return {"trades": {s: {"action":"hold","shares":0.0} for s in symbols}}, "API error - holding"
        except json.JSONDecodeError as e:
            print(f"[LLM Agent] Failed to parse JSON response: {e}")
            print(f"Response content: {content if 'content' in locals() else 'N/A'}")
            return {"trades": {s: {"action":"hold","shares":0.0} for s in symbols}}, "JSON parsing error - holding"
        except Exception as e:
            print(f"[LLM Agent] Error: {e}")
            return {"trades": {s: {"action":"hold","shares":0.0} for s in symbols}}, f"Error occurred: {str(e)} - holding"

    def _get_default_decision(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Return a safe default decision (hold all) when API calls fail."""
        return {sym: {"action": "hold", "shares": 0.0} for sym in symbols}

    def test_connection(self) -> bool:
        """Test if OpenAI API is accessible."""
        try:
            _ = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=8,
                temperature=0.0
            )
            return True
        except Exception as e:
            print(f"[LLM Agent] Connection test failed: {e}")
            return False


# Standalone factory function for easy import
def create_llm_agent() -> LLMReasoningAgent:
    """Factory function to create an LLM agent instance."""
    return LLMReasoningAgent(
        model_name="gpt-4o",
        max_tokens=500,
        temperature=0.7
    )
