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
        return """You are an expert quantitative trading advisor with deep knowledge of financial markets,
    technical analysis, and portfolio management. Your role is to:

    1) ANALYZE market conditions using provided signals (LSTM predictions, XGBoost forecasts, etc.)
    2) Think through the decision internally step-by-step, but DO NOT reveal your chain-of-thought.
    3) RECOMMEND specific trades for each stock (buy/sell/hold and amounts)
    4) EXPLAIN your reasoning concisely in 2–3 sentences

    CONSTRAINTS:
    - Portfolio value must be maintained (cash + stock_holdings)
    - Trades must be feasible given current cash and holdings
    - Consider transaction costs and risk management
    - Provide ONE clear recommendation per stock

    OUTPUT FORMAT (JSON):
    {
    "reasoning": "Concise rationale (2-3 sentences; no chain-of-thought)",
    "trades": {
        "STOCK1": {"action": "buy|sell|hold", "shares": 0},
        "STOCK2": {"action": "buy|sell|hold", "shares": 0},
        "STOCK3": {"action": "buy|sell|hold", "shares": 0}
    },
    "risk_assessment": "brief risk summary",
    "confidence": 0.0
    }
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
        Get trading decisions from GPT-4o.

        Returns:
            (trades_dict, reasoning_text)
        """
        try:
            # Build prompts
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(
                date, symbols, prices, portfolio, cash, signals, market_context
            )

            # Call OpenAI API (new SDK)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                # 要求 JSON 输出（模型需支持）
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            trades = result.get("trades", {})
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

            # Validate and format trades
            formatted_trades = {}
            for sym in symbols:
                if sym in trades:
                    trade = trades[sym] or {}
                    action = str(trade.get("action", "hold")).lower()
                    try:
                        shares = float(trade.get("shares", 0))
                    except Exception:
                        shares = 0.0
                    formatted_trades[sym] = {"action": action, "shares": shares}
                else:
                    formatted_trades[sym] = {"action": "hold", "shares": 0.0}

            return formatted_trades, full_reasoning

        except AuthenticationError:
            raise ValueError(
                "OpenAI API authentication failed. Please check your OPENAI_API_KEY."
            )
        except RateLimitError:
            print("[LLM Agent] Rate limit hit, using conservative default...")
            return self._get_default_decision(symbols), "Rate limit exceeded - holding positions"
        except APIError as e:
            print(f"[LLM Agent] OpenAI API error: {e}")
            return self._get_default_decision(symbols), f"API error - holding positions"
        except json.JSONDecodeError as e:
            print(f"[LLM Agent] Failed to parse JSON response: {e}")
            print(f"Response content: {content if 'content' in locals() else 'N/A'}")
            return self._get_default_decision(symbols), "JSON parsing error - holding positions"
        except Exception as e:
            print(f"[LLM Agent] Error: {e}")
            return self._get_default_decision(symbols), f"Error occurred: {str(e)} - holding positions"

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
