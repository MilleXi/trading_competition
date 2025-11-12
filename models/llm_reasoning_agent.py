"""
LLM Reasoning Agent - GPT-4o powered trading agent with Chain-of-Thought reasoning
位置: /models/llm_reasoning_agent.py

This agent uses GPT-4o to make trading decisions with interpretable rationales.
It demonstrates:
1. Chain-of-Thought reasoning for decision making
2. Lightweight predictive signals analysis
3. Interpretable trading explanations
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
import openai

# 确保设置了API key (从环境变量读取)
# 用户需要设置: export OPENAI_API_KEY="your-key-here"
openai.api_key = os.getenv("OPENAI_API_KEY", "")

class LLMReasoningAgent:
    """
    LLM-based trading agent using GPT-4o for decision making.
    
    Features:
    - Chain-of-Thought reasoning process
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
        
        # Validate API key
        if not openai.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it before using LLM Reasoning Agent."
            )
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt that defines the agent's role and constraints."""
        return """You are an expert quantitative trading advisor with deep knowledge of financial markets, 
technical analysis, and portfolio management. Your role is to:

1. ANALYZE market conditions using provided signals (LSTM predictions, XGBoost forecasts, etc.)
2. REASON through trading decisions step-by-step (Chain-of-Thought)
3. RECOMMEND specific trades for each stock (buy/sell/hold and amounts)
4. EXPLAIN your reasoning concisely

CRITICAL CONSTRAINTS:
- Portfolio value must be maintained (cash + stock_holdings)
- Trades must be feasible given current cash and holdings
- Consider transaction costs and risk management
- Provide ONE clear recommendation per stock

OUTPUT FORMAT (JSON):
{
  "reasoning": "Your step-by-step thinking process (2-3 sentences)",
  "trades": {
    "STOCK1": {"action": "buy|sell|hold", "shares": float},
    "STOCK2": {"action": "buy|sell|hold", "shares": float},
    "STOCK3": {"action": "buy|sell|hold", "shares": float}
  },
  "risk_assessment": "brief risk summary",
  "confidence": 0.0-1.0
}"""

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
        
        Args:
            date: Current trading date
            symbols: List of stock symbols
            prices: Current prices for each stock
            portfolio: Current holdings {symbol: shares}
            cash: Available cash
            signals: Predictive signals {model_name: {symbol: prediction}}
            market_context: Optional additional market information
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
        
        prompt += """
Please analyze this information and provide:
1. Your Chain-of-Thought reasoning about market conditions
2. Specific trade recommendations for EACH stock
3. Risk assessment and confidence level

Remember: Output must be valid JSON matching the specified format."""
        
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
        
        Args:
            date: Current trading date
            symbols: List of stock symbols to trade
            prices: Current prices {symbol: price}
            portfolio: Current holdings {symbol: shares}
            cash: Available cash
            signals: Predictive signals {model: {symbol: prediction}}
            market_context: Optional additional context
            
        Returns:
            Tuple of (trades_dict, reasoning_text)
            trades_dict format: {symbol: {"action": str, "shares": float}}
        """
        try:
            # Build prompts
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(
                date, symbols, prices, portfolio, cash, signals, market_context
            )
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            
            # Parse response
            content = response.choices[0].message.content
            result = json.loads(content)
            
            trades = result.get("trades", {})
            reasoning = result.get("reasoning", "No reasoning provided")
            risk = result.get("risk_assessment", "")
            confidence = result.get("confidence", 0.5)
            
            # Format reasoning with additional context
            full_reasoning = f"{reasoning}"
            if risk:
                full_reasoning += f" Risk: {risk}."
            if confidence:
                full_reasoning += f" (Confidence: {confidence:.0%})"
            
            # Validate and format trades
            formatted_trades = {}
            for sym in symbols:
                if sym in trades:
                    trade = trades[sym]
                    action = trade.get("action", "hold").lower()
                    shares = float(trade.get("shares", 0))
                    formatted_trades[sym] = {"action": action, "shares": shares}
                else:
                    # Default to hold if not specified
                    formatted_trades[sym] = {"action": "hold", "shares": 0}
            
            return formatted_trades, full_reasoning
            
        except openai.error.AuthenticationError:
            raise ValueError(
                "OpenAI API authentication failed. Please check your OPENAI_API_KEY."
            )
        except openai.error.RateLimitError:
            print("[LLM Agent] Rate limit hit, using conservative default...")
            return self._get_default_decision(symbols), "Rate limit exceeded - holding positions"
        except json.JSONDecodeError as e:
            print(f"[LLM Agent] Failed to parse JSON response: {e}")
            print(f"Response content: {content if 'content' in locals() else 'N/A'}")
            return self._get_default_decision(symbols), "JSON parsing error - holding positions"
        except Exception as e:
            print(f"[LLM Agent] Error: {e}")
            return self._get_default_decision(symbols), f"Error occurred: {str(e)} - holding positions"
    
    def _get_default_decision(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Return a safe default decision (hold all) when API calls fail."""
        return {sym: {"action": "hold", "shares": 0} for sym in symbols}
    
    def test_connection(self) -> bool:
        """Test if OpenAI API is accessible."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            print(f"[LLM Agent] Connection test failed: {e}")
            return False


# Standalone function for easy import in strategy_routes.py
def create_llm_agent() -> LLMReasoningAgent:
    """Factory function to create an LLM agent instance."""
    return LLMReasoningAgent(
        model_name="gpt-4o",
        max_tokens=500,
        temperature=0.7
    )
