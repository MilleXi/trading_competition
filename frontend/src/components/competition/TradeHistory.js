import React, { useEffect, useState } from 'react';
import axios from 'axios';
import '../../css/TradeHistory.css';

const TradeHistory = ({ userId, refreshHistory, selectedStock, gameId, agentType }) => {
  const [playerHistory, setPlayerHistory] = useState({});
  const [aiHistory, setAiHistory] = useState({});
  const [aiReasoning, setAiReasoning] = useState({});

  // Check if the current agent is LLM Reasoning
  const isLLMAgent = agentType === 'LLMReasoning' || agentType === 'llm_reasoning';

  useEffect(() => {
    const fetchHistory = async () => {
      console.log("fetching history")
      try {
        const playerResponse = await axios.get('http://localhost:8000/api/transactions', {
          params: {
            user_id: userId,
            game_id: gameId,
            stock_symbols: selectedStock.join(','),
          },
        });

        const aiResponse = await axios.get('http://localhost:8000/api/transactions', {
          params: {
            user_id: 'ai',
            game_id: gameId,
            stock_symbols: selectedStock.join(','),
          },
        });

        console.log('player history response data:', playerResponse.data);
        console.log('ai history response data:', aiResponse.data);

        setPlayerHistory(playerResponse.data.transactions_by_date);
        setAiHistory(aiResponse.data.transactions_by_date);

        // If LLM agent, fetch reasoning for each trading date
        if (isLLMAgent) {
          const reasoning = {};
          const dates = Object.keys(aiResponse.data.transactions_by_date);
          
          for (const date of dates) {
            try {
              const tradeLogResponse = await axios.get('http://localhost:8000/api/get_trade_log', {
                params: {
                  game_id: gameId,
                  model: 'LLMReasoning',
                  date: date
                }
              });
              
              if (tradeLogResponse.data && tradeLogResponse.data.reasoning) {
                reasoning[date] = tradeLogResponse.data.reasoning;
              }
            } catch (error) {
              console.log(`No reasoning found for ${date}`);
              reasoning[date] = "";
            }
          }
          
          setAiReasoning(reasoning);
        }
      } catch (error) {
        console.error('Error fetching trade history:', error);
      }
    };

    fetchHistory();
  }, [userId, refreshHistory, selectedStock, gameId, isLLMAgent]);

  console.log('player history:', playerHistory);
  console.log('ai history:', aiHistory);
  console.log('selectedStock:', selectedStock);
  console.log('AI reasoning:', aiReasoning);

  const renderHistory = (history, isAI = false) => (
    Object.keys(history).length > 0 ? (
      Object.keys(history).reverse().map((date, index) => (
        <tr key={index}>
          <td>{new Date(date).toLocaleDateString()}</td>
          {selectedStock.map((stock, idx) => {
            const tradeDetails = history[date][stock] || [];
            const tradeInfo = tradeDetails.length > 0
              ? tradeDetails.map(trade => {
                const type = trade.transaction_type.charAt(0).toUpperCase() + trade.transaction_type.slice(1);
                return `${type}: ${trade.amount}`;
              }).join(', ')
              : 'Hold: 0';

            return <td key={idx}>{tradeInfo}</td>;
          })}
          {/* Show reasoning column only for AI and only if LLM agent */}
          {isAI && isLLMAgent && (
            <td className="reasoning-cell">
              <div className="reasoning-text">
                {aiReasoning[date] || "No explanation"}
              </div>
            </td>
          )}
        </tr>
      ))
    ) : (
      <tr>
        <td colSpan={selectedStock.length + 1 + (isAI && isLLMAgent ? 1 : 0)} style={{ textAlign: 'center' }}>
          No trade history
        </td>
      </tr>
    )
  );

  return (
    <div>
      <h4 style={{ color: '#e0e0e0', marginBottom: '15px' }}>Trade History</h4>
      <div className="trade-history">
        <div className="history-header">
          <span className="history-title">Player</span>
          <span className="history-divider"></span>
          <span className="history-title">AI</span>
        </div>
        <div className="history-tables">
          <div className="history-table">
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  {selectedStock.map((stock, index) => (
                    <th key={index}>{stock}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {renderHistory(playerHistory, false)}
              </tbody>
            </table>
          </div>
          <div className="history-divider-vertical"></div>
          <div className="history-table">
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  {selectedStock.map((stock, index) => (
                    <th key={index}>{stock}</th>
                  ))}
                  {/* Add Reasoning column header only for LLM agent */}
                  {isLLMAgent && (
                    <th className="reasoning-header">AI Reasoning</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {renderHistory(aiHistory, true)}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      
      {/* Show legend for LLM agent */}
      {isLLMAgent && (
        <div className="llm-legend">
          <strong>LLM Agent Active:</strong> AI provides explanations for trading decisions. Scroll within the reasoning column to read full details.
        </div>
      )}
    </div>
  );
};

export default TradeHistory;