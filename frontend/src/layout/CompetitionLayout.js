import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { useLocation } from 'react-router-dom';
import Modal from 'react-modal';
import '../css/home.css';
import '../css/competition.css';
import { AppHeader, AppFooter, AppHeaderDropdown } from '../components/index';
import { CDropdown, CDropdownItem, CDropdownMenu, CDropdownToggle } from '@coreui/react';
import CandlestickChart from '../components/competition/CandlestickChart';
import StockTradeComponent from '../components/competition/StockTrade';
import FinancialReport from '../components/competition/FinancialReport';
import TradeHistory from '../components/competition/TradeHistory';
import PointsStoreModal from '../components/competition/PointsStoreModal';
import AIPrecomputeModal from '../components/competition/AIPrecomputeModal';
import { v4 as uuidv4 } from 'uuid';
import Box from '@mui/material/Box';
import Checkbox from '@mui/material/Checkbox';
import FormControlLabel from '@mui/material/FormControlLabel';
import Grid from '@mui/material/Grid';
import Typography from '@mui/material/Typography';
import GameEndModal from '../components/competition/GameEndModal';
import Standings from '../components/competition/Standings';
import { Button } from '@mui/material';

const CompetitionLayout = () => {
  const initialBalance = 100000;
  const [startDate, setStartDate] = useState(null);
  const [gameId, setGameId] = useState(uuidv4());
  const [modelList, setModelList] = useState([]);
  const [currentRound, setCurrentRound] = useState(1);
  const [currentDate, setCurrentDate] = useState(null);
  const [selectedStock, setSelectedStock] = useState('AAPL');
  const [selectedStockList, setSelectedStockList] = useState(['AAPL', 'MSFT', 'GOOGL']);
  const [stockData, setStockData] = useState([]);
  const [selectedTrades, setSelectedTrades] = useState(
    selectedStockList.reduce((acc, stock) => ({ ...acc, [stock]: { type: 'hold', amount: '0' } }), {})
  );
  const [cash, setCash] = useState(initialBalance);
  const [portfolioValue, setPortfolioValue] = useState(0);
  const [totalAssets, setTotalAssets] = useState(initialBalance);
  const [aiCash, setAiCash] = useState(initialBalance);
  const [aiPortfolioValue, setAiPortfolioValue] = useState(0);
  const [aiTotalAssets, setAiTotalAssets] = useState(initialBalance);
  const TMinus = 60;
  const MaxRound = 10;
  const [counter, setCounter] = useState(TMinus);
  const [gameEnd, setGameEnd] = useState(false);
  const [refreshHistory, setRefreshHistory] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(true);
  const [selectedTickers, setSelectedTickers] = useState([]);
  const userId = 1;
  const [CandlestickChartData, setCandlestickChartData] = useState([]);
  const location = useLocation();
  const { difficulty } = location.state || { difficulty: 'PPOPlanning' };
  const rootElement = document.getElementById('root');
  const [aiStrategy, setAiStrategy] = useState({});
  const [showStrategyModal, setShowStrategyModal] = useState(false);
  const [stopCounter, setStopCounter] = useState(true);
  const [showPointsStore, setShowPointsStore] = useState(false);
  const [stockInfo, setStockInfo] = useState({});
  const [userInfo, setUserInfo] = useState({});
  const [showGameEndModal, setShowGameEndModal] = useState(false);

  // 新增: AI预计算相关状态
  const [showAIPrecomputeModal, setShowAIPrecomputeModal] = useState(false);
  const [aiPrecomputeComplete, setAiPrecomputeComplete] = useState(false);
  const [gameCreated, setGameCreated] = useState(false);

  const handleClosePointsStore = () => setShowPointsStore(false);
  const handleShowPointsStore = () => setShowPointsStore(true);

  Modal.setAppElement(rootElement);

  // 初始化交易日期
  useEffect(() => {
    const randomDate = new Date(2023, 0, 1 + Math.floor(Math.random() * 300));
    console.log("Random Date:", randomDate);
    
    const fetchTradingDay = async () => {
      try {
        const response = await axios.post('http://localhost:8000/api/next_trading_day', {
          current_date: randomDate.toISOString().split('T')[0],
          n: 1
        });
        const tradingDay = new Date(response.data.next_trading_day);
        setStartDate(tradingDay);
        setCurrentDate(tradingDay);
      } catch (error) {
        console.error('Error fetching next trading day:', error);
      }
    };

    fetchTradingDay();
  }, []);

  // 设置modelList
  useEffect(() => {
    setModelList([difficulty]);
  }, [difficulty]);

  // 获取股票数据
  useEffect(() => {
    const fetchStockData = async () => {
      try {
        const response = await axios.get('http://localhost:8000/api/stored_stock_data', {
          params: {
            symbol: selectedStock,
            start_date: '2021-01-01',
            end_date: '2024-01-01'
          }
        });
        setStockData(response.data);
        setCandlestickChartData(response.data.map(data => ({
          date: new Date(data.date),
          open: parseFloat(data.open),
          high: parseFloat(data.high),
          low: parseFloat(data.low),
          close: parseFloat(data.close),
        })));
      } catch (error) {
        console.error('Error fetching stock data:', error);
      }
    };

    if (selectedStock && currentDate) {
      fetchStockData();
    }
  }, [selectedStock, currentDate]);

  // 初始化游戏信息
  useEffect(() => {
    // 只有在AI预计算完成且游戏创建后才初始化
    if (!startDate || !gameCreated) return;
    
    const initializeGameInfo = async () => {
      try {
        const initialData = {
          game_id: gameId,
          user_id: userId,
          cash: initialBalance,
          portfolio_value: 0,
          total_assets: initialBalance,
          stocks: selectedStockList.reduce((acc, stock) => ({ ...acc, [stock]: 0 }), {}),
          score: 0
        };
        setUserInfo(initialData);
        
        const aiData = {
          game_id: gameId,
          user_id: 'ai',
          cash: initialBalance,
          portfolio_value: 0,
          total_assets: initialBalance,
          stocks: {},
          score: 0
        };
        
        await axios.post('http://localhost:8000/api/game_info', initialData);
        await axios.post('http://localhost:8000/api/game_info', aiData);

        const initialTransaction = {
          game_id: gameId,
          user_id: 'ai',
          stock_symbol: 'INIT',
          transaction_type: 'init',
          amount: 0,
          price: 0,
          date: startDate.toISOString()
        };
        await saveTransaction(initialTransaction);

        console.log('✓ Game info initialized');
      } catch (error) {
        console.error('Error initializing game info:', error);
      }
    };

    initializeGameInfo();
    fetchStockInfo();
  }, [startDate, gameCreated]);

  // 计时器
  useEffect(() => {
    // 只有在AI预计算完成后才开始计时
    if (!aiPrecomputeComplete) return;
    
    const timerId = setTimeout(() => {
      if (counter > 0 && !stopCounter) {
        setCounter(counter - 1);
      } else if (counter === 0 && !gameEnd) {
        if (Object.keys(selectedTrades).length > 0)
          handleSubmit();
        else
          handleNextRound();
      }
    }, 1000);
    
    return () => clearTimeout(timerId);
  }, [counter, gameEnd, stopCounter, aiPrecomputeComplete]);

  // 获取股票信息
  useEffect(() => {
    if (currentDate && aiPrecomputeComplete) {
      fetchStockInfo();
    }
  }, [currentDate, selectedTrades]);

  const tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'BAC', 'C', 'WFC', 'GS',
    'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY', 'XOM', 'CVX', 'COP', 'SLB', 'BKR',
    'DIS', 'NFLX', 'CMCSA', 'NKE', 'SBUX', 'CAT', 'DE', 'MMM', 'GE', 'HON'
  ];

  const openModal = () => {
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
  };

  const handleTickerSelection = (ticker) => {
    setSelectedTickers((prev) => {
      if (prev.includes(ticker)) {
        return prev.filter((item) => item !== ticker);
      }
      if (prev.length < 3) {
        return [...prev, ticker];
      }
      return prev;
    });
  };

  const confirmSelection = async () => {
    console.log('=== confirmSelection START ===');
    console.log('selectedTickers:', selectedTickers);
    console.log('difficulty:', difficulty);
    
    if (selectedTickers.length === 3) {
      // 更新选股状态
      setSelectedStockList(selectedTickers);
      setSelectedStock(selectedTickers[0]);
      setSelectedTrades(
        selectedTickers.reduce((acc, stock) => 
          ({ ...acc, [stock]: { type: 'hold', amount: '0' } }), {})
      );
      
      // 关闭选股模态框
      closeModal();
      
      // 显示AI预计算进度模态框
      setShowAIPrecomputeModal(true);
      
      // 检查必需数据
      if (!startDate) {
        console.error('❌ startDate is null');
        alert('Error: Please wait for initialization to complete');
        setShowAIPrecomputeModal(false);
        return;
      }
      
      try {
        
        // ✅ agent_type格式转换
        const agentTypeMap = {
          'ppoplanning': 'ppo_planning',
          'hierarchical': 'hierarchical',
          'riskconstrained': 'risk_constrained',
          'llmreasoning': 'llm_reasoning',
          'naive': 'naive'
        };
        const calculateEndDate = async (start, rounds) => {
          try {
            const response = await axios.post('http://localhost:8000/api/next_trading_day', {
              current_date: start.toISOString().split('T')[0],
              n: rounds
            });
            return new Date(response.data.next_trading_day);
          } catch (error) {
            console.error('Error calculating end date:', error);
            // Fallback: 估算 20 个日历日
            return new Date(start.getTime() + 20 * 24 * 60 * 60 * 1000);
          }
        };
        const endDate = await calculateEndDate(startDate, MaxRound);
        const normalizedDifficulty = difficulty.toLowerCase().replace(/\s+/g, '');
        const agentType = agentTypeMap[normalizedDifficulty] || 'ppo_planning';
        
        console.log('Agent Type Mapping:', difficulty, '→', agentType);
        
        const gameData = {
          player_name: "Player",
          tickers: selectedTickers,
          rounds: MaxRound,
          start_cash: initialBalance,
          agent_type: agentType,
          start_date: startDate.toISOString().split('T')[0],
          end_date: endDate.toISOString().split('T')[0]
        };
        
        console.log('Creating game with data:', gameData);
        
        const response = await axios.post('http://localhost:8000/api/game/create', gameData);
        
        console.log('✓ Game created successfully:', response.data);
        console.log('Game ID:', response.data.id);
        setGameId(response.data.id);
        // 模态框会自动轮询状态并在完成后关闭
        
      } catch (error) {
        console.error('❌ Failed to create game:', error);
        
        if (error.response) {
          console.error('Server error:', error.response.data);
          alert(`Failed to create game: ${error.response.data.error || error.message}`);
        } else {
          alert(`Failed to create game: ${error.message}`);
        }
        
        setShowAIPrecomputeModal(false);
      }
    } else {
      alert('Please select exactly 3 stocks.');
    }
    
    console.log('=== confirmSelection END ===');
  };

  // AI预计算完成处理
  const handleAIPrecomputeComplete = (status) => {
    console.log('AI precompute completed:', status);
    
    if (status.status === 'completed') {
      // 成功完成
      setAiPrecomputeComplete(true);
      setGameCreated(true);
      setShowAIPrecomputeModal(false);
      setStopCounter(false); // 开始游戏计时
      
      console.log('✓ Ready to start game');
    } else if (status.status === 'failed') {
      // 失败
      console.error('AI precompute failed:', status.error);
      alert('Failed to initialize AI opponent. Please try again.');
      setShowAIPrecomputeModal(false);
    }
  };

  const saveTransaction = async (transaction) => {
    try {
      await axios.post('http://localhost:8000/api/transactions', transaction);
      console.log('Transaction saved:', transaction);
    } catch (error) {
      console.error('Error saving transaction:', error);
    }
  };

  const runAIStrategy = async () => {
    // 只在AI预计算完成后才执行
    if (!currentDate || !aiPrecomputeComplete) {
      console.log('Waiting for AI precompute to complete...');
      return;
    }

    const date = currentDate.toISOString().split('T')[0];
    const aiInfo = await fetchAiInfo();

    try {
      console.log('Loading AI strategy for date:', date);
      
      // 从数据库加载预计算的AI策略
      const aiResponse = await axios.get('http://localhost:8000/api/get_trade_log', {
        params: {
          game_id: gameId,
          model: modelList[0],
          date: date,
        }
      });

      if (aiResponse.data) {
        console.log("✓ AI Strategy loaded:", aiResponse.data);

        // 使用从数据库获取的策略
        const strategy = aiResponse.data.change || {};
        if (Object.keys(strategy).length === 0) {
          selectedStockList.forEach(stock => {
            strategy[stock] = 0;
          });
        }

        // 执行AI交易
        for (const [stock, amount] of Object.entries(strategy)) {
          const response = await axios.get('http://localhost:8000/api/stored_stock_data', {
            params: {
              symbol: stock,
              start_date: date,
              end_date: date
            }
          });
          const filteredData = response.data;

          if (filteredData.length === 0) {
            console.error(`Stock info for ${stock} on ${date} not found`);
            continue;
          }

          const stockInfo = filteredData[0];

          const aiTransaction = {
            game_id: gameId,
            user_id: 'ai',
            stock_symbol: stock,
            transaction_type: amount > 0 ? 'buy' : (amount === 0 ? 'hold' : 'sell'),
            amount: Math.abs(amount),
            price: stockInfo.open,
            date: currentDate.toISOString()
          };

          await saveTransaction(aiTransaction);

          // 更新AI持仓
          if (amount > 0) {
            aiInfo.cash -= stockInfo.open * amount;
            aiInfo.stocks[stock] = (aiInfo.stocks[stock] || 0) + amount;
          } else if (amount < 0) {
            aiInfo.cash += stockInfo.open * Math.abs(amount);
            aiInfo.stocks[stock] = (aiInfo.stocks[stock] || 0) - Math.abs(amount);
          }
        }

        // 计算AI投资组合价值
        const aiPortfolioValue = await selectedStockList.reduce(async (accPromise, stock) => {
          const acc = await accPromise;
          const response = await axios.get('http://localhost:8000/api/stored_stock_data', {
            params: {
              symbol: stock,
              start_date: date,
              end_date: date
            }
          });
          const filteredData = response.data;

          if (filteredData.length === 0) {
            console.error(`Stock info for ${stock} on ${date} not found`);
            return acc;
          }

          const stockInfo = filteredData[0];
          return acc + (aiInfo.stocks[stock] || 0) * stockInfo.close;
        }, Promise.resolve(0));

        aiInfo.portfolio_value = aiPortfolioValue;
        aiInfo.total_assets = aiInfo.cash + aiInfo.portfolio_value;

        // 更新AI信息
        try {
          await axios.post('http://localhost:8000/api/game_info', aiInfo);
        } catch (error) {
          console.error('Error updating AI info:', error);
        }

        setAiStrategy(aiResponse.data);
        setShowStrategyModal(true);

        // 更新前端AI状态
        setAiCash(aiInfo.cash);
        setAiPortfolioValue(aiInfo.portfolio_value);
        setAiTotalAssets(aiInfo.total_assets);
      } else {
        console.error('No AI strategy found');
      }
    } catch (error) {
      console.error('Error fetching AI strategy:', error);
      
      if (error.response && error.response.status === 404) {
        console.warn('No AI strategy found for this date');
      }
    }
  };

  const fetchStockInfo = async () => {
    if (!currentDate) return;
    
    const date = currentDate.toISOString().split('T')[0];
    let userInfo2 = await fetchUserInfo();

    if (!userInfo2) {
      userInfo2 = userInfo;
      console.log("fetchUserInfo failed, using cached userInfo");
      return;
    }

    const newStockInfo = {};

    // 获取所有股票的当前价格
    for (const stock of Object.keys(selectedTrades)) {
      try {
        const response = await axios.get('http://localhost:8000/api/stored_stock_data', {
          params: {
            symbol: stock,
            start_date: date,
            end_date: date
          }
        });

        if (response.data && response.data[0]) {
          newStockInfo[stock] = response.data[0];
        } else {
          console.error(`Stock info for ${stock} on ${date} not found`);
        }
      } catch (error) {
        console.error(`Error fetching stock data for ${stock}:`, error);
      }
    }

    setStockInfo(newStockInfo);

    if (!userInfo2.stocks) {
      userInfo2.stocks = selectedStockList.reduce((acc, stock) => ({ ...acc, [stock]: 0 }), {});
    }

    // 计算投资组合价值
    const portfolioValue = await selectedStockList.reduce(async (accPromise, stock) => {
      const acc = await accPromise;
      const response = await axios.get('http://localhost:8000/api/stored_stock_data', {
        params: {
          symbol: stock,
          start_date: date,
          end_date: date
        }
      });
      const filteredData = response.data;

      if (filteredData.length === 0) {
        console.error(`Stock info for ${stock} on ${date} not found`);
        return acc;
      }

      const stockInfo = filteredData[0];
      return acc + (userInfo2.stocks[stock] || 0) * stockInfo.close;
    }, Promise.resolve(0));

    userInfo2.portfolio_value = portfolioValue;
    userInfo2.total_assets = userInfo2.cash + userInfo2.portfolio_value;

    // 更新前端显示
    setCash(userInfo2.cash);
    setPortfolioValue(userInfo2.portfolio_value);
    setTotalAssets(userInfo2.total_assets);
    setUserInfo(userInfo2);
  };

  const handleSubmit = async () => {
    console.log('handleSubmit');
    const userInfo2 = await fetchUserInfo();
    
    // 执行用户交易
    for (const [stock, { type, amount }] of Object.entries(selectedTrades)) {
      const transaction = {
        game_id: gameId,
        user_id: userId,
        stock_symbol: stock,
        transaction_type: type,
        amount: parseFloat(amount),
        price: stockInfo[stock].open,
        date: currentDate.toISOString()
      };

      await saveTransaction(transaction);

      // 更新用户持仓
      if (type === 'buy') {
        userInfo2.cash -= stockInfo[stock].open * amount;
        userInfo2.stocks[stock] = (userInfo2.stocks[stock] || 0) + parseFloat(amount);
      } else if (type === 'sell') {
        userInfo2.cash += stockInfo[stock].open * amount;
        userInfo2.stocks[stock] = (userInfo2.stocks[stock] || 0) - Math.abs(amount);
      }
    }

    // 保存用户信息
    try {
      await axios.post('http://localhost:8000/api/game_info', userInfo2);
    } catch (error) {
      console.error('Error updating user info:', error);
    }

    // 加载并执行AI策略
    await runAIStrategy();

    setStopCounter(true);
  };

  const fetchUserInfo = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/game_info', {
        params: {
          game_id: gameId,
          user_id: userId
        }
      });
      return response.data[0];
    } catch (error) {
      console.error('Error fetching user info:', error);
      return null;
    }
  };

  const fetchAiInfo = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/game_info', {
        params: {
          game_id: gameId,
          user_id: 'ai'
        }
      });
      return response.data[0];
    } catch (error) {
      console.error('Error fetching AI info:', error);
      return null;
    }
  };

  const handleNextRound = async () => {
    const n = 1;
    try {
      const response = await axios.post('http://localhost:8000/api/next_trading_day', {
        current_date: currentDate.toISOString().split('T')[0],
        n: n
      });
      const nextDate = new Date(response.data.next_trading_day);

      if (currentRound === MaxRound) {
        setGameEnd(true);
        setStopCounter(true);
        setShowGameEndModal(true);
        return;
      }

      setCurrentRound(currentRound + 1);
      setCurrentDate(nextDate);
      setCounter(TMinus);
    } catch (error) {
      console.error('Error fetching next trading day:', error);
    }
  };

  const filteredCandlestickChartData = CandlestickChartData.filter(data => data.date < currentDate);

  const closeStrategyModal = () => {
    setShowStrategyModal(false);
    setSelectedTrades(selectedStockList.reduce((acc, stock) => ({ ...acc, [stock]: { type: 'hold', amount: '0' } }), {}));
    setRefreshHistory(prev => !prev);
    handleNextRound();
    setStopCounter(false);
  };

  if (!currentDate) {
    return <div>Loading...</div>;
  }

  return (
    <div className="background">
      <div className="app">
        <div className="wrapper d-flex flex-column min-vh-100" style={{ color: 'white' }}>
          <AppHeader />
          <div className="d-flex justify-content-between align-items-center w-100">
            <div className="d-flex justify-content-start" style={{ padding: '1em' }}>
              AI Opponent: {difficulty}
            </div>
            <div className="d-flex justify-content-center align-items-center flex-grow-1">
              <span className="mx-3">Current Round: {currentRound}/{MaxRound}</span>
              <span className="mx-3">Current Date: {currentDate.toISOString().split('T')[0]}</span>
              <span className="mx-3">Countdown: {counter}</span>
            </div>
            <CDropdown variant="dropdown">
              <CDropdownToggle caret={true}>
                <span style={{ color: 'white' }}>Game Credits: 50</span>
              </CDropdownToggle>

              <CDropdownMenu className='dropdown-menu'>
                <CDropdownItem className='dropdown-item' onClick={handleShowPointsStore}>
                  <span style={{ color: 'white' }}>Shop</span>
                </CDropdownItem>
              </CDropdownMenu>
            </CDropdown>

            <PointsStoreModal show={showPointsStore} handleClose={handleClosePointsStore} />
          </div>

          <div className="body flex-grow-1 px-3 d-flex flex-column align-items-center">
            <div className="d-flex justify-content-center w-100 mb-3" style={{ padding: '1em' }}>
              {selectedStockList.map((stock) => (
                <button key={stock} onClick={() => setSelectedStock(stock)}>{stock}</button>
              ))}
            </div>
            <div className="market-display d-flex" style={{ flexDirection: 'row', alignItems: 'end' }}>
              <div className="stock-info" style={{ backgroundColor: 'transparent', flex: '1', padding: '1em' }}>
                <div style={{ backgroundColor: 'white', color: 'black' }}>
                  <CandlestickChart data={filteredCandlestickChartData} stockName={selectedStock} style={{ zIndex: '1' }} />
                </div>
              </div>
              <div className="report" style={{ flex: "1", padding: '1em' }}>
                <FinancialReport 
                  selectedStock={selectedStock}
                  currentDate={currentDate}
                  stockData={stockData}
                  setStockData={setStockData}
                  chartWidth="95%"
                  chartHeight={250}
                  chartTop={80}
                  chartLeft={0}
                  chartRight={10}
                  titleColor="blue"
                  backgroundColor="rgba(255, 255, 255, 1)"
                  chartPaddingLeft={-20}
                  rowGap={20}
                  colGap={5}
                  chartContainerHeight={300}
                  rowsPerPage={5} 
                />
              </div>
            </div>
            <div className="bottom-section d-flex">
              <div className="left-section" style={{ flex: 1, marginRight: '20px' }}>
                <StockTradeComponent
                  selectedTrades={selectedTrades}
                  setSelectedTrades={setSelectedTrades}
                  initialBalance={initialBalance}
                  cash={cash}
                  userId={userId}
                  selectedStock={selectedStockList}
                  handleSubmit={handleSubmit}
                  stockData={stockInfo}
                  userInfo={userInfo}
                />
              </div>
              <div className="right-section" style={{ flex: 1 }}>
                <div className="financials mb-3">
                  <div className="d-flex justify-content-between align-items-center w-100 mb-3">
                    <div style={{ marginTop: '21px', marginRight: '4px' }}>Cash: ${cash.toFixed(2)}</div>
                    <div style={{ marginTop: '21px', marginRight: '4px' }}>Portfolio Value: ${portfolioValue.toFixed(2)}</div>
                    <div style={{ marginTop: '21px', marginRight: '4px' }}>Total Assets: ${totalAssets.toFixed(2)}</div>
                  </div>
                  <div className="d-flex justify-content-between align-items-center w-100 mb-3">
                    <div style={{ marginRight: '4px' }}>AI Cash: ${aiCash.toFixed(2)}</div>
                    <div style={{ marginRight: '4px' }}>AI Portfolio Value: ${aiPortfolioValue.toFixed(2)}</div>
                    <div style={{ marginRight: '4px' }}>AI Total Assets: ${aiTotalAssets.toFixed(2)}</div>
                  </div>
                </div>
                <Standings initialBalance={initialBalance} totalAssets={totalAssets} aiTotalAssets={aiTotalAssets} />
              </div>
            </div>

            <div className="history-section">
              <TradeHistory 
                userId={userId} 
                refreshHistory={refreshHistory} 
                selectedStock={selectedStockList} 
                gameId={gameId}
                agentType={difficulty}
              />
            </div>
          </div>
        </div>
      </div>

      {/* 股票选择模态框 */}
      <Modal 
        isOpen={isModalOpen} 
        onRequestClose={closeModal} 
        contentLabel="Select Stocks"
        shouldCloseOnEsc={false}
        shouldCloseOnOverlayClick={false}
        style={{
          content: {
            top: '50%',
            left: '50%',
            right: 'auto',
            bottom: 'auto',
            marginRight: '-50%',
            transform: 'translate(-50%, -50%)',
            width: '80%',
            height: 'auto',
            zIndex: '1000',
            border: '1px solid #ccc',
            borderRadius: '10px',
            padding: '20px',
            boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
          }
        }}
      >
        <h2 style={{ textAlign: 'center', color: '#333', marginBottom: '20px' }}>Please Select 3 Stocks</h2>
        <Grid container spacing={2}>
          {tickers.map((ticker) => (
            <Grid item xs={2} key={ticker}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={selectedTickers.includes(ticker)}
                    onChange={() => handleTickerSelection(ticker)}
                    name={ticker}
                    color="primary"
                  />
                }
                label={
                  <Typography sx={{ fontSize: '22px', color: 'black' }}>
                    {ticker}
                  </Typography>
                }
              />
            </Grid>
          ))}
        </Grid>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '20px' }}>
          <Button
            onClick={confirmSelection}
            disabled={selectedTickers.length !== 3}
            style={{
              padding: '10px 20px',
              backgroundColor: selectedTickers.length === 3 ? '#008CBA' : '#ccc',
              color: '#fff',
              border: 'none',
              borderRadius: '5px',
              cursor: selectedTickers.length === 3 ? 'pointer' : 'not-allowed',
              outline: 'none',
              fontSize: '20px'
            }}
          >
            Confirm {selectedTickers.length === 3 ? '' : `(${selectedTickers.length}/3)`}
          </Button>
        </div>
      </Modal>

      {/* AI预计算进度模态框 */}
      <AIPrecomputeModal
        gameId={gameId}
        agentName={difficulty}
        visible={showAIPrecomputeModal}
        onComplete={handleAIPrecomputeComplete}
      />

      {/* 策略显示模态框 */}
      <Modal
        isOpen={showStrategyModal}
        onRequestClose={closeStrategyModal}
        contentLabel="Strategy Modal"
        style={{
          overlay: {
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            backdropFilter: 'blur(5px)',
            zIndex: '1000'
          },
          content: {
            top: '50%',
            left: '50%',
            right: 'auto',
            bottom: 'auto',
            marginRight: '-50%',
            transform: 'translate(-50%, -50%)',
            width: '80%',
            maxWidth: '600px',
            height: 'auto',
            padding: '20px',
            borderRadius: '10px',
            border: 'none',
            boxShadow: '0 4px 8px rgba(0, 0, 0, 0.2)',
            backgroundColor: '#fff',
            color: '#333',
            textAlign: 'center'
          }
        }}
      >
        <h2 style={{ textAlign: 'center', marginBottom: '20px' }}>
          Strategy for {currentDate.toISOString().split('T')[0]}
        </h2>
        <div style={{ marginBottom: '20px' }}>
          <h3 style={{ marginBottom: '10px', color: '#444' }}>Player's Strategy:</h3>
          {Object.entries(selectedTrades).map(([stock, trade]) => (
            <div key={stock} style={{ marginBottom: '10px' }}>
              <p>{stock}: {trade.type} {trade.amount}</p>
            </div>
          ))}
        </div>
        <div style={{ marginBottom: '20px' }}>
          <h3 style={{ marginBottom: '10px', color: '#444' }}>AI's Strategy:</h3>
          {aiStrategy && aiStrategy.change ? (
            Object.entries(aiStrategy.change).map(([stock, change]) => (
              <div key={stock} style={{ marginBottom: '10px' }}>
                <p>{stock}: {change > 0 ? `buy ${change}` : (change === 0 ? `hold ${change}` : `sell ${Math.abs(change)}`)}</p>
              </div>
            ))
          ) : (
            <p>No AI strategy found</p>
          )}
          {aiStrategy && aiStrategy.rationale && (
            <div
              style={{
                marginTop: '15px',
                padding: '10px 12px',
                backgroundColor: '#f5f5f5',
                borderRadius: '8px',
                textAlign: 'left',
                maxHeight: '200px',
                overflowY: 'auto',
                whiteSpace: 'pre-wrap',
              }}
            >
              <strong>AI Explanation:</strong>
              <br />
              {aiStrategy.rationale}
            </div>
          )}
        </div>
        <Button
          onClick={closeStrategyModal}
          style={{
            display: 'block',
            margin: '0 auto',
            padding: '10px 20px',
            border: 'none',
            borderRadius: '5px',
            backgroundColor: '#007bff',
            color: '#fff',
            cursor: 'pointer'
          }}
          variant='outlined'
        >
          Close
        </Button>
      </Modal>

      {/* 游戏结束模态框 */}
      <GameEndModal
        isOpen={showGameEndModal}
        onRequestClose={() => setShowGameEndModal(false)}
        userAssets={totalAssets}
        aiAssets={aiTotalAssets}
        userProfit={totalAssets - initialBalance}
        aiProfit={aiTotalAssets - initialBalance}
      />
    </div>
  );
}

export default CompetitionLayout;