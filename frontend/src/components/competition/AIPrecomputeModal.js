import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  CModal, 
  CModalBody, 
  CModalHeader, 
  CModalTitle,
  CProgress,
  CProgressBar,
  CSpinner,
  CAlert
} from '@coreui/react';

/**
 * AI预计算进度组件 - 简化版
 * 只显示用户选择的单个agent的预计算进度
 */
const AIPrecomputeModal = ({ gameId, agentName, visible, onComplete }) => {
  const [status, setStatus] = useState({
    status: 'pending',
    progress: 0,
    error: ''
  });

  const [timeElapsed, setTimeElapsed] = useState(0);

  // 轮询游戏状态
  useEffect(() => {
    if (!visible || !gameId) return;

    const pollInterval = setInterval(async () => {
      try {
        const response = await axios.get(`http://localhost:8000/api/game/${gameId}/status`);
        const newStatus = response.data;
        setStatus(newStatus);

        // 如果完成了(无论成功还是失败),通知父组件
        if (newStatus.status === 'completed' || newStatus.status === 'failed') {
          setTimeout(() => {
            onComplete(newStatus);
          }, 1500); // 给用户1.5秒看到完成状态
        }
      } catch (error) {
        console.error('Error polling game status:', error);
      }
    }, 2000); // 每2秒轮询一次

    return () => clearInterval(pollInterval);
  }, [visible, gameId, onComplete]);

  // 计时器
  useEffect(() => {
    if (!visible) {
      setTimeElapsed(0);
      return;
    }

    const timer = setInterval(() => {
      setTimeElapsed(prev => prev + 1);
    }, 1000);

    return () => clearInterval(timer);
  }, [visible]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getProgressColor = () => {
    if (status.status === 'completed') return 'success';
    if (status.status === 'failed') return 'danger';
    return 'info';
  };

  const getStatusMessage = () => {
    switch (status.status) {
      case 'pending':
        return 'Initializing...';
      case 'running':
        return `Computing AI strategies for all trading rounds...`;
      case 'completed':
        return 'AI opponent is ready!';
      case 'failed':
        return 'Failed to initialize AI opponent';
      default:
        return 'Unknown status';
    }
  };

  return (
    <CModal
      visible={visible}
      backdrop="static"
      keyboard={false}
      size="lg"
      alignment="center"
    >
      <CModalHeader>
        <CModalTitle>Preparing AI Opponent</CModalTitle>
      </CModalHeader>
      <CModalBody>
        <div className="text-center mb-4">
          <h5 className="mb-3">{agentName || 'AI Agent'}</h5>
          <div className="d-flex justify-content-between align-items-center mb-2">
            <span className="text-muted">{getStatusMessage()}</span>
            <span className="badge bg-secondary">{formatTime(timeElapsed)}</span>
          </div>
        </div>

        <CProgress className="mb-3" height={30}>
          <CProgressBar
            color={getProgressColor()}
            value={status.progress}
            animated={status.status === 'running'}
          >
            {status.progress}%
          </CProgressBar>
        </CProgress>
        
        {status.status === 'completed' && (
          <CAlert color="success" className="mb-0">
            <div className="d-flex align-items-center">
              <i className="bi bi-check-circle-fill me-2" style={{ fontSize: '1.5rem' }}></i>
              <div>
                <strong>All set!</strong> Your AI opponent has analyzed all trading scenarios.
                The game will start shortly.
              </div>
            </div>
          </CAlert>
        )}
        
        {status.status === 'failed' && (
          <CAlert color="danger" className="mb-0">
            <div className="d-flex align-items-center">
              <i className="bi bi-x-circle-fill me-2" style={{ fontSize: '1.5rem' }}></i>
              <div>
                <strong>Initialization failed</strong>
                {status.error && (
                  <>
                    <br />
                    <small>{status.error}</small>
                  </>
                )}
              </div>
            </div>
          </CAlert>
        )}

        {status.status === 'running' && (
          <div className="text-center text-muted mt-3">
            <CSpinner className="me-2" size="sm" />
            <small>
              This may take 1-3 minutes. The AI is pre-computing trading decisions
              for all {' '} rounds to ensure smooth gameplay.
            </small>
          </div>
        )}

        {/* 进度阶段提示 */}
        {status.status === 'running' && (
          <div className="mt-4">
            <small className="text-muted d-block mb-2">Progress stages:</small>
            <div className="d-flex justify-content-between text-muted" style={{ fontSize: '0.85rem' }}>
              <div className={status.progress >= 10 ? 'text-primary fw-bold' : ''}>
                <i className={`bi ${status.progress >= 10 ? 'bi-check-circle-fill' : 'bi-circle'} me-1`}></i>
                Loading data
              </div>
              <div className={status.progress >= 40 ? 'text-primary fw-bold' : ''}>
                <i className={`bi ${status.progress >= 40 ? 'bi-check-circle-fill' : 'bi-circle'} me-1`}></i>
                Running strategy
              </div>
              <div className={status.progress >= 80 ? 'text-primary fw-bold' : ''}>
                <i className={`bi ${status.progress >= 80 ? 'bi-check-circle-fill' : 'bi-circle'} me-1`}></i>
                Saving results
              </div>
              <div className={status.progress >= 100 ? 'text-success fw-bold' : ''}>
                <i className={`bi ${status.progress >= 100 ? 'bi-check-circle-fill' : 'bi-circle'} me-1`}></i>
                Complete
              </div>
            </div>
          </div>
        )}
      </CModalBody>
    </CModal>
  );
};

export default AIPrecomputeModal;