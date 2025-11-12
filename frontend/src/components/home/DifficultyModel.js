// DifficultyModal.js

import React from 'react';
import { CModal, CModalHeader, CModalBody, CButton, CModalTitle } from '@coreui/react';

const DifficultyModal = ({ visible, onClose, onDifficultyClick }) => {
  return (
    <CModal visible={visible} onClose={onClose} className="custom-modal">
      <CModalHeader onClose={onClose}>
        <CModalTitle>AI Agent Selection</CModalTitle>
      </CModalHeader>
      <CModalBody>
        <div className="difficulty-options">
          <CButton
            color="primary"
            className="mb-2"
            onClick={() => onDifficultyClick('PPOPlanning')}
          >
            PPO Planning Agent
          </CButton>
          <CButton
            color="info"
            className="mb-2"
            onClick={() => onDifficultyClick('Hierarchical')}
          >
            Hierarchical RL Agent
          </CButton>
          <CButton
            color="warning"
            className="mb-2"
            onClick={() => onDifficultyClick('RiskConstrained')}
          >
            Risk-Constrained Agent
          </CButton>
          <CButton
            color="danger"
            className="mb-2"
            onClick={() => onDifficultyClick('LLMReasoning')}
          >
            LLM Reasoning Agent
          </CButton>
        </div>
      </CModalBody>
    </CModal>
  );
};

export default DifficultyModal;
