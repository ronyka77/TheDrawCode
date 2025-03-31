import React from 'react';
import { Box, Button, CircularProgress } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import OnlinePredictionIcon from '@mui/icons-material/OnlinePrediction';

interface ControlPanelProps {
  isTraining: boolean;
  onStartTraining: () => void;
  onStartPrediction: () => void;
}

const ControlPanel: React.FC<ControlPanelProps> = ({ 
  isTraining, 
  onStartTraining, 
  onStartPrediction 
}) => {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Button
        variant="contained"
        color="primary"
        startIcon={isTraining ? <CircularProgress size={20} color="inherit" /> : <PlayArrowIcon />}
        onClick={onStartTraining}
        disabled={isTraining}
        fullWidth
      >
        {isTraining ? 'Training in Progress...' : 'Start Training'}
      </Button>
      {/* Placeholder for Prediction Button */}
      <Button
        variant="contained"
        color="secondary"
        startIcon={<OnlinePredictionIcon />}
        onClick={onStartPrediction}
        disabled={isTraining}
        fullWidth
      >
        Run Prediction (Placeholder)
      </Button>
    </Box>
  );
};

export default ControlPanel; 