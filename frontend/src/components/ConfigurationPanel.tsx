import { useState } from 'react';
import {
  Box,
  Typography,
  FormControlLabel,
  Checkbox,
  Slider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Card,
  CardContent,
  Grid,
  SelectChangeEvent,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';

// Define the config shape for TypeScript (Export for use in Layout)
export interface ConfigProps {
  extra_base_model_type: string;
  meta_learner_type: string;
  calibrate: boolean;
  dynamic_weighting: boolean;
  target_precision: number;
  required_recall: number;
}

// Define props for the component
interface ConfigurationPanelProps {
  config: ConfigProps;
  onConfigChange: (newConfig: ConfigProps) => void;
  isTraining: boolean;
}

export default function ConfigurationPanel({ 
  config, 
  onConfigChange, 
  isTraining 
}: ConfigurationPanelProps) {
  
  // Helper functions to update specific config values via the callback
  const handleModelChange = (event: SelectChangeEvent) => {
    onConfigChange({ ...config, extra_base_model_type: event.target.value });
  };

  const handleMetaLearnerChange = (event: SelectChangeEvent) => {
    onConfigChange({ ...config, meta_learner_type: event.target.value });
  };

  const handleCheckboxChange = (name: keyof ConfigProps) => (event: React.ChangeEvent<HTMLInputElement>) => {
    onConfigChange({ ...config, [name]: event.target.checked });
  };

  const handleSliderChange = (name: keyof ConfigProps) => (event: any, newValue: number | number[]) => {
    if (typeof newValue === 'number') {
      onConfigChange({ ...config, [name]: newValue / 100 });
    }
  };

  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Training Configuration
        </Typography>

        <Grid container spacing={2}>
          <Box width="100%">
            <FormControl fullWidth size="small" sx={{ mb: 2 }}>
              <InputLabel id="extra-model-label">Extra Base Model</InputLabel>
              <Select
                labelId="extra-model-label"
                value={config.extra_base_model_type}
                label="Extra Base Model"
                onChange={handleModelChange}
                disabled={isTraining}
              >
                <MenuItem value="random_forest">Random Forest</MenuItem>
                <MenuItem value="svm">SVM</MenuItem>
                <MenuItem value="mlp">MLP</MenuItem>
                <MenuItem value="catboost">CatBoost</MenuItem>
              </Select>
            </FormControl>
          </Box>

          <Box width="100%">
            <FormControl fullWidth size="small" sx={{ mb: 2 }}>
              <InputLabel id="meta-learner-label">Meta Learner</InputLabel>
              <Select
                labelId="meta-learner-label"
                value={config.meta_learner_type}
                label="Meta Learner"
                onChange={handleMetaLearnerChange}
                disabled={isTraining}
              >
                <MenuItem value="lgb">LightGBM</MenuItem>
                <MenuItem value="xgb">XGBoost</MenuItem>
                <MenuItem value="logistic">Logistic</MenuItem>
                <MenuItem value="mlp">MLP</MenuItem>
              </Select>
            </FormControl>
          </Box>

          <Box width="100%">
            <FormControlLabel
              control={
                <Checkbox
                  checked={config.calibrate}
                  onChange={handleCheckboxChange('calibrate')}
                  disabled={isTraining}
                />
              }
              label="Calibrate Probabilities"
            />
          </Box>

          <Box width="100%">
            <FormControlLabel
              control={
                <Checkbox
                  checked={config.dynamic_weighting}
                  onChange={handleCheckboxChange('dynamic_weighting')}
                  disabled={isTraining}
                />
              }
              label="Dynamic Weighting"
            />
          </Box>

          <Box width="100%">
            <Typography id="target-precision-slider" gutterBottom>
              Target Precision: {config.target_precision.toFixed(2)}
            </Typography>
            <Slider
              value={config.target_precision * 100}
              onChange={handleSliderChange('target_precision')}
              aria-labelledby="target-precision-slider"
              valueLabelDisplay="auto"
              step={5}
              marks
              min={0}
              max={100}
              disabled={isTraining}
            />
          </Box>

          <Box width="100%">
            <Typography id="required-recall-slider" gutterBottom>
              Required Recall: {config.required_recall.toFixed(2)}
            </Typography>
            <Slider
              value={config.required_recall * 100}
              onChange={handleSliderChange('required_recall')}
              aria-labelledby="required-recall-slider"
              valueLabelDisplay="auto"
              step={5}
              marks
              min={0}
              max={100}
              disabled={isTraining}
            />
          </Box>
        </Grid>
      </CardContent>
    </Card>
  );
} 