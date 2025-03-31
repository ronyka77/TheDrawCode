import React from 'react';
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Divider,
  Grid
} from '@mui/material';

// Define the expected shape of results (matching backend.schemas.TrainingResults)
interface FeatureImportance {
  feature: string;
  importance: number;
}

interface TrainingResultsData {
  precision: number;
  recall: number;
  f1_score: number;
  optimal_threshold: number;
  feature_importance: FeatureImportance[];
}

interface ResultsDisplayProps {
  jobId: string | null;
  results: TrainingResultsData | null;
  isLoading: boolean;
  error: string | null;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ jobId, results, isLoading, error }) => {
  
  const formatMetric = (value: number | undefined): string => {
    return typeof value === 'number' ? value.toFixed(3) : 'N/A';
  };

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 200 }}>
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>Loading results...</Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        Error loading results: {error}
      </Alert>
    );
  }

  if (!results) {
    return (
      <Typography sx={{ mt: 2, fontStyle: 'italic' }}>
        {jobId ? 'Training completed, but no results data available.' : 'No training results to display. Run training first.'}
      </Typography>
    );
  }

  // Display the results
  return (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        Training Results (Job ID: {jobId})
      </Typography>
      <Paper elevation={3} sx={{ p: 2 }}>
        <Grid container spacing={2}>
          <Box sx={{ width: { xs: '100%', sm: '50%', md: '25%' }, p: 1 }}>
            <Typography variant="subtitle1">Precision</Typography>
            <Typography variant="h5">{formatMetric(results.precision)}</Typography>
          </Box>
          <Box sx={{ width: { xs: '100%', sm: '50%', md: '25%' }, p: 1 }}>
            <Typography variant="subtitle1">Recall</Typography>
            <Typography variant="h5">{formatMetric(results.recall)}</Typography>
          </Box>
          <Box sx={{ width: { xs: '100%', sm: '50%', md: '25%' }, p: 1 }}>
            <Typography variant="subtitle1">F1-Score</Typography>
            <Typography variant="h5">{formatMetric(results.f1_score)}</Typography>
          </Box>
          <Box sx={{ width: { xs: '100%', sm: '50%', md: '25%' }, p: 1 }}>
            <Typography variant="subtitle1">Optimal Threshold</Typography>
            <Typography variant="h5">{formatMetric(results.optimal_threshold)}</Typography>
          </Box>
        </Grid>
        
        <Divider sx={{ my: 3 }} />

        <Typography variant="h6" gutterBottom>
          Feature Importance
        </Typography>
        <TableContainer component={Paper} elevation={1} sx={{ maxHeight: 300 }}>
          <Table stickyHeader size="small">
            <TableHead>
              <TableRow>
                <TableCell>Feature</TableCell>
                <TableCell align="right">Importance</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {results.feature_importance.map((item) => (
                <TableRow key={item.feature} sx={{ '&:last-child td, &:last-child th': { border: 0 } }}>
                  <TableCell component="th" scope="row">
                    {item.feature}
                  </TableCell>
                  <TableCell align="right">{formatMetric(item.importance)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>
    </Box>
  );
};

export default ResultsDisplay; 