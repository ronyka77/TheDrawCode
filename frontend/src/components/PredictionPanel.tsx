import React, { useState, useRef } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  CircularProgress,
  Alert,
  Input,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Divider
} from '@mui/material';
import OnlinePredictionIcon from '@mui/icons-material/OnlinePrediction';
import UploadFileIcon from '@mui/icons-material/UploadFile';

// --- NEW: Define detailed prediction response types ---
interface PredictionResultItem {
  index: number;
  prediction: number; // 0 or 1
  probability: number;
}

// Update Prediction response interface to match backend DetailedPredictionResponse
interface DetailedPredictionResponse {
  message: string;
  file_name: string;
  data_shape: [number, number] | null;
  num_predictions_processed: number;
  num_positive_predictions: number;
  prediction_rate: number;
  predictions_sample: PredictionResultItem[]; // Array of prediction items
}

interface PredictionPanelProps {
  // Keep onPredict accepting File
  onPredict: (file: File) => Promise<void>; 
  // Update prop type for predictionResult to use the detailed response
  predictionResult: DetailedPredictionResponse | null; 
  predictionLoading: boolean;
  predictionError: string | null;
}

const PredictionPanel: React.FC<PredictionPanelProps> = ({
  onPredict,
  predictionResult,
  predictionLoading,
  predictionError
}) => {
  // State to hold the selected file
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  // Ref for the hidden file input
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handler for file selection
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      const file = event.target.files[0];
      // Optional: Add file type validation here if needed
      if (file.type !== 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet') {
          alert('Invalid file type. Please select an .xlsx file.');
          setSelectedFile(null);
          // Clear the input value so the same file can be selected again if needed
          if(event.target) event.target.value = ''
          return;
      }
      setSelectedFile(file);
      console.log('File selected:', file.name);
    } else {
      setSelectedFile(null);
    }
  };

  // Handler to trigger the hidden file input
  const handleUploadButtonClick = () => {
    fileInputRef.current?.click();
  };

  // Modify handlePredictClick to pass the file
  const handlePredictClick = () => {
    if (!selectedFile) {
      alert('Please select an XLSX file first.');
      return;
    }
    onPredict(selectedFile); // Call the handler from Layout with the file
  };

  return (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        Make Prediction from File
      </Typography>
      <Paper elevation={3} sx={{ p: 3 }}>
        <Typography variant="subtitle1" gutterBottom>
          Upload XLSX File
        </Typography>

        {/* Hidden file input */}
        <Input 
           type="file"
           inputRef={fileInputRef}
           onChange={handleFileChange}
           sx={{ display: 'none' }} // Hide the default input
           inputProps={{ accept: '.xlsx, application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }} // Accept only .xlsx
        />

        {/* Button to trigger file input */}
        <Button
            variant="outlined"
            startIcon={<UploadFileIcon />}
            onClick={handleUploadButtonClick}
            disabled={predictionLoading}
            sx={{ mr: 2, mb: 2 }}
        >
            Choose File
        </Button>

        {/* Display selected file name */}
        {selectedFile && (
            <Typography variant="body2" component="span" sx={{ mr: 2, mb: 2, verticalAlign: 'middle' }}>
                Selected: {selectedFile.name}
            </Typography>
        )}

        <Button
          variant="contained"
          color="secondary"
          startIcon={predictionLoading ? <CircularProgress size={20} color="inherit" /> : <OnlinePredictionIcon />}
          onClick={handlePredictClick}
          disabled={predictionLoading || !selectedFile} // Disable if loading or no file selected
          sx={{ mb: 2 }}
        >
          {predictionLoading ? 'Predicting...' : 'Get Prediction'}
        </Button>

        {predictionError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {predictionError}
          </Alert>
        )}

        {predictionResult && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="subtitle1" gutterBottom>Prediction Summary:</Typography>
            <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
              <Typography variant="body1" component="p" sx={{ mb: 1 }}>
                {predictionResult.message}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                File: {predictionResult.file_name} 
                {predictionResult.data_shape && `| Shape: (${predictionResult.data_shape[0]}, ${predictionResult.data_shape[1]})`}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Processed: {predictionResult.num_predictions_processed} 
                | Predicted Draws (1): {predictionResult.num_positive_predictions}
                | Prediction Rate: {(predictionResult.prediction_rate * 100).toFixed(2)}%
              </Typography>
            </Paper>

            {predictionResult.predictions_sample && predictionResult.predictions_sample.length > 0 && (
              <Box>
                <Divider sx={{ my: 2 }} />
                <Typography variant="subtitle1" gutterBottom>Prediction Sample (First {predictionResult.predictions_sample.length}):</Typography>
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small" aria-label="prediction sample table">
                    <TableHead>
                      <TableRow sx={{ '& th': { fontWeight: 'bold' } }}>
                        <TableCell>Index</TableCell>
                        <TableCell align="right">Prediction</TableCell>
                        <TableCell align="right">Probability</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {predictionResult.predictions_sample.map((row) => (
                        <TableRow
                          key={row.index}
                          sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                        >
                          <TableCell component="th" scope="row">
                            {row.index}
                          </TableCell>
                          <TableCell align="right">{row.prediction}</TableCell>
                          <TableCell align="right">{row.probability.toFixed(4)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>
            )}
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default PredictionPanel; 