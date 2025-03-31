import { useState, useEffect, useRef, useCallback } from 'react';
import { 
  AppBar, 
  Box, 
  Toolbar, 
  Typography, 
  Drawer, 
  Divider,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Tab,
  Tabs,
  IconButton
} from '@mui/material';
import { styled } from '@mui/material/styles';
import MenuIcon from '@mui/icons-material/Menu';
import SettingsIcon from '@mui/icons-material/Settings';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import TimelineIcon from '@mui/icons-material/Timeline';
import LogDisplay from './LogDisplay';
import ResultsDisplay from './ResultsDisplay';
import PredictionPanel from './PredictionPanel';
import ConfigurationPanel, { ConfigProps } from './ConfigurationPanel';
import ControlPanel from './ControlPanel';

// Constants
const DRAWER_WIDTH = 280;

// Custom styled components
const Main = styled('main', { shouldForwardProp: (prop) => prop !== 'open' })<{
  open?: boolean;
}>(({ theme, open }) => ({
  flexGrow: 1,
  padding: theme.spacing(3),
  transition: theme.transitions.create('margin', {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
  marginLeft: 0,
  ...(open && {
    transition: theme.transitions.create('margin', {
      easing: theme.transitions.easing.easeOut,
      duration: theme.transitions.duration.enteringScreen,
    }),
    marginLeft: `${DRAWER_WIDTH}px`,
  }),
}));

const TabPanel = (props: { children?: React.ReactNode; value: number; index: number }) => {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 0 }}>{children}</Box>}
    </div>
  );
};

// Define results type locally (or import if defined centrally)
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

export default function Layout() {
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [tabValue, setTabValue] = useState(0);

  // --- State for Config --- 
  const [config, setConfig] = useState<ConfigProps | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);

  // --- State for Training & Logs --- 
  const [isTraining, setIsTraining] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [statusMessage, setStatusMessage] = useState<string>("Ready");
  const websocketRef = useRef<WebSocket | null>(null);

  // --- State for Results (NEW) ---
  const [results, setResults] = useState<TrainingResultsData | null>(null);
  const [resultsLoading, setResultsLoading] = useState<boolean>(false);
  const [resultsError, setResultsError] = useState<string | null>(null);
  const [lastCompletedJobId, setLastCompletedJobId] = useState<string | null>(null); // Track job ID whose results were fetched

  // --- State for Prediction (Updated Types) ---
  const [predictionResult, setPredictionResult] = useState<DetailedPredictionResponse | null>(null);
  const [predictionLoading, setPredictionLoading] = useState<boolean>(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  // --- Event Handlers (handleTabChange, handleDrawerOpen/Close) ---
  const handleDrawerOpen = () => {
    setDrawerOpen(true);
  };

  const handleDrawerClose = () => {
    setDrawerOpen(false);
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // --- Fetch Initial Config --- 
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        setConfigError(null);
        // Use /api prefix configured in vite.config.ts proxy
        const response = await fetch('/api/config'); 
        if (!response.ok) {
          throw new Error(`Failed to fetch config: ${response.statusText}`);
        }
        const data: ConfigProps = await response.json();
        setConfig(data);
        console.log('Initial config fetched:', data);
      } catch (error: any) {
        console.error('Error fetching config:', error);
        setConfigError(error.message || 'Failed to load configuration.');
      }
    };

    fetchConfig();
  }, []);

  // --- Fetch Results Function (NEW) ---
  const fetchResults = useCallback(async (fetchJobId: string) => {
    if (!fetchJobId || lastCompletedJobId === fetchJobId) return; // Don't fetch if no ID or already fetched for this ID

    console.log(`Fetching results for job_id: ${fetchJobId}`);
    setResultsLoading(true);
    setResultsError(null);
    setResults(null); // Clear previous results

    try {
      const response = await fetch(`/api/results/${fetchJobId}`);
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to fetch results (${response.status}): ${errorText}`);
      }
      const data: TrainingResultsData = await response.json();
      setResults(data);
      setLastCompletedJobId(fetchJobId); // Mark this job ID's results as fetched
      console.log('Results fetched successfully:', data);
    } catch (error: any) {
      console.error('Error fetching results:', error);
      setResultsError(error.message || 'Failed to load results.');
      setLastCompletedJobId(fetchJobId); // Mark attempt even on error to avoid refetch loop
    } finally {
      setResultsLoading(false);
    }
  }, [lastCompletedJobId]); // Depend on lastCompletedJobId to prevent refetch loops

  // --- WebSocket Connection (Modified to trigger results fetch) ---
  useEffect(() => {
    let currentJobId: string | null = null; // Store jobId for cleanup reference

    if (jobId && isTraining) {
      currentJobId = jobId;
      // Construct WebSocket URL (adjust if your app is served from a different path)
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//${window.location.host}/ws/${jobId}`;
      
      console.log(`Connecting to WebSocket: ${wsUrl}`);
      setStatusMessage(`Connecting to job ${jobId}...`);
      setLogs([`Attempting to connect to WebSocket for job ${jobId}...`]);

      const ws = new WebSocket(wsUrl);
      websocketRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected');
        setStatusMessage('Connected. Waiting for logs...');
        setLogs(prev => [...prev, 'WebSocket connection established.']);
        // Optional: Send a ping or initial message if needed
        // ws.send(JSON.stringify({ type: 'client_ready' }));
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          console.log('WebSocket message received:', message);

          if (message.type === 'log') {
            setLogs(prev => [...prev, message.data]);
          } else if (message.type === 'status') {
            setStatusMessage(message.data);
            // Check for final statuses
            if (message.data === 'completed' || message.data === 'finished') {
              setIsTraining(false);
              websocketRef.current?.close(); // Close WS
              setLogs(prev => [...prev, `Training ${message.data}. WebSocket closed.`]);
              // Fetch results for the completed job (using the jobId captured in closure)
              if (currentJobId) {
                fetchResults(currentJobId);
              }
            } else if (message.data === 'failed') {
              setIsTraining(false);
              websocketRef.current?.close(); // Close WS
              setLogs(prev => [...prev, `Training failed. WebSocket closed.`]);
              // Optionally clear or mark results as failed for currentJobId
              setResults(null);
              setResultsError('Training job failed.');
              if(currentJobId) setLastCompletedJobId(currentJobId); // Prevent refetch on error too
            }
          } else {
             setLogs(prev => [...prev, `Unknown message type: ${JSON.stringify(message)}`]);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
          setLogs(prev => [...prev, `Error processing message: ${event.data}`]);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setStatusMessage('WebSocket error');
        setLogs(prev => [...prev, `WebSocket error occurred: ${error}`]);
        setIsTraining(false); // Stop training state on error
        setJobId(null);
      };

      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        // Don't reset status if it was already set to finished/failed/error
        if (isTraining) { // Check if it closed unexpectedly
           setStatusMessage('WebSocket closed unexpectedly');
           setLogs(prev => [...prev, `WebSocket connection closed unexpectedly (Code: ${event.code})`]);
           setIsTraining(false);
        }
        websocketRef.current = null;
      };

      // Cleanup function
      return () => {
        console.log('Cleaning up WebSocket connection...');
        ws.close();
        websocketRef.current = null;
      };
    }
  }, [jobId, isTraining, fetchResults]); // Add fetchResults to dependency array

  // --- Training Handler (Modified to clear results) ---
  const handleStartTraining = async () => {
    if (!config || isTraining) return;

    console.log('Requesting training start with config:', config);
    setIsTraining(true);
    setStatusMessage('Submitting training job...');
    setLogs(['Submitting training job...']); // Clear old logs
    setResults(null); 
    setResultsError(null); 
    setLastCompletedJobId(null); 
    setPredictionResult(null); 
    setPredictionError(null); 
    setJobId(null); 

    try {
      const response = await fetch('/api/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config), // Send current config
      });

      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(`Failed to start training: ${response.statusText} - ${errorData}`);
      }

      const result = await response.json(); 
      console.log('Training job submitted:', result);
      setStatusMessage(`Training job ${result.job_id} started.`);
      setJobId(result.job_id); // Set job ID to trigger WebSocket connection
      setLogs(prev => [...prev, `Training job ${result.job_id} submitted successfully.`]);
      
    } catch (error: any) {
      console.error('Error starting training:', error);
      setStatusMessage(`Error starting training: ${error.message}`);
      setLogs(prev => [...prev, `Error starting training: ${error.message}`]);
      setResults(null);
      setResultsError('Failed to start training.');
      setIsTraining(false);
      setJobId(null);
    }
  };

  // --- Prediction Handler (Modified for File Upload) ---
  const handlePredict = async (file: File) => {
    if (!file) return;

    console.log('Submitting file for prediction:', file.name);
    setPredictionLoading(true);
    setPredictionError(null);
    setPredictionResult(null);

    // Create FormData to send the file
    const formData = new FormData();
    formData.append('file', file); // Key must match FastAPI parameter name ('file')

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        // Do NOT set Content-Type header when using FormData;
        // The browser will set it correctly with the boundary.
        body: formData, 
      });

      if (!response.ok) {
        // Try to parse error response as JSON first, then text
        let errorDetail = `Prediction failed (${response.status})`;
        try {
          const errorJson = await response.json();
          errorDetail += `: ${errorJson.detail || JSON.stringify(errorJson)}`;
        } catch (jsonError) {
          const errorText = await response.text();
          errorDetail += `: ${errorText}`;
        }
        throw new Error(errorDetail);
      }

      // Expect DetailedPredictionResponse from backend
      const data: DetailedPredictionResponse = await response.json(); 
      setPredictionResult(data);
      console.log('Prediction successful:', data);
    } catch (error: any) {
      console.error('Prediction error:', error);
      setPredictionError(error.message || 'Failed to get prediction.');
    } finally {
      setPredictionLoading(false);
    }
  };

  // --- Update Config Handler --- 
  const handleConfigChange = async (newConfig: ConfigProps) => {
    // Optimistic UI update
    setConfig(newConfig);
    
    try {
      setConfigError(null);
      const response = await fetch('/api/config', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newConfig),
      });
      if (!response.ok) {
        // Revert optimistic update on failure
        const currentConfigResponse = await fetch('/api/config');
        const currentConfig = await currentConfigResponse.json();
        setConfig(currentConfig);
        throw new Error(`Failed to update config: ${response.statusText}`);
      }
      console.log('Config updated successfully');
    } catch (error: any) {
      console.error('Error updating config:', error);
      setConfigError(error.message || 'Failed to save configuration.');
    }
  };

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            onClick={handleDrawerOpen}
            edge="start"
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            Soccer Prediction UI
          </Typography>
          <IconButton color="inherit" aria-label="settings">
            <SettingsIcon />
          </IconButton>
        </Toolbar>
      </AppBar>

      <Drawer
        sx={{
          width: DRAWER_WIDTH,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: DRAWER_WIDTH,
            boxSizing: 'border-box',
            mt: ['56px', '64px'],
            height: 'calc(100% - 64px)',
          },
        }}
        variant="persistent"
        anchor="left"
        open={drawerOpen}
      >
        <Toolbar />
        <Box sx={{ overflow: 'auto', p: 2 }}>
          <Typography variant="h6" gutterBottom>Configuration</Typography>
          {config ? (
            <ConfigurationPanel 
              config={config} 
              onConfigChange={handleConfigChange} 
              isTraining={isTraining}
            />
          ) : (
            <Typography color="error">{configError || 'Loading configuration...'}</Typography>
          )}
          <Divider sx={{ my: 2 }} />
          <ControlPanel 
              isTraining={isTraining} 
              onStartTraining={handleStartTraining} 
              onStartPrediction={() => { 
                // Placeholder: Need to get input data from PredictionPanel
                // This is awkward. PredictionPanel needs to own its input state
                // and call handlePredict itself, or we need complex state lifting.
                // For now, trigger prediction with dummy data
                console.log('Triggering prediction from ControlPanel (needs input data!)'); 
                // handlePredict({ feature1: 0, feature2: 0, feature3: 'test'}); // Example call
              }}
          />
          <Typography variant="caption" display="block" sx={{ mt: 2 }}>{statusMessage}</Typography>
        </Box>
      </Drawer>

      <Main open={drawerOpen}>
        <Toolbar />
        <Box sx={{ width: '100%' }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={tabValue} onChange={handleTabChange} aria-label="Main content tabs">
              <Tab label="Logs" icon={<TimelineIcon />} iconPosition="start" id="tab-0" aria-controls="tabpanel-0" />
              <Tab label="Results" id="tab-1" aria-controls="tabpanel-1" />
              <Tab label="Prediction" id="tab-2" aria-controls="tabpanel-2" />
            </Tabs>
          </Box>
          <TabPanel value={tabValue} index={0}>
            <LogDisplay logs={logs} />
          </TabPanel>
          <TabPanel value={tabValue} index={1}>
            <ResultsDisplay 
              jobId={lastCompletedJobId}
              results={results} 
              isLoading={resultsLoading} 
              error={resultsError} 
            />
          </TabPanel>
          <TabPanel value={tabValue} index={2}>
            <PredictionPanel 
              onPredict={handlePredict} 
              predictionResult={predictionResult} 
              predictionLoading={predictionLoading}
              predictionError={predictionError}
            />
          </TabPanel>
        </Box>
      </Main>
    </Box>
  );
} 