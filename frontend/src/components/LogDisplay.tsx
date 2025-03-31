import { useEffect, useRef } from 'react';
import { Box, Typography, Paper, List, ListItem, ListItemText, Divider } from '@mui/material';

// Remove demo logs
// const DEMO_LOGS = [...];

// Define props
interface LogDisplayProps {
  logs: string[];
}

export default function LogDisplay({ logs }: LogDisplayProps) {
  // const [logs, setLogs] = useState<string[]>(DEMO_LOGS); // Removed internal state
  const logEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to the latest log
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // In a real implementation, this would connect to a WebSocket to receive logs in real-time
  // useEffect(() => {
  //   // WebSocket connection code would go here
  //   // For example: const ws = new WebSocket(`ws://localhost:8000/ws/${jobId}`);
  //   // ws.onmessage = (event) => {
  //   //   const data = JSON.parse(event.data);
  //   //   if (data.type === 'log') {
  //   //     setLogs(prevLogs => [...prevLogs, data.data]);
  //   //   }
  //   // };
  //   // return () => ws.close();
  // }, []);

  return (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        Training Logs
      </Typography>
      <Paper
        elevation={3}
        sx={{
          height: 300,
          overflow: 'auto',
          p: 2,
          backgroundColor: '#f5f5f5',
        }}
      >
        <List dense>
          {logs.map((log, index) => (
            <>
              <ListItem key={index} sx={{ py: 0.5 }}>
                <ListItemText
                  primary={log}
                  primaryTypographyProps={{
                    sx: {
                      fontFamily: 'monospace',
                      fontSize: '0.8rem',
                    },
                  }}
                />
              </ListItem>
              {index < logs.length - 1 && <Divider component="li" />}
            </>
          ))}
          <div ref={logEndRef} />
        </List>
      </Paper>
    </Box>
  );
} 