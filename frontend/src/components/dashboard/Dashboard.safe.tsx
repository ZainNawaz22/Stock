import React from 'react';
import { Card, CardContent, Typography, Box, Button, CircularProgress } from '@mui/material';
import { Refresh as RefreshIcon } from '@mui/icons-material';
import { useSystemStatus } from '../../hooks';
import { LoadingSpinner, ErrorMessage } from '../';

export const Dashboard: React.FC = () => {
  const { data: systemStatus, loading, error, execute } = useSystemStatus();

  // COMPLETELY MANUAL - NO AUTO LOADING
  
  const handleLoadData = () => {
    console.log('Manual load triggered');
    execute();
  };

  if (loading && !systemStatus) {
    return <LoadingSpinner message="Loading system status..." />;
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" component="h1">
          Dashboard (Manual Mode)
        </Typography>
        <Button
          variant="contained"
          startIcon={loading ? <CircularProgress size={16} /> : <RefreshIcon />}
          onClick={handleLoadData}
          disabled={loading}
          size="large"
          color="primary"
        >
          {loading ? 'Loading...' : 'Load System Status'}
        </Button>
      </Box>

      {error && !systemStatus && (
        <ErrorMessage error={error} onRetry={handleLoadData} />
      )}
      
      {systemStatus && (
        <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 3 }}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                System Status
              </Typography>
              <Typography variant="h5" component="h2">
                {systemStatus.status || 'Unknown'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Health: {systemStatus.health?.status || 'Unknown'} ({systemStatus.health?.score || 0}%)
              </Typography>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Stocks
              </Typography>
              <Typography variant="h5" component="h2">
                {systemStatus.data?.total_symbols || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {systemStatus.data?.total_records || 0} records
              </Typography>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Available Models
              </Typography>
              <Typography variant="h5" component="h2">
                {systemStatus.models?.available_count || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Est. {systemStatus.models?.estimated_total || 0} total
              </Typography>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Last Update
              </Typography>
              <Typography variant="body2" component="p">
                {systemStatus.timestamp ? 
                  new Date(systemStatus.timestamp).toLocaleString() : 
                  'Never'
                }
              </Typography>
              <Typography variant="body2" color="text.secondary">
                API Started: {systemStatus.api_started ? 
                  new Date(systemStatus.api_started).toLocaleString() : 
                  'Unknown'
                }
              </Typography>
            </CardContent>
          </Card>
        </Box>
      )}

      {!systemStatus && !loading && !error && (
        <Card>
          <CardContent>
            <Typography variant="h6" color="text.secondary" align="center">
              Click "Load System Status" to fetch data from the server
            </Typography>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};
