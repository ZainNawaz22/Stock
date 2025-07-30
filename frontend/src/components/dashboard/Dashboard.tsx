import React, { useEffect } from 'react';
import { Card, CardContent, Typography, Box, Button, CircularProgress } from '@mui/material';
import { Refresh as RefreshIcon } from '@mui/icons-material';
import { useSystemStatus } from '../../hooks';
import { LoadingSpinner, ErrorMessage } from '../';

export const Dashboard: React.FC = () => {
  const { data: systemStatus, loading, error, execute, refresh } = useSystemStatus();

  // Load data on component mount only
  useEffect(() => {
    execute();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleRefresh = () => {
    refresh();
  };

  if (loading && !systemStatus) {
    return <LoadingSpinner message="Loading system status..." />;
  }

  if (error && !systemStatus) {
    return <ErrorMessage error={error} onRetry={refresh} />;
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" component="h1">
          Dashboard
        </Typography>
        <Button
          variant="outlined"
          startIcon={loading ? <CircularProgress size={16} /> : <RefreshIcon />}
          onClick={handleRefresh}
          disabled={loading}
          size="small"
        >
          {loading ? 'Refreshing...' : 'Refresh'}
        </Button>
      </Box>
      
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 3 }}>
        <Card>
          <CardContent>
            <Typography color="textSecondary" gutterBottom>
              System Status
            </Typography>
            <Typography variant="h5" component="h2">
              {systemStatus?.status || 'Unknown'}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Health: {systemStatus?.health?.status || 'Unknown'} ({systemStatus?.health?.score || 0}%)
            </Typography>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent>
            <Typography color="textSecondary" gutterBottom>
              Total Stocks
            </Typography>
            <Typography variant="h5" component="h2">
              {systemStatus?.data?.total_symbols || 0}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {systemStatus?.data?.total_records || 0} records
            </Typography>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent>
            <Typography color="textSecondary" gutterBottom>
              Available Models
            </Typography>
            <Typography variant="h5" component="h2">
              {systemStatus?.models?.available_count || 0}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Est. {systemStatus?.models?.estimated_total || 0} total
            </Typography>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent>
            <Typography color="textSecondary" gutterBottom>
              Last Update
            </Typography>
            <Typography variant="body2" component="p">
              {systemStatus?.timestamp ? 
                new Date(systemStatus.timestamp).toLocaleString() : 
                'Never'
              }
            </Typography>
            <Typography variant="body2" color="text.secondary">
              API Started: {systemStatus?.api_started ? 
                new Date(systemStatus.api_started).toLocaleString() : 
                'Unknown'
              }
            </Typography>
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
};