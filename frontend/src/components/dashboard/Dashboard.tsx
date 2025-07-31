import React, { useState, useCallback, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Switch,
  FormControlLabel,
  CircularProgress,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  AutorenewOutlined as AutoRefreshIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
} from '@mui/icons-material';
import { 
  useSystemStatus, 
  useStocks, 
  usePredictions, 
  useAutoRefresh 
} from '../../hooks';
import { SummaryCard } from './SummaryCard';
import { SystemStatusIndicator } from './SystemStatusIndicator';
import { LoadingSpinner, ErrorMessage } from '../common';

const DEFAULT_REFRESH_INTERVAL = 5 * 60 * 1000; // 5 minutes

interface DashboardProps {
  onNavigateToStocks?: () => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ onNavigateToStocks }) => {
  const [autoRefreshEnabled, setAutoRefreshEnabled] = useState(false);
  const [refreshInterval] = useState(DEFAULT_REFRESH_INTERVAL);
  const [lastManualRefresh, setLastManualRefresh] = useState<Date | null>(null);

  // API hooks
  const systemStatus = useSystemStatus();
  const stocks = useStocks(20, undefined, false); // Get first 20 stocks for summary, no predictions needed for dashboard
  const predictions = usePredictions(); // Keep separate predictions for dashboard summary

  // Manual refresh function
  const handleManualRefresh = useCallback(async () => {
    setLastManualRefresh(new Date());
    
    try {
      await Promise.all([
        systemStatus.execute(),
        stocks.execute(),
        predictions.execute(),
      ]);
    } catch (error) {
      console.error('Error during manual refresh:', error);
    }
  }, [systemStatus, stocks, predictions]);

  // Auto-refresh setup
  useAutoRefresh({
    interval: refreshInterval,
    enabled: autoRefreshEnabled,
    onRefresh: handleManualRefresh,
  });

  // Initial data load
  useEffect(() => {
    handleManualRefresh();
  }, []); // Only run once on mount

  const isLoading = systemStatus.loading || stocks.loading || predictions.loading;
  const hasError = systemStatus.error || stocks.error || predictions.error;

  // Calculate summary statistics
  const totalStocks = systemStatus.data?.data?.total_symbols || 0;
  const activePredictions = predictions.data?.successful_count || 0;
  const failedPredictions = predictions.data?.failed_count || 0;
  const predictionAccuracy = predictions.data?.predictions && predictions.data.predictions.length > 0 
    ? (predictions.data.predictions.reduce((sum, p) => sum + p.model_accuracy, 0) / predictions.data.predictions.length * 100).toFixed(1)
    : '0';

  const upPredictions = predictions.data?.predictions?.filter(p => p.prediction === 'UP').length || 0;
  const downPredictions = predictions.data?.predictions?.filter(p => p.prediction === 'DOWN').length || 0;

  return (
    <Box>
      {/* Header */}
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        mb: 3,
        flexWrap: 'wrap',
        gap: 2
      }}>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 'bold' }}>
          Dashboard
        </Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
          {/* System Status Indicator */}
          <SystemStatusIndicator 
            systemStatus={systemStatus.data}
            loading={systemStatus.loading}
            lastUpdated={systemStatus.data?.timestamp}
          />
          
          {/* Auto-refresh controls */}
          <FormControlLabel
            control={
              <Switch
                checked={autoRefreshEnabled}
                onChange={(e) => setAutoRefreshEnabled(e.target.checked)}
                icon={<PauseIcon />}
                checkedIcon={<PlayIcon />}
              />
            }
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <AutoRefreshIcon fontSize="small" />
                Auto-refresh
              </Box>
            }
          />
          
          {/* Manual refresh button */}
          <Button
            variant="contained"
            startIcon={isLoading ? <CircularProgress size={16} /> : <RefreshIcon />}
            onClick={handleManualRefresh}
            disabled={isLoading}
            size="medium"
          >
            {isLoading ? 'Refreshing...' : 'Refresh'}
          </Button>
        </Box>
      </Box>

      {/* Last update info */}
      {(lastManualRefresh || autoRefreshEnabled) && (
        <Paper sx={{ p: 2, mb: 3, bgcolor: 'background.default' }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              {lastManualRefresh && `Last updated: ${lastManualRefresh.toLocaleString()}`}
            </Typography>
            {autoRefreshEnabled && (
              <Typography variant="body2" color="text.secondary">
                Auto-refresh: Every {Math.floor(refreshInterval / 60000)} minutes
              </Typography>
            )}
          </Box>
        </Paper>
      )}

      {/* Error handling */}
      {hasError && !isLoading && (
        <Box sx={{ mb: 3 }}>
          {systemStatus.error && (
            <ErrorMessage error={systemStatus.error} onRetry={systemStatus.execute} />
          )}
          {stocks.error && (
            <ErrorMessage error={stocks.error} onRetry={stocks.execute} />
          )}
          {predictions.error && (
            <ErrorMessage error={predictions.error} onRetry={predictions.execute} />
          )}
        </Box>
      )}

      {/* Summary Cards Grid */}
      <Box sx={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', 
        gap: 3, 
        mb: 4 
      }}>
        {/* Total Stocks */}
        <SummaryCard
          title="Total Stocks"
          value={totalStocks}
          subtitle={`${systemStatus.data?.data?.total_records || 0} total records`}
          icon="storage"
          status="info"
          loading={systemStatus.loading}
          clickable={true}
          onClick={onNavigateToStocks}
        />

        {/* Active Predictions */}
        <SummaryCard
          title="Active Predictions"
          value={activePredictions}
          subtitle={failedPredictions > 0 ? `${failedPredictions} failed` : 'All successful'}
          icon={failedPredictions > 0 ? 'warning' : 'check'}
          status={failedPredictions > 0 ? 'warning' : 'success'}
          loading={predictions.loading}
        />

        {/* Prediction Accuracy */}
        <SummaryCard
          title="Avg. Model Accuracy"
          value={`${predictionAccuracy}%`}
          subtitle={`Based on ${predictions.data?.predictions?.length || 0} models`}
          icon="assessment"
          status={parseFloat(predictionAccuracy) >= 60 ? 'success' : 'warning'}
          loading={predictions.loading}
        />

        {/* System Health */}
        <SummaryCard
          title="System Health"
          value={`${systemStatus.data?.health?.score || 0}%`}
          subtitle={systemStatus.data?.health?.status || 'Unknown'}
          icon={
            (systemStatus.data?.health?.score || 0) >= 90 ? 'check' :
            (systemStatus.data?.health?.score || 0) >= 70 ? 'warning' : 'error'
          }
          status={
            (systemStatus.data?.health?.score || 0) >= 90 ? 'success' :
            (systemStatus.data?.health?.score || 0) >= 70 ? 'warning' : 'error'
          }
          loading={systemStatus.loading}
        />
      </Box>

      {/* Prediction Summary */}
      {predictions.data && predictions.data.predictions.length > 0 && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold' }}>
            Prediction Summary
          </Typography>
          <Box sx={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', 
            gap: 3 
          }}>
            <SummaryCard
              title="Bullish Predictions"
              value={upPredictions}
              subtitle={`${((upPredictions / (upPredictions + downPredictions)) * 100).toFixed(1)}% of total`}
              icon="trending-up"
              status="success"
              loading={predictions.loading}
            />
            <SummaryCard
              title="Bearish Predictions"
              value={downPredictions}
              subtitle={`${((downPredictions / (upPredictions + downPredictions)) * 100).toFixed(1)}% of total`}
              icon="trending-down"
              status="error"
              loading={predictions.loading}
            />
            <SummaryCard
              title="Cache Status"
              value={predictions.data.cache_used ? 'Cached' : 'Fresh'}
              subtitle={`Updated: ${new Date(predictions.data.last_updated).toLocaleString()}`}
              icon="speed"
              status={predictions.data.cache_used ? 'info' : 'success'}
              loading={predictions.loading}
            />
          </Box>
        </Paper>
      )}

      {/* Data Storage Info */}
      {systemStatus.data && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold' }}>
            Data Storage Information
          </Typography>
          <Box sx={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
            gap: 2 
          }}>
            <Box>
              <Typography variant="body2" color="text.secondary">Storage Size</Typography>
              <Typography variant="h6">{systemStatus.data.data?.storage_size_mb?.toFixed(1) || 0} MB</Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary">Latest Data</Typography>
              <Typography variant="h6">
                {systemStatus.data.data?.latest_data_date 
                  ? new Date(systemStatus.data.data.latest_data_date).toLocaleDateString()
                  : 'N/A'
                }
              </Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary">Oldest Data</Typography>
              <Typography variant="h6">
                {systemStatus.data.data?.oldest_data_date 
                  ? new Date(systemStatus.data.data.oldest_data_date).toLocaleDateString()
                  : 'N/A'
                }
              </Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary">Available Models</Typography>
              <Typography variant="h6">{systemStatus.data.models?.available_count || 0}</Typography>
            </Box>
          </Box>
        </Paper>
      )}

      {/* Loading state for initial load */}
      {isLoading && !systemStatus.data && !stocks.data && !predictions.data && (
        <LoadingSpinner message="Loading dashboard data..." />
      )}
    </Box>
  );
};