import React, { useState, useCallback, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Switch,
  FormControlLabel,
  CircularProgress,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  AutorenewOutlined as AutoRefreshIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Psychology as PredictIcon,
} from '@mui/icons-material';
import { 
  useSystemStatus, 
  useStocks, 
  usePredictions, 
  useAutoRefresh,
  usePredictionRegenerate
} from '../../hooks';
import { SummaryCard } from './SummaryCard';
import { SystemStatusIndicator } from './SystemStatusIndicator';
import { LoadingSpinner, ErrorMessage } from '../common';

const DEFAULT_REFRESH_INTERVAL = 5 * 60 * 1000; // 5 minutes

interface DashboardProps {
  onNavigateToStocks?: () => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ onNavigateToStocks }) => {
  const theme = useTheme();
  const isDesktop = useMediaQuery(theme.breakpoints.up('lg'));
  
  const [autoRefreshEnabled, setAutoRefreshEnabled] = useState(false);
  const [refreshInterval] = useState(DEFAULT_REFRESH_INTERVAL);
  const [lastManualRefresh, setLastManualRefresh] = useState<Date | null>(null);

  // API hooks
  const systemStatus = useSystemStatus();
  const stocks = useStocks(undefined, undefined, true); // Get ALL stocks with predictions included
  const predictions = usePredictions(); // Keep separate predictions for additional summary data

  // Regenerate predictions hook
  const { regenerate: regeneratePredictions, loading: regenerateLoading } = usePredictionRegenerate({
    onSuccess: (data) => {
      console.log(`Successfully regenerated ${data.successful_count} predictions`);
      // Refresh the data to show updated predictions
      handleManualRefresh();
    },
    onError: (error) => {
      console.error('Error regenerating predictions:', error);
    }
  });

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

  // Re-predict all function
  const handleRepredictAll = useCallback(async () => {
    try {
      await regeneratePredictions(undefined, false, 10); // Regenerate up to 10 predictions
    } catch (error) {
      console.error('Error during re-prediction:', error);
    }
  }, [regeneratePredictions]);

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

  const isLoading = systemStatus.loading || stocks.loading || predictions.loading || regenerateLoading;
  const hasError = systemStatus.error || stocks.error || predictions.error;

  // Calculate summary statistics from stocks data (which includes predictions)
  const totalStocks = systemStatus.data?.data?.total_symbols || 0;
  
  // Get prediction statistics from stocks data
  const stocksWithPredictions = stocks.data?.stocks || [];
  const activePredictions = stocksWithPredictions.filter(stock => stock.prediction).length;
  const failedPredictions = 0; // We'll get this from system status if available
  
  // Calculate prediction accuracy from stocks data
  const predictionsWithAccuracy = stocksWithPredictions.filter(stock => 
    stock.prediction && stock.prediction.model_accuracy
  );
  const predictionAccuracy = predictionsWithAccuracy.length > 0 
    ? (predictionsWithAccuracy.reduce((sum, stock) => sum + (stock.prediction?.model_accuracy || 0), 0) / predictionsWithAccuracy.length * 100).toFixed(1)
    : '0';

  // Count UP/DOWN predictions from stocks data
  const upPredictions = stocksWithPredictions.filter(stock => stock.prediction?.prediction === 'UP').length;
  const downPredictions = stocksWithPredictions.filter(stock => stock.prediction?.prediction === 'DOWN').length;

  return (
    <Box sx={{ 
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
    }}>
      {/* Header */}
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: { xs: 'flex-start', lg: 'center' }, 
        mb: { xs: 3, lg: 4 },
        flexDirection: { xs: 'column', lg: 'row' },
        gap: { xs: 2, lg: 0 },
      }}>
        <Box>
          <Typography 
            variant="h4" 
            component="h1" 
            sx={{ 
              fontWeight: 700,
              fontSize: { xs: '1.75rem', lg: '2rem', xl: '2.25rem' },
              color: theme.palette.text.primary,
              mb: 0.5,
            }}
          >
            Welcome to Dashboard
          </Typography>
          <Typography 
            variant="subtitle1" 
            color="text.secondary"
            sx={{ 
              fontSize: { xs: '0.9rem', lg: '1rem' },
            }}
          >
            Monitor your stock portfolio and market insights in real-time
          </Typography>
        </Box>
        
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: { xs: 1.5, lg: 2 }, 
          flexWrap: 'wrap',
          width: { xs: '100%', lg: 'auto' },
          justifyContent: { xs: 'flex-start', lg: 'flex-end' },
        }}>
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
                size={isDesktop ? "medium" : "small"}
              />
            }
            label={
              <Box sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: 0.5,
                fontSize: { xs: '0.875rem', lg: '0.95rem' },
              }}>
                <AutoRefreshIcon fontSize="small" />
                <Box sx={{ display: { xs: 'none', sm: 'block' } }}>
                  Auto-refresh
                </Box>
              </Box>
            }
            sx={{
              '& .MuiFormControlLabel-label': {
                fontSize: { xs: '0.875rem', lg: '0.95rem' },
              },
            }}
          />
          
          {/* Manual refresh button */}
          <Button
            variant="contained"
            startIcon={isLoading ? <CircularProgress size={16} color="inherit" /> : <RefreshIcon />}
            onClick={handleManualRefresh}
            disabled={isLoading}
            size={isDesktop ? "large" : "medium"}
            sx={{
              borderRadius: { xs: 2, lg: 2.5 },
              padding: { xs: '8px 16px', lg: '10px 20px' },
              fontSize: { xs: '0.875rem', lg: '0.95rem' },
              minWidth: { xs: 'auto', lg: '140px' },
              transition: 'all 0.2s ease',
              '&:hover': {
                transform: isDesktop ? 'translateY(-2px)' : 'none',
                boxShadow: isDesktop ? '0 6px 16px rgba(25, 118, 210, 0.3)' : 'inherit',
              },
            }}
          >
            {isLoading ? 'Refreshing...' : (isDesktop ? 'Refresh' : 'Refresh')}
          </Button>

          {/* Re-predict all button */}
          <Button
            variant="outlined"
            color="secondary"
            startIcon={regenerateLoading ? <CircularProgress size={16} color="inherit" /> : <PredictIcon />}
            onClick={handleRepredictAll}
            disabled={isLoading || regenerateLoading}
            size={isDesktop ? "large" : "medium"}
            sx={{
              borderRadius: { xs: 2, lg: 2.5 },
              padding: { xs: '8px 16px', lg: '10px 20px' },
              fontSize: { xs: '0.875rem', lg: '0.95rem' },
              minWidth: { xs: 'auto', lg: '140px' },
              transition: 'all 0.2s ease',
              '&:hover': {
                transform: isDesktop ? 'translateY(-2px)' : 'none',
                boxShadow: isDesktop ? '0 6px 16px rgba(156, 39, 176, 0.3)' : 'inherit',
              },
            }}
          >
            {regenerateLoading ? 'Re-predicting...' : (isDesktop ? 'Re-predict All' : 'Re-predict')}
          </Button>
        </Box>
      </Box>

      {/* Last update info */}
      {(lastManualRefresh || autoRefreshEnabled) && (
        <Paper sx={{ 
          p: { xs: 2, lg: 3 }, 
          mb: { xs: 3, lg: 4 }, 
          bgcolor: 'background.default',
          borderRadius: { xs: 2, lg: 3 },
          boxShadow: isDesktop ? '0 2px 12px rgba(0,0,0,0.08)' : 1,
        }}>
          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            flexDirection: { xs: 'column', sm: 'row' },
            gap: { xs: 1, sm: 0 },
          }}>
            <Typography variant="body2" color="text.secondary" sx={{
              fontSize: { xs: '0.8rem', lg: '0.875rem' },
            }}>
              {lastManualRefresh && `Last updated: ${lastManualRefresh.toLocaleString()}`}
            </Typography>
            {autoRefreshEnabled && (
              <Typography variant="body2" color="text.secondary" sx={{
                fontSize: { xs: '0.8rem', lg: '0.875rem' },
              }}>
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
        gridTemplateColumns: { 
          xs: '1fr',
          sm: 'repeat(2, 1fr)',
          lg: 'repeat(auto-fit, minmax(280px, 1fr))',
          xl: 'repeat(4, 1fr)',
        }, 
        gap: { xs: 2, lg: 3, xl: 4 }, 
        mb: { xs: 4, lg: 6 },
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

        {/* Market Sentiment */}
        <SummaryCard
          title="Market Sentiment"
          value={upPredictions > downPredictions ? 'Bullish' : downPredictions > upPredictions ? 'Bearish' : 'Neutral'}
          subtitle={`${upPredictions} UP, ${downPredictions} DOWN`}
          icon={upPredictions > downPredictions ? 'trending-up' : downPredictions > upPredictions ? 'trending-down' : 'assessment'}
          status={upPredictions > downPredictions ? 'success' : downPredictions > upPredictions ? 'error' : 'info'}
          loading={predictions.loading}
        />
      </Box>

      {/* Prediction Summary */}
      {stocksWithPredictions.length > 0 && (
        <Paper sx={{ 
          p: { xs: 2, lg: 3 }, 
          mb: { xs: 3, lg: 4 },
          borderRadius: { xs: 2, lg: 3 },
          boxShadow: isDesktop ? '0 2px 12px rgba(0,0,0,0.08)' : 1,
        }}>
          <Typography variant="h6" sx={{ 
            mb: { xs: 2, lg: 3 }, 
            fontWeight: 'bold',
            fontSize: { xs: '1.125rem', lg: '1.25rem' },
          }}>
            Prediction Summary ({activePredictions} stocks analyzed)
          </Typography>
          <Box sx={{ 
            display: 'grid', 
            gridTemplateColumns: { 
              xs: '1fr',
              sm: 'repeat(2, 1fr)',
              lg: 'repeat(3, 1fr)',
            }, 
            gap: { xs: 2, lg: 3 }
          }}>
            <SummaryCard
              title="Bullish Predictions"
              value={upPredictions}
              subtitle={`${((upPredictions / Math.max(upPredictions + downPredictions, 1)) * 100).toFixed(1)}% of predictions`}
              icon="trending-up"
              status="success"
              loading={stocks.loading}
            />
            <SummaryCard
              title="Bearish Predictions"
              value={downPredictions}
              subtitle={`${((downPredictions / Math.max(upPredictions + downPredictions, 1)) * 100).toFixed(1)}% of predictions`}
              icon="trending-down"
              status="error"
              loading={stocks.loading}
            />
            <SummaryCard
              title="Coverage"
              value={`${((activePredictions / Math.max(totalStocks, 1)) * 100).toFixed(1)}%`}
              subtitle={`${activePredictions} of ${totalStocks} stocks predicted`}
              icon="assessment"
              status="info"
              loading={stocks.loading}
            />
            <SummaryCard
              title="Bearish Predictions"
              value={downPredictions}
              subtitle={`${((downPredictions / Math.max(upPredictions + downPredictions, 1)) * 100).toFixed(1)}% of total`}
              icon="trending-down"
              status="error"
              loading={stocks.loading}
            />
            <SummaryCard
              title="Total Stocks"
              value={stocks.data?.total_count || 0}
              subtitle={`${stocks.data?.returned_count || 0} loaded`}
              icon="assessment"
              status="info"
              loading={stocks.loading}
            />
          </Box>
        </Paper>
      )}

      {/* Data Storage Info */}
      {systemStatus.data && (
        <Paper sx={{ 
          p: { xs: 2, lg: 3 },
          borderRadius: { xs: 2, lg: 3 },
          boxShadow: isDesktop ? '0 2px 12px rgba(0,0,0,0.08)' : 1,
        }}>
          <Typography variant="h6" sx={{ 
            mb: { xs: 2, lg: 3 }, 
            fontWeight: 'bold',
            fontSize: { xs: '1.125rem', lg: '1.25rem' },
          }}>
            Data Storage Information
          </Typography>
          <Box sx={{ 
            display: 'grid', 
            gridTemplateColumns: { 
              xs: 'repeat(2, 1fr)',
              sm: 'repeat(2, 1fr)',
              lg: 'repeat(4, 1fr)',
            }, 
            gap: { xs: 2, lg: 3 }
          }}>
            <Box>
              <Typography variant="body2" color="text.secondary" sx={{ fontSize: { xs: '0.8rem', lg: '0.875rem' } }}>
                Storage Size
              </Typography>
              <Typography variant="h6" sx={{ fontSize: { xs: '1.125rem', lg: '1.25rem' } }}>
                {systemStatus.data.data?.storage_size_mb?.toFixed(1) || 0} MB
              </Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary" sx={{ fontSize: { xs: '0.8rem', lg: '0.875rem' } }}>
                Latest Data
              </Typography>
              <Typography variant="h6" sx={{ fontSize: { xs: '1rem', lg: '1.125rem' } }}>
                {systemStatus.data.data?.latest_data_date 
                  ? new Date(systemStatus.data.data.latest_data_date).toLocaleDateString()
                  : 'N/A'
                }
              </Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary" sx={{ fontSize: { xs: '0.8rem', lg: '0.875rem' } }}>
                Oldest Data
              </Typography>
              <Typography variant="h6" sx={{ fontSize: { xs: '1rem', lg: '1.125rem' } }}>
                {systemStatus.data.data?.oldest_data_date 
                  ? new Date(systemStatus.data.data.oldest_data_date).toLocaleDateString()
                  : 'N/A'
                }
              </Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary" sx={{ fontSize: { xs: '0.8rem', lg: '0.875rem' } }}>
                Available Models
              </Typography>
              <Typography variant="h6" sx={{ fontSize: { xs: '1.125rem', lg: '1.25rem' } }}>
                {systemStatus.data.models?.available_count || 0}
              </Typography>
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