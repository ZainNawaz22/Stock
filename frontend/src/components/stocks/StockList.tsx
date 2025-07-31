import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  TextField,
  InputAdornment,
  Pagination,
  Typography,
  Chip,
  IconButton,
  Tooltip,
  CircularProgress,
} from '@mui/material';
import {
  Search as SearchIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Remove as NeutralIcon,
  Refresh as RefreshIcon,
  ArrowBack as ArrowBackIcon,
} from '@mui/icons-material';
import { useStocks } from '../../hooks';
import { LoadingSpinner, ErrorMessage } from '../common';
import type { StockInfo, PredictionResult } from '../../types/api';

// Combined stock data with prediction
interface StockWithPrediction extends StockInfo {
  prediction?: PredictionResult;
}

// Sort configuration
type SortField = 'symbol' | 'name' | 'current_price' | 'change_percent' | 'volume' | 'prediction';
type SortDirection = 'asc' | 'desc';

interface SortConfig {
  field: SortField;
  direction: SortDirection;
}

const ITEMS_PER_PAGE = 25;

interface StockListProps {
  onNavigateBack?: () => void;
}

export const StockList: React.FC<StockListProps> = ({ onNavigateBack }) => {
  // State management
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    field: 'symbol',
    direction: 'asc',
  });

  // API hooks - fetch all stocks with predictions included
  const stocks = useStocks(undefined, undefined, true); // Include predictions

  // Use stocks data directly since predictions are now included
  const stocksWithPredictions = useMemo<StockWithPrediction[]>(() => {
    if (!stocks.data?.stocks) return [];
    
    return stocks.data.stocks.map(stock => ({
      ...stock,
      prediction: stock.prediction,
    }));
  }, [stocks.data?.stocks]);

  // Filter stocks based on search term
  const filteredStocks = useMemo(() => {
    if (!searchTerm.trim()) return stocksWithPredictions;
    
    const searchLower = searchTerm.toLowerCase();
    return stocksWithPredictions.filter(stock =>
      stock.symbol.toLowerCase().includes(searchLower) ||
      stock.name.toLowerCase().includes(searchLower)
    );
  }, [stocksWithPredictions, searchTerm]);

  // Sort stocks
  const sortedStocks = useMemo(() => {
    const sorted = [...filteredStocks].sort((a, b) => {
      let aValue: any;
      let bValue: any;

      switch (sortConfig.field) {
        case 'symbol':
          aValue = a.symbol;
          bValue = b.symbol;
          break;
        case 'name':
          aValue = a.name;
          bValue = b.name;
          break;
        case 'current_price':
          aValue = a.current_price;
          bValue = b.current_price;
          break;
        case 'change_percent':
          aValue = a.change_percent;
          bValue = b.change_percent;
          break;
        case 'volume':
          aValue = a.volume;
          bValue = b.volume;
          break;
        case 'prediction':
          // Sort by prediction: UP first, then DOWN, then no prediction
          aValue = a.prediction ? (a.prediction.prediction === 'UP' ? 2 : 1) : 0;
          bValue = b.prediction ? (b.prediction.prediction === 'UP' ? 2 : 1) : 0;
          break;
        default:
          aValue = a.symbol;
          bValue = b.symbol;
      }

      // Handle string comparison
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        const comparison = aValue.localeCompare(bValue);
        return sortConfig.direction === 'asc' ? comparison : -comparison;
      }

      // Handle numeric comparison
      if (aValue < bValue) {
        return sortConfig.direction === 'asc' ? -1 : 1;
      }
      if (aValue > bValue) {
        return sortConfig.direction === 'asc' ? 1 : -1;
      }
      return 0;
    });

    return sorted;
  }, [filteredStocks, sortConfig]);

  // Paginate stocks
  const paginatedStocks = useMemo(() => {
    const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
    const endIndex = startIndex + ITEMS_PER_PAGE;
    return sortedStocks.slice(startIndex, endIndex);
  }, [sortedStocks, currentPage]);

  // Calculate pagination info
  const totalPages = Math.ceil(sortedStocks.length / ITEMS_PER_PAGE);

  // Handle sorting
  const handleSort = useCallback((field: SortField) => {
    setSortConfig(prev => ({
      field,
      direction: prev.field === field && prev.direction === 'asc' ? 'desc' : 'asc',
    }));
  }, []);

  // Handle search
  const handleSearchChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(event.target.value);
    setCurrentPage(1); // Reset to first page when searching
  }, []);

  // Handle page change
  const handlePageChange = useCallback((_: React.ChangeEvent<unknown>, page: number) => {
    setCurrentPage(page);
  }, []);

  // Refresh data
  const handleRefresh = useCallback(async () => {
    await stocks.execute();
  }, [stocks]);

  // Load data on mount
  useEffect(() => {
    handleRefresh();
  }, []);

  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-PK', {
      style: 'currency',
      currency: 'PKR',
      minimumFractionDigits: 2,
    }).format(value);
  };

  // Format percentage
  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  // Format volume
  const formatVolume = (value: number) => {
    if (value >= 1000000) {
      return `${(value / 1000000).toFixed(1)}M`;
    } else if (value >= 1000) {
      return `${(value / 1000).toFixed(1)}K`;
    }
    return value.toString();
  };

  // Render prediction chip
  const renderPredictionChip = (prediction?: PredictionResult) => {
    if (!prediction) {
      return (
        <Chip
          icon={<NeutralIcon />}
          label="No Prediction"
          size="small"
          variant="outlined"
          color="default"
        />
      );
    }

    return (
      <Tooltip title={`Confidence: ${(prediction.confidence * 100).toFixed(1)}%`}>
        <Chip
          icon={prediction.prediction === 'UP' ? <TrendingUpIcon /> : <TrendingDownIcon />}
          label={prediction.prediction}
          size="small"
          color={prediction.prediction === 'UP' ? 'success' : 'error'}
          variant="filled"
        />
      </Tooltip>
    );
  };

  const isLoading = stocks.loading;
  const hasError = stocks.error;

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
          Stock List
        </Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {onNavigateBack && (
            <Tooltip title="Back to Dashboard">
              <IconButton
                onClick={onNavigateBack}
                color="primary"
              >
                <ArrowBackIcon />
              </IconButton>
            </Tooltip>
          )}
          <Tooltip title="Refresh data">
            <IconButton
              onClick={handleRefresh}
              disabled={isLoading}
              color="primary"
            >
              {isLoading ? <CircularProgress size={24} /> : <RefreshIcon />}
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Search and Stats */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 2,
          mb: 2
        }}>
          <TextField
            placeholder="Search by symbol or name..."
            value={searchTerm}
            onChange={handleSearchChange}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
            }}
            sx={{ minWidth: 300 }}
          />
          
          <Box sx={{ display: 'flex', gap: 3, alignItems: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              Showing {paginatedStocks.length} of {sortedStocks.length} stocks
            </Typography>
            {searchTerm && (
              <Typography variant="body2" color="text.secondary">
                Filtered from {stocksWithPredictions.length} total
              </Typography>
            )}
          </Box>
        </Box>

        {/* Quick stats */}
        <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
          <Typography variant="body2" color="text.secondary">
            Total Stocks: {stocksWithPredictions.length}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            With Predictions: {stocksWithPredictions.filter(s => s.prediction).length}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Bullish: {stocksWithPredictions.filter(s => s.prediction?.prediction === 'UP').length}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Bearish: {stocksWithPredictions.filter(s => s.prediction?.prediction === 'DOWN').length}
          </Typography>
        </Box>
      </Paper>

      {/* Error handling */}
      {hasError && (
        <Box sx={{ mb: 3 }}>
          <ErrorMessage error={stocks.error} onRetry={stocks.execute} />
        </Box>
      )}

      {/* Loading state */}
      {isLoading && !stocks.data && (
        <LoadingSpinner message="Loading stocks..." />
      )}

      {/* Stock table */}
      {stocks.data && (
        <Paper>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>
                    <TableSortLabel
                      active={sortConfig.field === 'symbol'}
                      direction={sortConfig.field === 'symbol' ? sortConfig.direction : 'asc'}
                      onClick={() => handleSort('symbol')}
                    >
                      Symbol
                    </TableSortLabel>
                  </TableCell>
                  <TableCell>
                    <TableSortLabel
                      active={sortConfig.field === 'name'}
                      direction={sortConfig.field === 'name' ? sortConfig.direction : 'asc'}
                      onClick={() => handleSort('name')}
                    >
                      Name
                    </TableSortLabel>
                  </TableCell>
                  <TableCell align="right">
                    <TableSortLabel
                      active={sortConfig.field === 'current_price'}
                      direction={sortConfig.field === 'current_price' ? sortConfig.direction : 'asc'}
                      onClick={() => handleSort('current_price')}
                    >
                      Price
                    </TableSortLabel>
                  </TableCell>
                  <TableCell align="right">
                    <TableSortLabel
                      active={sortConfig.field === 'change_percent'}
                      direction={sortConfig.field === 'change_percent' ? sortConfig.direction : 'asc'}
                      onClick={() => handleSort('change_percent')}
                    >
                      Change %
                    </TableSortLabel>
                  </TableCell>
                  <TableCell align="right">
                    <TableSortLabel
                      active={sortConfig.field === 'volume'}
                      direction={sortConfig.field === 'volume' ? sortConfig.direction : 'asc'}
                      onClick={() => handleSort('volume')}
                    >
                      Volume
                    </TableSortLabel>
                  </TableCell>
                  <TableCell align="center">
                    <TableSortLabel
                      active={sortConfig.field === 'prediction'}
                      direction={sortConfig.field === 'prediction' ? sortConfig.direction : 'asc'}
                      onClick={() => handleSort('prediction')}
                    >
                      Prediction
                    </TableSortLabel>
                  </TableCell>
                  <TableCell align="right">Last Updated</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {paginatedStocks.map((stock) => (
                  <TableRow
                    key={stock.symbol}
                    hover
                    sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                  >
                    <TableCell component="th" scope="row">
                      <Typography variant="body2" fontWeight="bold">
                        {stock.symbol}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" noWrap>
                        {stock.name}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Typography variant="body2" fontWeight="medium">
                        {formatCurrency(stock.current_price)}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Typography
                        variant="body2"
                        color={stock.change_percent >= 0 ? 'success.main' : 'error.main'}
                        fontWeight="medium"
                      >
                        {formatPercentage(stock.change_percent)}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Typography variant="body2">
                        {formatVolume(stock.volume)}
                      </Typography>
                    </TableCell>
                    <TableCell align="center">
                      {renderPredictionChip(stock.prediction)}
                    </TableCell>
                    <TableCell align="right">
                      <Typography variant="body2" color="text.secondary">
                        {new Date(stock.last_updated).toLocaleString()}
                      </Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          {/* Pagination */}
          {totalPages > 1 && (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
              <Pagination
                count={totalPages}
                page={currentPage}
                onChange={handlePageChange}
                color="primary"
                showFirstButton
                showLastButton
              />
            </Box>
          )}

          {/* No results message */}
          {sortedStocks.length === 0 && !isLoading && (
            <Box sx={{ p: 4, textAlign: 'center' }}>
              <Typography variant="h6" color="text.secondary">
                {searchTerm ? 'No stocks found matching your search' : 'No stocks available'}
              </Typography>
              {searchTerm && (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Try adjusting your search terms
                </Typography>
              )}
            </Box>
          )}
        </Paper>
      )}
    </Box>
  );
};