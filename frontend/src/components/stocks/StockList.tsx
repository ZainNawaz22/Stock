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
  useTheme,
  useMediaQuery,
  Button,
} from '@mui/material';
import {
  Search as SearchIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Remove as NeutralIcon,
  Refresh as RefreshIcon,
  ArrowBack as ArrowBackIcon,
  ViewColumn as ViewColumnIcon,
  FilterList as FilterListIcon,
  Psychology as PredictIcon,
} from '@mui/icons-material';
import { useStocks, usePredictionRegenerate } from '../../hooks';
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

const ITEMS_PER_PAGE_OPTIONS = [25, 50, 100];
const DEFAULT_ITEMS_PER_PAGE = 25;

interface StockListProps {
  onNavigateBack?: () => void;
}

export const StockList: React.FC<StockListProps> = ({ onNavigateBack }) => {
  const theme = useTheme();
  const isDesktop = useMediaQuery(theme.breakpoints.up('lg'));
  const isLargeDesktop = useMediaQuery(theme.breakpoints.up('xl'));
  
  // State management
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(DEFAULT_ITEMS_PER_PAGE);
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    field: 'symbol',
    direction: 'asc',
  });

  // API hooks - fetch all stocks with predictions included
  const stocks = useStocks(undefined, undefined, true); // Include predictions

  // Regenerate predictions hook
  const { regenerate: regeneratePredictions, loading: regenerateLoading } = usePredictionRegenerate({
    onSuccess: (data) => {
      console.log(`Successfully regenerated ${data.successful_count} predictions`);
      // Refresh the stocks data to show updated predictions
      stocks.execute();
    },
    onError: (error) => {
      console.error('Error regenerating predictions:', error);
    }
  });

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
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    return sortedStocks.slice(startIndex, endIndex);
  }, [sortedStocks, currentPage, itemsPerPage]);

  // Calculate pagination info
  const totalPages = Math.ceil(sortedStocks.length / itemsPerPage);

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

  // Re-predict all visible stocks
  const handleRepredictAll = useCallback(async () => {
    try {
      // Get current visible stocks
      const currentStocks = paginatedStocks.map(stock => stock.symbol);
      await regeneratePredictions(currentStocks, false); // Regenerate for current page
    } catch (error) {
      console.error('Error during re-prediction:', error);
    }
  }, [regeneratePredictions, paginatedStocks]);

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

  const isLoading = stocks.loading || regenerateLoading;
  const hasError = stocks.error;

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
            Stock List
          </Typography>
          <Typography 
            variant="subtitle1" 
            color="text.secondary"
            sx={{ 
              fontSize: { xs: '0.9rem', lg: '1rem' },
            }}
          >
            Browse and analyze all available stocks with AI predictions
          </Typography>
        </Box>
        
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: { xs: 1.5, lg: 2 },
          width: { xs: '100%', lg: 'auto' },
          justifyContent: { xs: 'flex-start', lg: 'flex-end' },
        }}>
          {onNavigateBack && (
            <Tooltip title="Back to Dashboard">
              <IconButton
                onClick={onNavigateBack}
                color="primary"
                size={isDesktop ? "large" : "medium"}
                sx={{
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    transform: isDesktop ? 'translateY(-1px)' : 'none',
                  },
                }}
              >
                <ArrowBackIcon />
              </IconButton>
            </Tooltip>
          )}
          <Tooltip title="Refresh data">
            <Button
              variant="contained"
              startIcon={isLoading ? <CircularProgress size={16} color="inherit" /> : <RefreshIcon />}
              onClick={handleRefresh}
              disabled={isLoading}
              size={isDesktop ? "large" : "medium"}
              sx={{
                borderRadius: { xs: 2, lg: 2.5 },
                padding: { xs: '8px 16px', lg: '10px 20px' },
                fontSize: { xs: '0.875rem', lg: '0.95rem' },
                minWidth: { xs: 'auto', lg: '120px' },
                transition: 'all 0.2s ease',
                '&:hover': {
                  transform: isDesktop ? 'translateY(-2px)' : 'none',
                  boxShadow: isDesktop ? '0 6px 16px rgba(25, 118, 210, 0.3)' : 'inherit',
                },
              }}
            >
              {isLoading ? 'Refreshing...' : 'Refresh'}
            </Button>
          </Tooltip>
          
          {/* Re-predict All button */}
          <Tooltip title="Re-predict all visible stocks">
            <Button
              variant="outlined"
              color="secondary"
              startIcon={regenerateLoading ? <CircularProgress size={16} color="inherit" /> : <PredictIcon />}
              onClick={handleRepredictAll}
              disabled={isLoading || stocksWithPredictions.length === 0}
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
          </Tooltip>
        </Box>
      </Box>

      {/* Search and Stats */}
      <Paper sx={{ 
        p: { xs: 2, lg: 3 }, 
        mb: { xs: 3, lg: 4 },
        borderRadius: { xs: 2, lg: 3 },
        boxShadow: isDesktop ? '0 2px 12px rgba(0,0,0,0.08)' : 1,
      }}>
        <Box sx={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: { xs: 'flex-start', lg: 'center' },
          flexDirection: { xs: 'column', lg: 'row' },
          gap: { xs: 2, lg: 3 },
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
            sx={{ 
              minWidth: { xs: '100%', sm: 300, lg: 400 },
              '& .MuiOutlinedInput-root': {
                borderRadius: { xs: 2, lg: 2.5 },
                fontSize: { xs: '0.875rem', lg: '0.95rem' },
                padding: { lg: '4px 14px' },
              },
            }}
          />
          
          <Box sx={{ 
            display: 'flex', 
            gap: { xs: 2, lg: 4 }, 
            alignItems: 'center',
            flexWrap: 'wrap',
            width: { xs: '100%', lg: 'auto' },
            justifyContent: { xs: 'space-between', lg: 'flex-end' },
          }}>
            <Typography variant="body2" color="text.secondary" sx={{
              fontSize: { xs: '0.8rem', lg: '0.875rem' },
            }}>
              Showing {paginatedStocks.length} of {sortedStocks.length} stocks
            </Typography>
            {searchTerm && (
              <Typography variant="body2" color="text.secondary" sx={{
                fontSize: { xs: '0.8rem', lg: '0.875rem' },
              }}>
                Filtered from {stocksWithPredictions.length} total
              </Typography>
            )}
          </Box>
        </Box>

        {/* Quick stats */}
        <Box sx={{ 
          display: 'flex', 
          gap: { xs: 2, lg: 3 }, 
          flexWrap: 'wrap',
          justifyContent: { xs: 'space-between', lg: 'flex-start' },
        }}>
          <Typography variant="body2" color="text.secondary" sx={{
            fontSize: { xs: '0.8rem', lg: '0.875rem' },
          }}>
            Total Stocks: {stocksWithPredictions.length}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{
            fontSize: { xs: '0.8rem', lg: '0.875rem' },
          }}>
            With Predictions: {stocksWithPredictions.filter(s => s.prediction).length}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{
            fontSize: { xs: '0.8rem', lg: '0.875rem' },
          }}>
            Bullish: {stocksWithPredictions.filter(s => s.prediction?.prediction === 'UP').length}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{
            fontSize: { xs: '0.8rem', lg: '0.875rem' },
          }}>
            Bearish: {stocksWithPredictions.filter(s => s.prediction?.prediction === 'DOWN').length}
          </Typography>
        </Box>
      </Paper>

      {/* Error handling */}
      {hasError && stocks.error && (
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
        <Paper sx={{
          borderRadius: { xs: 2, lg: 3 },
          boxShadow: isDesktop ? '0 2px 12px rgba(0,0,0,0.08)' : 1,
          overflow: 'hidden',
        }}>
          <TableContainer sx={{
            maxHeight: isDesktop ? '70vh' : '60vh',
            '&::-webkit-scrollbar': {
              width: isDesktop ? '8px' : '6px',
              height: isDesktop ? '8px' : '6px',
            },
            '&::-webkit-scrollbar-track': {
              background: '#f1f1f1',
              borderRadius: '4px',
            },
            '&::-webkit-scrollbar-thumb': {
              background: '#c1c1c1',
              borderRadius: '4px',
              '&:hover': {
                background: '#a1a1a1',
              },
            },
          }}>
            <Table stickyHeader sx={{
              '& .MuiTableHead-root': {
                '& .MuiTableCell-root': {
                  backgroundColor: isDesktop ? '#f8f9fa' : 'inherit',
                  fontWeight: 600,
                  fontSize: { xs: '0.8rem', lg: '0.9rem' },
                  padding: { xs: '12px 8px', lg: '16px 16px' },
                  borderBottom: isDesktop ? '2px solid #e0e0e0' : '1px solid #e0e0e0',
                },
              },
              '& .MuiTableBody-root': {
                '& .MuiTableRow-root': {
                  transition: 'background-color 0.2s ease',
                  '&:hover': {
                    backgroundColor: isDesktop ? '#f5f5f5' : 'inherit',
                    transform: isDesktop ? 'scale(1.01)' : 'none',
                  },
                  '& .MuiTableCell-root': {
                    padding: { xs: '8px 8px', lg: '12px 16px' },
                    fontSize: { xs: '0.8rem', lg: '0.9rem' },
                    borderBottom: isDesktop ? '1px solid #f0f0f0' : '1px solid #e0e0e0',
                  },
                },
              },
            }}>
              <TableHead>
                <TableRow>
                  <TableCell>
                    <TableSortLabel
                      active={sortConfig.field === 'symbol'}
                      direction={sortConfig.field === 'symbol' ? sortConfig.direction : 'asc'}
                      onClick={() => handleSort('symbol')}
                      sx={{
                        '& .MuiTableSortLabel-root': {
                          fontSize: { xs: '0.8rem', lg: '0.9rem' },
                          fontWeight: 600,
                        },
                      }}
                    >
                      Symbol
                    </TableSortLabel>
                  </TableCell>
                  <TableCell sx={{ display: { xs: 'none', sm: 'table-cell' } }}>
                    <TableSortLabel
                      active={sortConfig.field === 'name'}
                      direction={sortConfig.field === 'name' ? sortConfig.direction : 'asc'}
                      onClick={() => handleSort('name')}
                      sx={{
                        '& .MuiTableSortLabel-root': {
                          fontSize: { xs: '0.8rem', lg: '0.9rem' },
                          fontWeight: 600,
                        },
                      }}
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
            <Box sx={{ 
              display: 'flex', 
              justifyContent: { xs: 'center', lg: 'space-between' },
              alignItems: 'center',
              p: { xs: 2, lg: 3 },
              backgroundColor: isDesktop ? '#fafafa' : 'transparent',
              borderTop: isDesktop ? '1px solid #e0e0e0' : 'none',
              flexDirection: { xs: 'column', lg: 'row' },
              gap: { xs: 2, lg: 0 },
            }}>
              <Typography variant="body2" color="text.secondary" sx={{
                fontSize: { xs: '0.8rem', lg: '0.875rem' },
                order: { xs: 2, lg: 1 },
              }}>
                Page {currentPage} of {totalPages} ({sortedStocks.length} total stocks)
              </Typography>
              <Pagination
                count={totalPages}
                page={currentPage}
                onChange={handlePageChange}
                color="primary"
                showFirstButton={isDesktop}
                showLastButton={isDesktop}
                size={isDesktop ? "large" : "medium"}
                siblingCount={isDesktop ? 2 : 1}
                boundaryCount={isDesktop ? 2 : 1}
                sx={{
                  order: { xs: 1, lg: 2 },
                  '& .MuiPaginationItem-root': {
                    fontSize: { xs: '0.8rem', lg: '0.9rem' },
                    minWidth: { xs: '32px', lg: '40px' },
                    height: { xs: '32px', lg: '40px' },
                    borderRadius: { xs: 1, lg: 1.5 },
                    transition: 'all 0.2s ease',
                    '&:hover': {
                      transform: isDesktop ? 'translateY(-1px)' : 'none',
                      boxShadow: isDesktop ? '0 2px 8px rgba(0,0,0,0.1)' : 'none',
                    },
                  },
                }}
              />
            </Box>
          )}

          {/* No results message */}
          {sortedStocks.length === 0 && !isLoading && (
            <Box sx={{ 
              p: { xs: 4, lg: 6 }, 
              textAlign: 'center',
              minHeight: { xs: '200px', lg: '300px' },
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
            }}>
              <Typography variant="h6" color="text.secondary" sx={{
                fontSize: { xs: '1.125rem', lg: '1.25rem' },
                mb: 1,
              }}>
                {searchTerm ? 'No stocks found matching your search' : 'No stocks available'}
              </Typography>
              {searchTerm && (
                <Typography variant="body2" color="text.secondary" sx={{ 
                  mt: 1,
                  fontSize: { xs: '0.875rem', lg: '0.95rem' },
                }}>
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