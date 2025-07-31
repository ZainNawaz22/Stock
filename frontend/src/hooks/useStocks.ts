import { useCallback } from 'react';
import { useApi } from './useApi';
import { StockService } from '../services/stockService';
import type { StockInfo } from '../types/api';

export function useStocks(limit?: number, search?: string, includePredictions: boolean = true) {
  const apiCall = useCallback(() => StockService.getStocks(limit, search, includePredictions), [limit, search, includePredictions]);
  return useApi<{
    stocks: StockInfo[];
    total_count: number;
    returned_count: number;
    has_more: boolean;
  }>(apiCall);
}