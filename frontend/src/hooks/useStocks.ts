import { useCallback } from 'react';
import { useApi } from './useApi';
import { StockService } from '../services/stockService';
import type { StockInfo } from '../types/api';

export function useStocks(limit?: number, search?: string) {
  const apiCall = useCallback(() => StockService.getStocks(limit, search), [limit, search]);
  return useApi<{
    stocks: StockInfo[];
    total_count: number;
    returned_count: number;
    has_more: boolean;
  }>(apiCall);
}