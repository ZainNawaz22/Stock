import { useCallback } from 'react';
import { useApi } from './useApi';
import { StockService } from '../services/stockService';
import type { StockInfo } from '../types/api';

export function useStocks() {
  const apiCall = useCallback(() => StockService.getStocks(), []);
  return useApi<StockInfo[]>(apiCall);
}