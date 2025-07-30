import { useCallback } from 'react';
import { useApi } from './useApi';
import { StockService } from '../services/stockService';
import type { PredictionResult } from '../types/api';

export function usePredictions(symbols?: string[], forceRefresh?: boolean) {
  const apiCall = useCallback(() => StockService.getPredictions(symbols, forceRefresh), [symbols, forceRefresh]);
  return useApi<{
    predictions: PredictionResult[];
    total_count: number;
    successful_count: number;
    failed_count: number;
    cache_used: boolean;
    last_updated: string;
  }>(apiCall);
}