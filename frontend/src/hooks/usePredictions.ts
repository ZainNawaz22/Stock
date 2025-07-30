import { useCallback } from 'react';
import { useApi } from './useApi';
import { StockService } from '../services/stockService';
import type { PredictionResult } from '../types/api';

export function usePredictions() {
  const apiCall = useCallback(() => StockService.getPredictions(), []);
  return useApi<PredictionResult[]>(apiCall);
}