import { useCallback } from 'react';
import { useApi } from './useApi';
import { StockService } from '../services/stockService';
import type { SystemStatus } from '../types/api';

export function useSystemStatus() {
  const apiCall = useCallback(() => StockService.getSystemStatus(), []);
  return useApi<SystemStatus>(apiCall);
}