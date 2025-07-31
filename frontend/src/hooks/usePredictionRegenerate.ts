import { useCallback, useState } from 'react';
import { StockService } from '../services/stockService';
import type { PredictionResult, ApiError } from '../types/api';

interface UsePredictionRegenerateOptions {
  onSuccess?: (data: {
    predictions: PredictionResult[];
    total_count: number;
    successful_count: number;
    failed_count: number;
    cache_used: boolean;
    last_updated: string;
  }) => void;
  onError?: (error: ApiError) => void;
}

export function usePredictionRegenerate(options: UsePredictionRegenerateOptions = {}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const regenerate = useCallback(async (
    symbols?: string[], 
    retrainModels?: boolean, 
    limit?: number
  ) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await StockService.regeneratePredictions(symbols, retrainModels, limit);
      options.onSuccess?.(result);
      return result;
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError);
      options.onError?.(apiError);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [options]);

  const regenerateOne = useCallback(async (symbol: string, retrainModel?: boolean) => {
    setLoading(true);
    setError(null);
    try {
      const result = await StockService.regeneratePredictionForSymbol(symbol, retrainModel);
      // adapt to same success callback contract by wrapping single result
      options.onSuccess?.({
        predictions: [result],
        total_count: 1,
        successful_count: result.prediction ? 1 : 0,
        failed_count: result.prediction ? 0 : 1,
        cache_used: false,
        last_updated: new Date().toISOString(),
      });
      return result;
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError);
      options.onError?.(apiError);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [options]);

  return {
    regenerate,
    regenerateOne,
    loading,
    error,
  };
}
