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

  return {
    regenerate,
    loading,
    error,
  };
}
