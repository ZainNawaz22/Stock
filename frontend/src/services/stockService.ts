import api from './api';
import type { 
  SystemStatus,
  StockInfo,
  PredictionResult
} from '../types/api';

export class StockService {
  // Get system status
  static async getSystemStatus(): Promise<SystemStatus> {
    const response = await api.get<SystemStatus>('/api/system/status');
    return response.data;
  }

  // Get stocks list
  static async getStocks(limit?: number, search?: string): Promise<{
    stocks: StockInfo[];
    total_count: number;
    returned_count: number;
    has_more: boolean;
  }> {
    const params = new URLSearchParams();
    if (limit) params.append('limit', limit.toString());
    if (search) params.append('search', search);
    
    const response = await api.get(`/api/stocks?${params.toString()}`);
    return response.data;
  }

  // Get predictions
  static async getPredictions(symbols?: string[], forceRefresh?: boolean): Promise<{
    predictions: PredictionResult[];
    total_count: number;
    successful_count: number;
    failed_count: number;
    cache_used: boolean;
    last_updated: string;
  }> {
    const params = new URLSearchParams();
    if (symbols && symbols.length > 0) {
      params.append('symbols', symbols.join(','));
    }
    if (forceRefresh) {
      params.append('force_refresh', 'true');
    }
    
    const response = await api.get(`/api/predictions?${params.toString()}`);
    return response.data;
  }
}