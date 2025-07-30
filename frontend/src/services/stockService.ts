import api from './api';
import type { 
  StockInfo, 
  StockDataResponse, 
  PredictionResult, 
  SystemStatus 
} from '../types/api';

export class StockService {
  // Get list of available stocks
  static async getStocks(): Promise<StockInfo[]> {
    const response = await api.get<StockInfo[]>('/api/stocks');
    return response.data;
  }

  // Get detailed data for a specific stock
  static async getStockData(symbol: string): Promise<StockDataResponse> {
    const response = await api.get<StockDataResponse>(`/api/stocks/${symbol}/data`);
    return response.data;
  }

  // Get all current predictions
  static async getPredictions(): Promise<PredictionResult[]> {
    const response = await api.get<PredictionResult[]>('/api/predictions');
    return response.data;
  }

  // Get system status
  static async getSystemStatus(): Promise<SystemStatus> {
    const response = await api.get<SystemStatus>('/api/system/status');
    return response.data;
  }

  // Get prediction for specific stock
  static async getStockPrediction(symbol: string): Promise<PredictionResult> {
    const response = await api.get<PredictionResult>(`/api/predictions/${symbol}`);
    return response.data;
  }
}