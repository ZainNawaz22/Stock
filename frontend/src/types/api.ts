// API Response Types
export interface StockData {
  date: string;
  symbol: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  sma_50?: number;
  sma_200?: number;
  rsi_14?: number;
  macd?: number;
  macd_signal?: number;
  macd_histogram?: number;
}

export interface StockInfo {
  symbol: string;
  name: string;
  current_price: number;
  change: number;
  change_percent: number;
  volume: number;
  last_updated: string;
  prediction?: PredictionResult;
}

export interface PredictionResult {
  symbol: string;
  current_price: number;
  prediction: 'UP' | 'DOWN';
  confidence: number;
  model_accuracy: number;
  last_updated: string;
}

export interface SystemStatus {
  status: string;
  health: {
    score: number;
    status: string;
    issues: string[];
  };
  data: {
    total_symbols: number;
    total_records: number;
    storage_size_mb: number;
    latest_data_date: string | null;
    oldest_data_date: string | null;
    data_directory: string;
    sample_size: number;
  };
  models: {
    available_count: number;
    sample_checked: number;
    estimated_total: number;
  };
  cache: {
    predictions_cached: number;
    summaries_cached: number;
    indicators_cached: number;
    last_updated: string | null;
    cache_expiry_minutes: number;
  };
  performance: {
    optimized: boolean;
    parallel_processing: boolean;
    caching_enabled: boolean;
    max_symbols_per_request: number;
    max_data_points_default: number;
  };
  api_started: string | null;
  version: string;
  timestamp: string;
}

export interface StockDataResponse {
  symbol: string;
  data: StockData[];
  technical_indicators: {
    sma_50: number[];
    sma_200: number[];
    rsi_14: number[];
    macd: number[];
    macd_signal: number[];
    macd_histogram: number[];
  };
}

export interface ApiError {
  detail: string;
  status_code: number;
}