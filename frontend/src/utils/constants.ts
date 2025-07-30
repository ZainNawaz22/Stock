// Application constants

export const REFRESH_INTERVALS = {
  FAST: 30000,    // 30 seconds
  NORMAL: 60000,  // 1 minute
  SLOW: 300000,   // 5 minutes
} as const;

export const CHART_COLORS = {
  PRIMARY: '#1976d2',
  SECONDARY: '#dc004e',
  SUCCESS: '#2e7d32',
  WARNING: '#ed6c02',
  ERROR: '#d32f2f',
  UP: '#4caf50',
  DOWN: '#f44336',
} as const;

export const PREDICTION_COLORS = {
  UP: '#4caf50',
  DOWN: '#f44336',
} as const;

export const TIME_PERIODS = [
  { label: '1D', value: '1D' },
  { label: '7D', value: '7D' },
  { label: '1M', value: '1M' },
  { label: '3M', value: '3M' },
  { label: '6M', value: '6M' },
  { label: '1Y', value: '1Y' },
] as const;

export const TECHNICAL_INDICATORS = {
  SMA_50: 'SMA 50',
  SMA_200: 'SMA 200',
  RSI_14: 'RSI 14',
  MACD: 'MACD',
} as const;