import api from './api';
import type { 
  SystemStatus 
} from '../types/api';

export class StockService {
  // Get system status
  static async getSystemStatus(): Promise<SystemStatus> {
    const response = await api.get<SystemStatus>('/api/system/status');
    return response.data;
  }
}