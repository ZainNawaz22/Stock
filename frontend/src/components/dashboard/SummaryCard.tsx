import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Skeleton,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Assessment,
  Storage,
  Speed,
  CheckCircle,
  Error,
  Warning,
} from '@mui/icons-material';

interface SummaryCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: 'trending-up' | 'trending-down' | 'assessment' | 'storage' | 'speed' | 'check' | 'error' | 'warning';
  status?: 'success' | 'error' | 'warning' | 'info';
  loading?: boolean;
  trend?: {
    direction: 'up' | 'down';
    value: string;
  };
}

const iconMap = {
  'trending-up': TrendingUp,
  'trending-down': TrendingDown,
  'assessment': Assessment,
  'storage': Storage,
  'speed': Speed,
  'check': CheckCircle,
  'error': Error,
  'warning': Warning,
};

const statusColors = {
  success: 'success.main',
  error: 'error.main',
  warning: 'warning.main',
  info: 'info.main',
};

export const SummaryCard: React.FC<SummaryCardProps> = ({
  title,
  value,
  subtitle,
  icon,
  status = 'info',
  loading = false,
  trend,
}) => {
  const IconComponent = icon ? iconMap[icon] : Assessment;

  if (loading) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Skeleton variant="circular" width={24} height={24} sx={{ mr: 1 }} />
            <Skeleton variant="text" width="60%" />
          </Box>
          <Skeleton variant="text" width="40%" height={32} sx={{ mb: 1 }} />
          <Skeleton variant="text" width="80%" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ height: '100%', position: 'relative' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <IconComponent 
            sx={{ 
              color: statusColors[status], 
              mr: 1,
              fontSize: 24
            }} 
          />
          <Typography color="textSecondary" variant="body2" sx={{ fontWeight: 500 }}>
            {title}
          </Typography>
        </Box>
        
        <Typography variant="h4" component="h2" sx={{ mb: 1, fontWeight: 'bold' }}>
          {value}
        </Typography>
        
        {subtitle && (
          <Typography variant="body2" color="text.secondary">
            {subtitle}
          </Typography>
        )}

        {trend && (
          <Box sx={{ mt: 2, display: 'flex', alignItems: 'center' }}>
            <Chip
              icon={trend.direction === 'up' ? <TrendingUp /> : <TrendingDown />}
              label={trend.value}
              size="small"
              color={trend.direction === 'up' ? 'success' : 'error'}
              variant="outlined"
            />
          </Box>
        )}
      </CardContent>
    </Card>
  );
};