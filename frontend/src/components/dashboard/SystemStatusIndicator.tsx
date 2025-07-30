import React from 'react';
import {
  Box,
  Chip,
  Typography,
  Tooltip,
  CircularProgress,
} from '@mui/material';
import {
  CheckCircle,
  Error,
  Warning,
  Info,
} from '@mui/icons-material';
import type { SystemStatus } from '../../types/api';

interface SystemStatusIndicatorProps {
  systemStatus: SystemStatus | null;
  loading?: boolean;
  lastUpdated?: string;
}

const getStatusConfig = (status: string, healthScore: number) => {
  if (status === 'operational' && healthScore >= 90) {
    return {
      color: 'success' as const,
      icon: CheckCircle,
      label: 'Operational',
    };
  } else if (status === 'operational' && healthScore >= 70) {
    return {
      color: 'warning' as const,
      icon: Warning,
      label: 'Degraded',
    };
  } else if (status === 'error' || healthScore < 50) {
    return {
      color: 'error' as const,
      icon: Error,
      label: 'Error',
    };
  } else {
    return {
      color: 'info' as const,
      icon: Info,
      label: 'Unknown',
    };
  }
};

export const SystemStatusIndicator: React.FC<SystemStatusIndicatorProps> = ({
  systemStatus,
  loading = false,
  lastUpdated,
}) => {
  if (loading) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <CircularProgress size={16} />
        <Typography variant="body2" color="text.secondary">
          Checking system status...
        </Typography>
      </Box>
    );
  }

  if (!systemStatus) {
    return (
      <Chip
        icon={<Info />}
        label="Status Unknown"
        color="default"
        variant="outlined"
        size="small"
      />
    );
  }

  const statusConfig = getStatusConfig(
    systemStatus.status,
    systemStatus.health?.score || 0
  );

  const IconComponent = statusConfig.icon;

  const tooltipContent = (
    <Box>
      <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
        System Health: {systemStatus.health?.score || 0}%
      </Typography>
      <Typography variant="body2">
        Status: {systemStatus.health?.status || 'Unknown'}
      </Typography>
      {systemStatus.health?.issues && systemStatus.health.issues.length > 0 && (
        <Box sx={{ mt: 1 }}>
          <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
            Issues:
          </Typography>
          {systemStatus.health.issues.map((issue, index) => (
            <Typography key={index} variant="body2" sx={{ ml: 1 }}>
              • {issue}
            </Typography>
          ))}
        </Box>
      )}
      {lastUpdated && (
        <Typography variant="body2" sx={{ mt: 1, fontStyle: 'italic' }}>
          Last updated: {new Date(lastUpdated).toLocaleString()}
        </Typography>
      )}
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      <Tooltip title={tooltipContent} arrow>
        <Chip
          icon={<IconComponent />}
          label={statusConfig.label}
          color={statusConfig.color}
          variant="filled"
          size="small"
        />
      </Tooltip>
      {lastUpdated && (
        <Typography variant="body2" color="text.secondary">
          Updated: {new Date(lastUpdated).toLocaleString()}
        </Typography>
      )}
    </Box>
  );
};