import React from 'react';
import { Alert, AlertTitle, Box, Button } from '@mui/material';
import type { ApiError } from '../../types/api';

interface ErrorMessageProps {
  error: ApiError;
  onRetry?: () => void;
}

export const ErrorMessage: React.FC<ErrorMessageProps> = ({ error, onRetry }) => {
  return (
    <Box sx={{ margin: 2 }}>
      <Alert 
        severity="error" 
        action={
          onRetry && (
            <Button color="inherit" size="small" onClick={onRetry}>
              Retry
            </Button>
          )
        }
      >
        <AlertTitle>Error {error.status_code > 0 ? `(${error.status_code})` : ''}</AlertTitle>
        {error.detail}
      </Alert>
    </Box>
  );
};