import React from 'react';
import { Card, CardContent, Typography, Box, Button } from '@mui/material';

export const Dashboard: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" component="h1" sx={{ mb: 2 }}>
        Dashboard - Testing (No API Calls)
      </Typography>
      
      <Card>
        <CardContent>
          <Typography color="textSecondary" gutterBottom>
            System Status
          </Typography>
          <Typography variant="h5" component="h2">
            Testing Mode - No Auto Requests
          </Typography>
          <Typography variant="body2" color="text.secondary">
            This is a static dashboard for testing - no API calls are made automatically.
          </Typography>
          <Button variant="outlined" sx={{ mt: 2 }}>
            Manual Test Button (No API Call)
          </Button>
        </CardContent>
      </Card>
    </Box>
  );
};
