import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  IconButton,
  Button,
  Breadcrumbs,
  Link,
} from '@mui/material';
import { TrendingUp, Dashboard as DashboardIcon, List as ListIcon } from '@mui/icons-material';

interface LayoutProps {
  children: React.ReactNode;
  currentView?: 'dashboard' | 'stocks';
  onNavigateToDashboard?: () => void;
  onNavigateToStocks?: () => void;
}

export const Layout: React.FC<LayoutProps> = ({ 
  children, 
  currentView = 'dashboard',
  onNavigateToDashboard,
  onNavigateToStocks 
}) => {
  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          <IconButton
            size="large"
            edge="start"
            color="inherit"
            aria-label="logo"
            sx={{ mr: 2 }}
            onClick={onNavigateToDashboard}
          >
            <TrendingUp />
          </IconButton>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            PSX AI Advisor
          </Typography>
          
          {/* Navigation buttons */}
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              color="inherit"
              startIcon={<DashboardIcon />}
              onClick={onNavigateToDashboard}
              variant={currentView === 'dashboard' ? 'outlined' : 'text'}
              sx={{ 
                borderColor: currentView === 'dashboard' ? 'white' : 'transparent',
                '&:hover': { backgroundColor: 'rgba(255, 255, 255, 0.1)' }
              }}
            >
              Dashboard
            </Button>
            <Button
              color="inherit"
              startIcon={<ListIcon />}
              onClick={onNavigateToStocks}
              variant={currentView === 'stocks' ? 'outlined' : 'text'}
              sx={{ 
                borderColor: currentView === 'stocks' ? 'white' : 'transparent',
                '&:hover': { backgroundColor: 'rgba(255, 255, 255, 0.1)' }
              }}
            >
              Stocks
            </Button>
          </Box>
        </Toolbar>
      </AppBar>
      
      {/* Breadcrumbs */}
      <Container maxWidth="xl" sx={{ mt: 2 }}>
        <Breadcrumbs aria-label="breadcrumb">
          <Link
            underline="hover"
            color={currentView === 'dashboard' ? 'primary' : 'inherit'}
            onClick={onNavigateToDashboard}
            sx={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 0.5 }}
          >
            <DashboardIcon fontSize="small" />
            Dashboard
          </Link>
          {currentView === 'stocks' && (
            <Typography color="text.primary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <ListIcon fontSize="small" />
              Stocks
            </Typography>
          )}
        </Breadcrumbs>
      </Container>
      
      <Container maxWidth="xl" sx={{ mt: 2, mb: 4 }}>
        {children}
      </Container>
    </Box>
  );
};