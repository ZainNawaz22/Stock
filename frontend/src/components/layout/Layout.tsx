import React, { useState } from 'react';
import {
  Box,
  useTheme,
  useMediaQuery,
  Typography,
  IconButton,
  Breadcrumbs,
  Link,
  Chip,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  List as ListIcon,
  ChevronRight as ChevronRightIcon,
} from '@mui/icons-material';
import { Sidebar } from './Sidebar';

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
  const theme = useTheme();
  const isDesktop = useMediaQuery(theme.breakpoints.up('lg'));
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const sidebarWidth = sidebarCollapsed ? 80 : 280;

  // Get page title and description
  const getPageInfo = () => {
    switch (currentView) {
      case 'dashboard':
        return {
          title: 'Dashboard',
          description: 'Overview of your stock portfolio and market insights',
          icon: <DashboardIcon />,
        };
      case 'stocks':
        return {
          title: 'Stock List',
          description: 'Browse and analyze all available stocks with AI predictions',
          icon: <ListIcon />,
        };
      default:
        return {
          title: 'PSX AI Advisor',
          description: 'Stock analysis platform',
          icon: <DashboardIcon />,
        };
    }
  };

  const pageInfo = getPageInfo();

  if (!isDesktop) {
    // Mobile fallback - simple layout without sidebar
    return (
      <Box sx={{ minHeight: '100vh', backgroundColor: theme.palette.background.default }}>
        <Box sx={{ 
          backgroundColor: theme.palette.background.paper,
          borderBottom: `1px solid ${theme.palette.divider}`,
          p: 2,
        }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            PSX AI Advisor
          </Typography>
          <Box sx={{ mt: 1, display: 'flex', gap: 1 }}>
            <Chip
              label="Dashboard"
              variant={currentView === 'dashboard' ? 'filled' : 'outlined'}
              onClick={onNavigateToDashboard}
              size="small"
            />
            <Chip
              label="Stocks"
              variant={currentView === 'stocks' ? 'filled' : 'outlined'}
              onClick={onNavigateToStocks}
              size="small"
            />
          </Box>
        </Box>
        <Box sx={{ p: 2 }}>
          {children}
        </Box>
      </Box>
    );
  }

  return (
    <Box sx={{ 
      display: 'flex',
      minHeight: '100vh',
      backgroundColor: theme.palette.background.default,
    }}>
      {/* Sidebar */}
      <Sidebar
        currentView={currentView}
        onNavigateToDashboard={onNavigateToDashboard || (() => {})}
        onNavigateToStocks={onNavigateToStocks || (() => {})}
        isCollapsed={sidebarCollapsed}
      />

      {/* Main Content Area */}
      <Box
        sx={{
          flex: 1,
          marginLeft: `${sidebarWidth}px`,
          transition: 'margin-left 0.3s ease',
          display: 'flex',
          flexDirection: 'column',
          minHeight: '100vh',
        }}
      >
        {/* Top Header Bar */}
        <Box
          sx={{
            backgroundColor: theme.palette.background.paper,
            borderBottom: `1px solid ${theme.palette.divider}`,
            px: 3,
            py: 2,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            minHeight: 70,
            boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {/* Sidebar Toggle */}
            <IconButton
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              sx={{
                p: 1,
                borderRadius: 2,
                transition: 'all 0.2s ease',
                '&:hover': {
                  backgroundColor: theme.palette.action.hover,
                  transform: 'scale(1.05)',
                },
              }}
            >
              <MenuIcon />
            </IconButton>

            {/* Breadcrumbs */}
            <Breadcrumbs 
              separator={<ChevronRightIcon fontSize="small" />}
              sx={{
                '& .MuiBreadcrumbs-li': {
                  fontSize: '0.9rem',
                },
              }}
            >
              <Link
                underline="hover"
                color="inherit"
                onClick={onNavigateToDashboard}
                sx={{ 
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 0.5,
                  fontWeight: 500,
                  '&:hover': {
                    color: theme.palette.primary.main,
                  },
                }}
              >
                <DashboardIcon fontSize="small" />
                Home
              </Link>
              {currentView !== 'dashboard' && (
                <Typography 
                  color="text.primary" 
                  sx={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: 0.5,
                    fontWeight: 600,
                  }}
                >
                  {pageInfo.icon}
                  {pageInfo.title}
                </Typography>
              )}
            </Breadcrumbs>
          </Box>

          {/* Page Title Section */}
          <Box sx={{ textAlign: 'right' }}>
            <Typography variant="h6" sx={{ fontWeight: 600, fontSize: '1.1rem' }}>
              {pageInfo.title}
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.8rem' }}>
              {pageInfo.description}
            </Typography>
          </Box>
        </Box>

        {/* Page Content */}
        <Box
          sx={{
            flex: 1,
            p: 3,
            backgroundColor: theme.palette.background.default,
            overflow: 'auto',
            '&::-webkit-scrollbar': {
              width: '8px',
            },
            '&::-webkit-scrollbar-track': {
              background: '#f1f1f1',
              borderRadius: '4px',
            },
            '&::-webkit-scrollbar-thumb': {
              background: '#c1c1c1',
              borderRadius: '4px',
              '&:hover': {
                background: '#a1a1a1',
              },
            },
          }}
        >
          <Box
            sx={{
              maxWidth: '1400px',
              margin: '0 auto',
              width: '100%',
            }}
          >
            {children}
          </Box>
        </Box>
      </Box>
    </Box>
  );
};