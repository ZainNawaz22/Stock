import React from 'react';
import {
  Box,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Divider,
  Avatar,
  Chip,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  List as StocksIcon,
  TrendingUp,
  Analytics as AnalyticsIcon,
  Settings as SettingsIcon,
  Person as PersonIcon,
  Notifications as NotificationsIcon,
} from '@mui/icons-material';

interface SidebarProps {
  currentView: 'dashboard' | 'stocks';
  onNavigateToDashboard: () => void;
  onNavigateToStocks: () => void;
  isCollapsed?: boolean;
}

interface NavigationItem {
  id: string;
  label: string;
  icon: React.ReactNode;
  view: 'dashboard' | 'stocks';
  onClick: () => void;
  badge?: string | number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentView,
  onNavigateToDashboard,
  onNavigateToStocks,
  isCollapsed = false,
}) => {
  const theme = useTheme();

  const navigationItems: NavigationItem[] = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      icon: <DashboardIcon />,
      view: 'dashboard',
      onClick: onNavigateToDashboard,
    },
    {
      id: 'stocks',
      label: 'Stock List',
      icon: <StocksIcon />,
      view: 'stocks',
      onClick: onNavigateToStocks,
    },
  ];

  const secondaryItems = [
    {
      id: 'analytics',
      label: 'Analytics',
      icon: <AnalyticsIcon />,
      disabled: true,
    },
    {
      id: 'notifications',
      label: 'Notifications',
      icon: <NotificationsIcon />,
      badge: '3',
      disabled: true,
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: <SettingsIcon />,
      disabled: true,
    },
  ];

  const sidebarWidth = isCollapsed ? 80 : 280;

  return (
    <Box
      sx={{
        width: sidebarWidth,
        height: '100vh',
        backgroundColor: theme.palette.mode === 'dark' ? '#1e1e1e' : '#ffffff',
        borderRight: `1px solid ${theme.palette.divider}`,
        display: 'flex',
        flexDirection: 'column',
        position: 'fixed',
        left: 0,
        top: 0,
        zIndex: theme.zIndex.drawer,
        transition: 'width 0.3s ease',
        boxShadow: '2px 0 8px rgba(0,0,0,0.1)',
      }}
    >
      {/* Logo/Brand Section */}
      <Box
        sx={{
          p: isCollapsed ? 2 : 3,
          display: 'flex',
          alignItems: 'center',
          gap: 2,
          borderBottom: `1px solid ${theme.palette.divider}`,
          minHeight: 80,
        }}
      >
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 40,
            height: 40,
            borderRadius: 2,
            background: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.primary.light})`,
            color: 'white',
            flexShrink: 0,
          }}
        >
          <TrendingUp sx={{ fontSize: 24 }} />
        </Box>
        {!isCollapsed && (
          <Box>
            <Typography
              variant="h6"
              sx={{
                fontWeight: 700,
                fontSize: '1.1rem',
                background: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.primary.light})`,
                backgroundClip: 'text',
                color: 'transparent',
                lineHeight: 1.2,
              }}
            >
              PSX AI Advisor
            </Typography>
            <Typography
              variant="caption"
              sx={{
                color: theme.palette.text.secondary,
                fontSize: '0.75rem',
              }}
            >
              Stock Analysis Platform
            </Typography>
          </Box>
        )}
      </Box>

      {/* Navigation Section */}
      <Box sx={{ flex: 1, py: 2 }}>
        {!isCollapsed && (
          <Typography
            variant="overline"
            sx={{
              px: 3,
              pb: 1,
              color: theme.palette.text.secondary,
              fontSize: '0.7rem',
              fontWeight: 600,
              letterSpacing: 1,
            }}
          >
            MAIN NAVIGATION
          </Typography>
        )}
        
        <List sx={{ px: isCollapsed ? 1 : 2 }}>
          {navigationItems.map((item) => {
            const isActive = currentView === item.view;
            
            return (
              <ListItem key={item.id} disablePadding sx={{ mb: 0.5 }}>
                <ListItemButton
                  onClick={item.onClick}
                  sx={{
                    borderRadius: 2,
                    minHeight: 48,
                    px: isCollapsed ? 1 : 2,
                    py: 1.5,
                    backgroundColor: isActive
                      ? alpha(theme.palette.primary.main, 0.1)
                      : 'transparent',
                    border: isActive
                      ? `1px solid ${alpha(theme.palette.primary.main, 0.2)}`
                      : '1px solid transparent',
                    transition: 'all 0.2s ease',
                    '&:hover': {
                      backgroundColor: isActive
                        ? alpha(theme.palette.primary.main, 0.15)
                        : alpha(theme.palette.action.hover, 0.8),
                      transform: 'translateX(6px)',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                    },
                  }}
                >
                  <ListItemIcon
                    sx={{
                      minWidth: isCollapsed ? 'auto' : 40,
                      color: isActive
                        ? theme.palette.primary.main
                        : theme.palette.text.secondary,
                      mr: isCollapsed ? 0 : 1,
                    }}
                  >
                    {item.icon}
                  </ListItemIcon>
                  {!isCollapsed && (
                    <ListItemText
                      primary={item.label}
                      sx={{
                        '& .MuiListItemText-primary': {
                          fontSize: '0.9rem',
                          fontWeight: isActive ? 600 : 500,
                          color: isActive
                            ? theme.palette.primary.main
                            : theme.palette.text.primary,
                        },
                      }}
                    />
                  )}
                  {!isCollapsed && item.badge && (
                    <Chip
                      label={item.badge}
                      size="small"
                      sx={{
                        height: 20,
                        fontSize: '0.7rem',
                        backgroundColor: theme.palette.error.main,
                        color: 'white',
                      }}
                    />
                  )}
                </ListItemButton>
              </ListItem>
            );
          })}
        </List>

        {/* Secondary Navigation */}
        {!isCollapsed && (
          <>
            <Divider sx={{ mx: 2, my: 3 }} />
            <Typography
              variant="overline"
              sx={{
                px: 3,
                pb: 1,
                color: theme.palette.text.secondary,
                fontSize: '0.7rem',
                fontWeight: 600,
                letterSpacing: 1,
              }}
            >
              COMING SOON
            </Typography>
          </>
        )}

        <List sx={{ px: isCollapsed ? 1 : 2 }}>
          {secondaryItems.map((item) => (
            <ListItem key={item.id} disablePadding sx={{ mb: 0.5 }}>
              <ListItemButton
                disabled={item.disabled}
                sx={{
                  borderRadius: 2,
                  minHeight: 48,
                  px: isCollapsed ? 1 : 2,
                  py: 1.5,
                  opacity: item.disabled ? 0.5 : 1,
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    backgroundColor: alpha(theme.palette.action.hover, 0.5),
                    transform: !item.disabled ? 'translateX(6px)' : 'none',
                    boxShadow: !item.disabled ? '0 2px 8px rgba(0,0,0,0.1)' : 'none',
                  },
                }}
              >
                <ListItemIcon
                  sx={{
                    minWidth: isCollapsed ? 'auto' : 40,
                    color: theme.palette.text.secondary,
                    mr: isCollapsed ? 0 : 1,
                  }}
                >
                  {item.icon}
                </ListItemIcon>
                {!isCollapsed && (
                  <ListItemText
                    primary={item.label}
                    sx={{
                      '& .MuiListItemText-primary': {
                        fontSize: '0.9rem',
                        fontWeight: 500,
                        color: theme.palette.text.secondary,
                      },
                    }}
                  />
                )}
                {!isCollapsed && item.badge && (
                  <Chip
                    label={item.badge}
                    size="small"
                    sx={{
                      height: 20,
                      fontSize: '0.7rem',
                      backgroundColor: theme.palette.warning.main,
                      color: 'white',
                    }}
                  />
                )}
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Box>

      {/* User Profile Section */}
      <Box
        sx={{
          p: isCollapsed ? 1 : 2,
          borderTop: `1px solid ${theme.palette.divider}`,
          backgroundColor: alpha(theme.palette.background.paper, 0.8),
        }}
      >
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: isCollapsed ? 0 : 2,
            p: 1,
            borderRadius: 2,
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            '&:hover': {
              backgroundColor: alpha(theme.palette.action.hover, 0.5),
            },
          }}
        >
          <Avatar
            sx={{
              width: 36,
              height: 36,
              backgroundColor: theme.palette.primary.main,
              fontSize: '0.9rem',
              fontWeight: 600,
            }}
          >
            <PersonIcon />
          </Avatar>
          {!isCollapsed && (
            <Box sx={{ flex: 1, overflow: 'hidden' }}>
              <Typography
                variant="body2"
                sx={{
                  fontWeight: 600,
                  fontSize: '0.85rem',
                  color: theme.palette.text.primary,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                PSX Trader
              </Typography>
              <Typography
                variant="caption"
                sx={{
                  color: theme.palette.text.secondary,
                  fontSize: '0.75rem',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                Stock Analyst
              </Typography>
            </Box>
          )}
        </Box>
      </Box>
    </Box>
  );
};
