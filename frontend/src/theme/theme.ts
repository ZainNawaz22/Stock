import { createTheme } from '@mui/material/styles';
import { CHART_COLORS } from '../utils/constants';

export const theme = createTheme({
  palette: {
    primary: {
      main: CHART_COLORS.PRIMARY,
    },
    secondary: {
      main: CHART_COLORS.SECONDARY,
    },
    success: {
      main: CHART_COLORS.SUCCESS,
    },
    warning: {
      main: CHART_COLORS.WARNING,
    },
    error: {
      main: CHART_COLORS.ERROR,
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 600,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          borderRadius: 8,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
        },
      },
    },
  },
});