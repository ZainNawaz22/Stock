import { createTheme } from '@mui/material/styles';
import { CHART_COLORS } from '../utils/constants';

export const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
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
    background: {
      default: '#f8f9fa',
      paper: '#ffffff',
    },
    grey: {
      50: '#fafafa',
      100: '#f5f5f5',
      200: '#eeeeee',
      300: '#e0e0e0',
      400: '#bdbdbd',
      500: '#9e9e9e',
      600: '#757575',
      700: '#616161',
      800: '#424242',
      900: '#212121',
    },
  },
  typography: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
      '@media (min-width:1024px)': {
        fontSize: '3rem',
      },
      '@media (min-width:1440px)': {
        fontSize: '3.5rem',
      },
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      '@media (min-width:1024px)': {
        fontSize: '2.25rem',
      },
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
      '@media (min-width:1024px)': {
        fontSize: '2rem',
      },
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600,
      '@media (min-width:1024px)': {
        fontSize: '1.75rem',
      },
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
      '@media (min-width:1024px)': {
        fontSize: '1.375rem',
      },
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 600,
      '@media (min-width:1024px)': {
        fontSize: '1.125rem',
      },
    },
    body1: {
      '@media (min-width:1024px)': {
        fontSize: '1rem',
        lineHeight: 1.6,
      },
    },
    body2: {
      '@media (min-width:1024px)': {
        fontSize: '0.925rem',
        lineHeight: 1.5,
      },
    },
  },
  breakpoints: {
    values: {
      xs: 0,
      sm: 600,
      md: 900,
      lg: 1200,
      xl: 1536,
    },
  },
  spacing: (factor: number) => `${0.5 * factor}rem`,
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        '@media (min-width:1024px)': {
          '*::-webkit-scrollbar': {
            width: '8px',
            height: '8px',
          },
          '*::-webkit-scrollbar-track': {
            background: '#f1f1f1',
            borderRadius: '4px',
          },
          '*::-webkit-scrollbar-thumb': {
            background: '#c1c1c1',
            borderRadius: '4px',
            '&:hover': {
              background: '#a1a1a1',
            },
          },
        },
      },
    },
    MuiContainer: {
      styleOverrides: {
        root: {
          '@media (min-width:1024px)': {
            paddingLeft: '24px',
            paddingRight: '24px',
          },
          '@media (min-width:1440px)': {
            paddingLeft: '32px',
            paddingRight: '32px',
          },
        },
        maxWidthXl: {
          '@media (min-width:1536px)': {
            maxWidth: '1920px',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
          borderRadius: 12,
          transition: 'box-shadow 0.3s ease, transform 0.2s ease',
          '&:hover': {
            boxShadow: '0 4px 16px rgba(0,0,0,0.12)',
            '@media (min-width:1024px)': {
              transform: 'translateY(-2px)',
            },
          },
          '@media (min-width:1024px)': {
            borderRadius: 16,
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
          fontWeight: 500,
          transition: 'all 0.2s ease',
          '@media (min-width:1024px)': {
            borderRadius: 10,
            fontSize: '0.95rem',
            padding: '10px 20px',
            '&:hover': {
              transform: 'translateY(-1px)',
              boxShadow: '0 4px 8px rgba(0,0,0,0.15)',
            },
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          '@media (min-width:1024px)': {
            borderRadius: 12,
          },
        },
        elevation1: {
          boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
        },
        elevation4: {
          boxShadow: '0 4px 16px rgba(0,0,0,0.12)',
        },
      },
    },
    MuiTableContainer: {
      styleOverrides: {
        root: {
          '@media (min-width:1024px)': {
            borderRadius: 12,
            '& .MuiTable-root': {
              '& .MuiTableHead-root': {
                '& .MuiTableCell-root': {
                  backgroundColor: '#f8f9fa',
                  fontWeight: 600,
                  fontSize: '0.9rem',
                  padding: '16px',
                },
              },
              '& .MuiTableBody-root': {
                '& .MuiTableRow-root': {
                  '&:hover': {
                    backgroundColor: '#f5f5f5',
                  },
                  '& .MuiTableCell-root': {
                    padding: '12px 16px',
                    fontSize: '0.9rem',
                  },
                },
              },
            },
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          '@media (min-width:1024px)': {
            fontSize: '0.8rem',
            height: '28px',
            borderRadius: '14px',
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '@media (min-width:1024px)': {
            '& .MuiOutlinedInput-root': {
              borderRadius: '10px',
              transition: 'all 0.2s ease',
              '&:hover': {
                '& .MuiOutlinedInput-notchedOutline': {
                  borderColor: '#bdbdbd',
                },
              },
              '&.Mui-focused': {
                '& .MuiOutlinedInput-notchedOutline': {
                  borderWidth: '2px',
                },
              },
            },
          },
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          '@media (min-width:1024px)': {
            boxShadow: '0 2px 12px rgba(0,0,0,0.08)',
          },
        },
      },
    },
  },
});