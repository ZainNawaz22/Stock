// Application constants

export const CHART_COLORS = {
  PRIMARY: '#1976d2',
  SECONDARY: '#dc004e',
  SUCCESS: '#2e7d32',
  WARNING: '#ed6c02',
  ERROR: '#d32f2f',
  UP: '#4caf50',
  DOWN: '#f44336',
} as const;

// Desktop optimization constants
export const BREAKPOINTS = {
  MOBILE: 0,
  TABLET: 600,
  DESKTOP: 900,
  LARGE_DESKTOP: 1200,
  EXTRA_LARGE: 1536,
} as const;

export const LAYOUT_CONSTANTS = {
  HEADER_HEIGHT: {
    MOBILE: 56,
    DESKTOP: 64,
  },
  SIDEBAR_WIDTH: {
    COLLAPSED: 60,
    EXPANDED: 280,
  },
  CONTENT_MAX_WIDTH: 1920,
  PADDING: {
    MOBILE: 16,
    DESKTOP: 24,
    LARGE_DESKTOP: 32,
  },
  GAP: {
    MOBILE: 12,
    DESKTOP: 20,
    LARGE_DESKTOP: 24,
  },
} as const;

// Animation and transition constants
export const ANIMATIONS = {
  HOVER_TRANSFORM: 'translateY(-2px)',
  BUTTON_HOVER_TRANSFORM: 'translateY(-1px)',
  TRANSITION_DURATION: '0.2s',
  TRANSITION_EASING: 'ease',
  HOVER_SHADOW: '0 6px 16px rgba(0,0,0,0.15)',
  CARD_SHADOW: '0 2px 12px rgba(0,0,0,0.08)',
} as const;

// Desktop-specific styling constants
export const DESKTOP_STYLES = {
  BORDER_RADIUS: {
    SMALL: 8,
    MEDIUM: 12,
    LARGE: 16,
  },
  SCROLLBAR: {
    WIDTH: 8,
    TRACK_COLOR: '#f1f1f1',
    THUMB_COLOR: '#c1c1c1',
    THUMB_HOVER_COLOR: '#a1a1a1',
  },
  TYPOGRAPHY: {
    FONT_FAMILY: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif',
  },
} as const;