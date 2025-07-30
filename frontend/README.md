# PSX AI Advisor Frontend

A React TypeScript application for the Pakistan Stock Exchange AI Advisor system.

## Features

- **Modern React Setup**: Built with Vite and TypeScript for fast development
- **Material-UI Design**: Responsive UI components with Material Design
- **API Integration**: Axios-based API client with interceptors and error handling
- **Custom Hooks**: Reusable hooks for API calls and state management
- **Type Safety**: Full TypeScript support with comprehensive type definitions

## Project Structure

```
src/
├── components/          # React components
│   ├── common/         # Reusable UI components
│   ├── layout/         # Layout components
│   └── dashboard/      # Dashboard-specific components
├── hooks/              # Custom React hooks
├── services/           # API services and HTTP client
├── types/              # TypeScript type definitions
├── utils/              # Utility functions and constants
└── theme/              # Material-UI theme configuration
```

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
```

2. Copy environment variables:
```bash
cp .env.example .env
```

3. Update environment variables in `.env`:
```
VITE_API_BASE_URL=http://localhost:8000
VITE_API_TIMEOUT=30000
VITE_REFRESH_INTERVAL=60000
```

### Development

Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:5173`

### Building

Build for production:
```bash
npm run build
```

Preview production build:
```bash
npm run preview
```

## API Integration

The frontend is configured to communicate with the PSX AI Advisor API server. Key features:

- **Base URL Configuration**: Configurable API endpoint via environment variables
- **Request/Response Interceptors**: Automatic error handling and logging
- **Type-Safe API Calls**: Full TypeScript support for API responses
- **Error Handling**: Standardized error handling with user-friendly messages

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_BASE_URL` | API server base URL | `http://localhost:8000` |
| `VITE_API_TIMEOUT` | API request timeout (ms) | `30000` |
| `VITE_REFRESH_INTERVAL` | Data refresh interval (ms) | `60000` |

## Technology Stack

- **React 19** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Material-UI v6** - UI component library
- **Axios** - HTTP client
- **Emotion** - CSS-in-JS styling