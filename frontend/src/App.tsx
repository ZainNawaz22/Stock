
import React, { useState } from 'react';
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import { theme } from './theme/theme';
import { Layout } from './components';
import { Dashboard } from './components/dashboard/Dashboard';
import { StockList } from './components/stocks/StockList';

type CurrentView = 'dashboard' | 'stocks';

function App() {
  const [currentView, setCurrentView] = useState<CurrentView>('dashboard');

  const handleNavigateToStocks = () => {
    setCurrentView('stocks');
  };

  const handleNavigateToDashboard = () => {
    setCurrentView('dashboard');
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Layout 
        currentView={currentView}
        onNavigateToDashboard={handleNavigateToDashboard}
        onNavigateToStocks={handleNavigateToStocks}
      >
        {currentView === 'dashboard' && (
          <Dashboard onNavigateToStocks={handleNavigateToStocks} />
        )}
        {currentView === 'stocks' && <StockList onNavigateBack={handleNavigateToDashboard} />}
      </Layout>
    </ThemeProvider>
  );
}

export default App;
