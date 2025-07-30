
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import { theme } from './theme/theme';
import { Layout } from './components';
import { Dashboard } from './components/dashboard/Dashboard';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Layout>
        <Dashboard />
      </Layout>
    </ThemeProvider>
  );
}

export default App;
