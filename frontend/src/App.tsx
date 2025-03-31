import { ThemeProvider, CssBaseline, createTheme } from '@mui/material';
// import theme from './theme'; // Keep this commented for now, unless you have a custom theme
import Layout from './components/Layout'; // Import the main layout
import './App.css'; // Keep or remove default CSS as needed

// Create a default theme instance
const defaultTheme = createTheme();

function App() {
  // Remove useState and default Vite/React content
  // const [count, setCount] = useState(0)

  return (
    // Pass the default theme to the provider
    <ThemeProvider theme={defaultTheme}> 
      <CssBaseline /> {/* Normalizes CSS across browsers */}
      <Layout />      {/* Render the main application layout */}
    </ThemeProvider>
    
    /* Remove default Vite/React elements
    <>
      <div>
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>Vite + React</h1>
      <div className="card">
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
        <p>
          Edit <code>src/App.tsx</code> and save to test HMR
        </p>
      </div>
      <p className="read-the-docs">
        Click on the Vite and React logos to learn more
      </p>
    </>
    */
  );
}

export default App;
