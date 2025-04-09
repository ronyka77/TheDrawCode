# Plan: Web UI for Soccer Prediction Project

**Version:** 2.0
**Date:** $(Get-Date -Format "yyyy-MM-dd")

## 1. Goal Relaunch

Develop a web-based user interface for the Soccer Prediction Project. The core functionality remains the same as the previous UI plan:
- Configure model training parameters.
- Initiate and monitor the ensemble model training process.
- View training results and logs in real-time.
- Input data to make predictions using the trained model.
- View prediction outputs.
- **New Emphasis:** Modern design, responsive interface, eye-catching visuals, and icons.

## 2. Proposed Technology Stack & Architecture

We'll adopt a standard modern web architecture:

- **Backend:** Python with **FastAPI**.
    - **Why:** High performance, asynchronous support (good for WebSockets), automatic interactive API documentation (Swagger UI), uses Pydantic for data validation. It's a modern and efficient choice for building APIs in Python.
- **Frontend:** **React** (with **TypeScript**).
    - **Why:** Highly popular, component-based, large ecosystem, good for building interactive SPAs (Single Page Applications). TypeScript adds type safety, improving maintainability.
- **UI Components & Styling:** **Material UI (MUI)**.
    - **Why:** Provide pre-built, customizable, modern React components (buttons, inputs, cards, tabs, icons, etc.) that follow Material Design principles. This accelerates development and ensures a polished look. Styling can be done via MUI's built-in systems (like Emotion).
- **Real-time Communication:** **WebSockets**.
    - **Why:** Essential for pushing logs and status updates from the backend (training progress) to the frontend without constant polling. FastAPI has excellent WebSocket support.
- **State Management (Frontend):** React Context API / `useState` / `useReducer` initially. Consider **Zustand** if state complexity grows.
    - **Why:** Start simple with built-in React tools, escalate to a lightweight library like Zustand if needed, avoiding the boilerplate of older solutions like Redux unless necessary.
- **Background Tasks (Training):** Initially, run directly via FastAPI's async capabilities. **Optional Enhancement:** Use **Celery** with **Redis** or **RabbitMQ** for robust background task management.
- **API Design:** RESTful API for standard actions (config, start training, get results, predict) and WebSockets for streaming updates.

**Architecture Diagram (Conceptual):**

```mermaid
graph LR
    User[User Browser] -- HTTP/WebSocket --> Frontend[React SPA (MUI Components)]
    Frontend -- REST API / WebSocket --> Backend[FastAPI Server]
    Backend -- Calls --> PythonLogic[Existing Python Logic (run_ensemble, predict, etc.)]
    PythonLogic -- Logs/Status --> Backend
    Backend -- WebSocket Push --> Frontend
    User -- Interacts --> Frontend

    subgraph Optional Background Tasks
        Backend -- Task Queue --> Broker[(Redis/RabbitMQ)]
        Broker -- Task --> Worker[Celery Worker]
        Worker -- Runs --> PythonLogic
        Worker -- Results/Logs/Status --> Broker
        Broker -- Updates --> Backend
    end
```

## 3. Core Components & Features (Web Implementation)

- **Layout:** A responsive layout (e.g., using MUI's `Grid` or `Stack` components) possibly featuring a persistent sidebar for configuration/navigation and a main content area with tabs (MUI `Tabs`) for Logs, Results, and Prediction. A `Snackbar` component for status updates or a dedicated status bar footer.
- **Configuration (`ConfigurationPanel`):** A React component using MUI form controls (`TextField`, `Select`, `Slider`, `Checkbox`, `FormControlLabel`) bound to the application state. Changes trigger API calls to update the backend config.
- **Controls (`ControlPanel`):** React component with styled MUI `Button`s, incorporating MUI Icons (`@mui/icons-material`). Button clicks trigger API calls to start training or prediction. Buttons change appearance (disabled state, loading indicator via `LoadingButton`) based on application status.
- **Logging (`LogDisplay`):** A React component establishing a WebSocket connection upon training start. It receives log messages pushed from the backend and appends them to a scrollable area (perhaps using a virtualized list for performance with many logs). Timestamps and log levels can be styled differently.
- **Results (`ResultsDisplay`):** React component displaying metrics (Precision, Recall, F1, Threshold) fetched via API after training completes. Could use MUI `Card` or `Table` for presentation.
- **Prediction (`PredictionPanel`):** React component with MUI form inputs for prediction features. On submit, calls the prediction API endpoint and displays the returned result.
- **Styling & Theme:** Utilize MUI's theming capabilities to define the color palette (primary, secondary, success, error colors), typography (font families, sizes), and component overrides for a consistent, modern look.

## 4. Development Plan (Phased Approach)

- **Phase 0: Setup & Backend Foundation**
    - **[ ] Task 0.1:** Set up project structure (`/frontend`, `/backend` - note: backend code might now live under `src/backend`).
    - **[ ] Task 0.2:** Initialize FastAPI backend: Basic app, manage dependencies with `uv` and `pyproject.toml`.
    - **[ ] Task 0.3:** Define Pydantic models for configuration, training requests, results, predictions.
    - **[ ] Task 0.4:** Implement basic API endpoints (GET/PUT config, placeholder POST for training/predict).
    - **[ ] Task 0.5:** Set up basic WebSocket endpoint in FastAPI.
    - **[ ] Task 0.6:** Initialize React frontend: Use Vite with React+TS template (`npm create vite@latest frontend -- --template react-ts`).
    - **[ ] Task 0.7:** Install frontend dependencies (`npm install @mui/material @emotion/react @emotion/styled @mui/icons-material`).
- **Phase 1: Frontend Layout & Configuration**
    - **[ ] Task 1.1:** Set up MUI Theme provider.
    - **[ ] Task 1.2:** Implement main layout (`App`, `Layout` components).
    - **[ ] Task 1.3:** Implement `ConfigurationPanel` component with MUI controls.
    - **[ ] Task 1.4:** Implement state management for configuration.
    - **[ ] Task 1.5:** Connect configuration UI to backend API (fetch initial config, update config).
- **Phase 2: Training Initiation & Real-time Logs**
    - **[ ] Task 2.1:** Implement `ControlPanel` with styled buttons and icons.
    - **[ ] Task 2.2:** Implement `LogDisplay` component.
    - **[ ] Task 2.3:** Connect "Start Training" button to the backend API. Backend starts the training (initially directly, later maybe via Celery) and returns a job identifier.
    - **[ ] Task 2.4:** Frontend uses the job ID to connect to the specific WebSocket endpoint.
    - **[ ] Task 2.5:** Backend: Modify training logic (`run_ensemble` wrapper) to push log messages via the WebSocket connection associated with the job ID.
    - **[ ] Task 2.6:** Frontend: Display received logs in `LogDisplay`.
    - **[ ] Task 2.7:** Implement button disabling/loading states during training.
- **Phase 3: Results & Status Updates**
    - **[ ] Task 3.1:** Backend: Store results upon training completion. Implement API endpoint to fetch results by job ID. Modify WebSocket to push status updates (e.g., "Training", "Finished", "Error").
    - **[ ] Task 3.2:** Frontend: Implement `ResultsDisplay` component. Fetch and display results when training finishes.
    - **[ ] Task 3.3:** Frontend: Implement `StatusBar` or `Snackbar` to display status updates received via WebSocket.
- **Phase 4: Prediction**
    - **[ ] Task 4.1:** Implement `PredictionPanel` with input fields.
    - **[ ] Task 4.2:** Backend: Implement prediction API endpoint, loading the necessary model (consider how the model is persisted/accessed after training).
    - **[ ] Task 4.3:** Frontend: Connect "Predict" button and form to the prediction API. Display results.
- **Phase 5: Styling Refinement & Polish**
    - **[ ] Task 5.1:** Fine-tune MUI theme (colors, typography, component variants).
    - **[ ] Task 5.2:** Ensure responsiveness across different screen sizes.
    - **[ ] Task 5.3:** Add loading indicators and better error handling/display.
    - **[ ] Task 5.4:** Improve visual feedback and user experience details.
    - **[ ] Task 5.5:** (Optional) Integrate Celery for background tasks if direct async handling proves insufficient.

## 5. Potential Challenges

- **Real-time Sync:** Ensuring smooth and reliable real-time updates via WebSockets.
- **State Management Complexity:** Managing state across different frontend components as the application grows.
- **Background Task Robustness:** Handling errors and ensuring scalability if using direct async calls vs. a dedicated task queue like Celery.
- **Model Persistence/Loading:** Defining a strategy for how the trained model is saved by the backend and loaded for predictions.
- **Feature Input for Prediction:** Designing a user-friendly web form for potentially numerous features.

## 6. Future Enhancements (Optional)

- User Authentication.
- Persistent storage of configurations and results.
- More advanced visualizations (e.g., using Chart.js or Nivo).
- Comparison of different training runs.
- Deployment strategy (e.g., Docker containers).