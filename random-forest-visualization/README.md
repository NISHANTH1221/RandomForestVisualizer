# Random Forest Visualization

A web application for visualizing Random Forest models, exploring decision trees, and making predictions with interactive visualizations.

## Quick Start Guide

### Prerequisites

- Python 3.8+ 
- Node.js 18+
- npm or yarn

### Starting the Backend (FastAPI)

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the backend server:
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

The backend API will be available at: `http://localhost:8000`

### Starting the Frontend (Next.js)

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

The frontend application will be available at: `http://localhost:3000`

## Running Both Services

For the complete application, you need both services running:

1. **Terminal 1** - Backend:
   ```bash
   cd backend
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Terminal 2** - Frontend:
   ```bash
   cd frontend
   npm run dev
   ```

Then open your browser to `http://localhost:3000` to use the application.

## Features

- **Tree Visualization**: Interactive visualization of individual decision trees
- **Model Analysis**: Explore Random Forest model structure and statistics
- **Prediction Interface**: Make predictions with feature input forms
- **Decision Path Tracking**: Follow decision paths through trees

## API Documentation

Once the backend is running, visit `http://localhost:8000/docs` for interactive API documentation.

## Project Structure

```
random-forest-visualization/
├── backend/          # FastAPI backend
│   ├── app.py       # Main application
│   ├── models/      # ML model services
│   └── data/        # Model files and data
├── frontend/         # Next.js frontend
│   ├── src/
│   │   ├── app/     # Next.js app router pages
│   │   └── components/ # React components
│   └── public/      # Static assets
└── README.md        # This file
