# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Pages (Home, Dashboard, ModelDetail)                │  │
│  │  Components (Forms, Charts, Cards, Layout)           │  │
│  │  Services (API Client, Model Service)                │  │
│  │  Hooks (useFetch, usePrediction, useModels)         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↕ HTTP/REST
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  API Layer (Routes, Models, Dependencies)            │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  Model Layer                                   │  │  │
│  │  │  ├─ Registry: Model configurations            │  │  │
│  │  │  ├─ Loader: Load and cache models             │  │  │
│  │  │  └─ Predictor: Make predictions               │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  Utils Layer                                   │  │  │
│  │  │  ├─ Logger: Structured logging                │  │  │
│  │  │  ├─ Exceptions: Custom error handling         │  │  │
│  │  │  └─ Validators: Input validation              │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↕ File I/O
┌─────────────────────────────────────────────────────────────┐
│              ML Models Storage (Pickle Files)               │
│  ├─ AdaBoost, CatBoost, XGBoost, Bayesian, ...            │
│  └─ Each model includes preprocessing pipeline             │
└─────────────────────────────────────────────────────────────┘
```

## Backend Architecture

### 1. API Layer (`backend/api/`)

**Responsibility:** Handle HTTP requests and responses

**Components:**
- `routes.py`: FastAPI route handlers
- `models.py`: Pydantic request/response models
- `dependencies.py`: Dependency injection

**Key Features:**
- RESTful endpoints for model access and predictions
- Automatic request validation via Pydantic
- Swagger/OpenAPI documentation
- Error handling with proper HTTP status codes

### 2. Model Layer (`backend/models/`)

**Responsibility:** ML model management and prediction

**Components:**

#### Registry (`registry.py`)
- Central configuration for all 11 algorithms
- Model metadata (name, algorithm, dataset, target)
- Feature information
- Model file paths

#### Loader (`loader.py`)
- Loads models from pickle files on startup
- Caches models in memory for fast access
- Tracks loading status
- Handles loading errors gracefully

#### Predictor (`predictor.py`)
- Makes predictions using loaded models
- Handles feature preprocessing (scaling, encoding)
- Supports both dictionary-based and raw models
- Returns formatted predictions

**Data Flow:**
```
Input Data → Validation → Feature Preprocessing → Model Prediction → Output
```

### 3. Utils Layer (`backend/utils/`)

**Responsibility:** Cross-cutting concerns

**Components:**

#### Logger (`logger.py`)
- Centralized logging configuration
- Consistent log formatting
- Support for different log levels

#### Exceptions (`exceptions.py`)
- Custom exception hierarchy
- HTTP status code mapping
- Informative error messages

#### Validators (`validators.py`)
- Input validation logic
- Feature validation
- Model name validation

### 4. Configuration (`config.py`)

**Responsibility:** Centralized settings management

**Features:**
- Environment variable loading
- Path management
- Feature flags
- Logging configuration
- CORS settings

## Frontend Architecture

### 1. Pages Layer

**Components:**
- `Home.jsx`: Landing page
- `Dashboard.jsx`: Model overview
- `ModelDetail.jsx`: Individual model interface
- `NotFound.jsx`: 404 page

**Responsibility:** Top-level page routing and layout

### 2. Components Layer

**Organization by Feature:**

#### Layout Components
- `Header.jsx`: Navigation bar
- `Footer.jsx`: Footer
- `Sidebar.jsx`: Side navigation

#### Form Components
- `PredictionForm.jsx`: Input form for predictions

#### Visualization Components
- `ChartDisplay.jsx`: Chart rendering
- `PerformanceMetrics.jsx`: Model metrics
- `ModelComparison.jsx`: Compare multiple models

#### Card Components
- `ModelCard.jsx`: Model summary card
- `ResultCard.jsx`: Prediction result card

#### Common Components
- `Loading.jsx`: Loading spinner
- `Error.jsx`: Error message display
- `Button.jsx`: Reusable button

**Responsibility:** Reusable UI components

### 3. Services Layer (`services/`)

**Components:**
- `api.js`: Axios HTTP client configuration
- `modelService.js`: Model-related API calls

**Responsibility:** Centralized API communication

### 4. Hooks Layer (`hooks/`)

**Custom Hooks:**
- `useFetch.js`: Generic data fetching
- `usePrediction.js`: Prediction logic
- `useModels.js`: Model list management

**Responsibility:** Logic reuse and state management

### 5. Utils Layer (`utils/`)

**Components:**
- `constants.js`: Application constants
- `formatters.js`: Data formatting utilities
- `validators.js`: Frontend validation

**Responsibility:** Helper functions and constants

## Data Flow

### Prediction Flow

```
1. User Input
   ↓
2. Frontend Validation
   ↓
3. API Request (POST /predict)
   ↓
4. Backend Validation
   ↓
5. Feature Preprocessing
   ↓
6. Model Prediction
   ↓
7. Result Formatting
   ↓
8. API Response
   ↓
9. Frontend Display
```

### Model Loading Flow

```
1. Application Startup
   ↓
2. Load Configuration (registry.py)
   ↓
3. Initialize ModelLoader
   ↓
4. For Each Model:
   - Locate pickle file
   - Load with joblib
   - Cache in memory
   - Log status
   ↓
5. Ready for Predictions
```

## Technology Stack

### Backend
- **Framework:** FastAPI
- **Server:** Uvicorn
- **ML Libraries:** scikit-learn, CatBoost, XGBoost
- **Data:** Pandas, NumPy
- **Validation:** Pydantic
- **Serialization:** Joblib

### Frontend
- **Framework:** React 19
- **Build Tool:** Vite
- **Routing:** React Router v7
- **HTTP:** Axios
- **Charts:** Recharts
- **Icons:** Lucide React
- **Styling:** CSS3

## Scalability Considerations

### Current Implementation
- Single-process backend
- In-memory model caching
- Synchronous predictions

### Future Improvements
- Model versioning
- A/B testing support
- Batch prediction API
- Model monitoring and metrics
- Async prediction processing
- Load balancing for multiple instances
- Model serving with TensorFlow Serving or similar

## Security Considerations

### Current Implementation
- CORS enabled for all origins (configurable)
- Input validation via Pydantic
- Error messages don't expose sensitive info

### Recommendations
- Add authentication (JWT, OAuth2)
- Implement rate limiting
- Add request logging
- Use HTTPS in production
- Validate file uploads
- Implement API versioning

## Performance Optimization

### Backend
- Models loaded once on startup
- In-memory caching
- Efficient feature preprocessing
- Minimal data copying

### Frontend
- Code splitting via Vite
- Lazy loading of routes
- Component memoization
- Efficient state management

## Monitoring and Logging

### Backend Logging
- Structured logging with timestamps
- Log levels: DEBUG, INFO, WARNING, ERROR
- Configurable via environment variables

### Future Monitoring
- Model prediction metrics
- API response times
- Error rates
- Model accuracy tracking
- System resource usage

## Deployment Architecture

### Development
```
Frontend (npm run dev) → Backend (python -m backend.main)
```

### Production
```
Frontend (npm run build) → Static files
Backend (uvicorn with gunicorn) → Docker container
Models → Persistent storage
```

## Error Handling Strategy

### Levels
1. **Input Validation:** Pydantic models
2. **Business Logic:** Custom exceptions
3. **HTTP Response:** Proper status codes
4. **Logging:** All errors logged

### Error Types
- `ModelNotFoundError` (404)
- `PredictionError` (500)
- `ValidationError` (400)
- `ModelLoadError` (500)

## Testing Strategy

### Backend
- Unit tests for models, utils, API
- Integration tests for API endpoints
- Mock model loading for tests

### Frontend
- Component tests
- Integration tests
- E2E tests for critical flows
