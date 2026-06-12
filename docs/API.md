# API Documentation

## Overview

The ML-Algorithm-Explorer API provides a RESTful interface for accessing 11 pre-trained machine learning models. All endpoints return JSON responses and support CORS for cross-origin requests.

## Base URL

```
http://localhost:8000/api
```

## Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible.

## Response Format

All responses follow a consistent JSON format:

### Success Response
```json
{
  "data": {},
  "status": "success"
}
```

### Error Response
```json
{
  "error": "Error message",
  "status_code": 400,
  "details": "Additional details"
}
```

## Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Description:** Check API health and model status.

**Response:**
```json
{
  "status": "healthy",
  "total_models": 11,
  "loaded_models": 11,
  "version": "1.0.0"
}
```

**Status Codes:**
- `200`: API is healthy

---

### 2. List All Models

**Endpoint:** `GET /models`

**Description:** Retrieve information about all available models.

**Response:**
```json
{
  "total": 11,
  "loaded": 11,
  "models": {
    "linear": {
      "name": "Simple House Price Calculator",
      "algorithm": "Linear Regression",
      "dataset": "House Prices (Simple)",
      "target": "House Price ($)",
      "description": "Basic linear regression model...",
      "loaded": true,
      "features": {
        "square_feet": {
          "label": "Square Feet",
          "type": "number",
          "default": 0
        },
        "location": {
          "label": "Location",
          "type": "select",
          "options": ["urban", "suburban", "rural"],
          "default": "urban"
        }
      }
    }
  }
}
```

**Status Codes:**
- `200`: Success

---

### 3. Get Model Details

**Endpoint:** `GET /models/{model_id}`

**Description:** Retrieve detailed information about a specific model.

**Parameters:**
- `model_id` (path): Model identifier (e.g., `linear`, `xgboost`)

**Response:**
```json
{
  "name": "Simple House Price Calculator",
  "algorithm": "Linear Regression",
  "dataset": "House Prices (Simple)",
  "target": "House Price ($)",
  "description": "Basic linear regression model for house price estimation.",
  "loaded": true,
  "features": {
    "square_feet": {
      "label": "Square Feet",
      "type": "number",
      "default": 0
    }
  }
}
```

**Status Codes:**
- `200`: Success
- `404`: Model not found

---

### 4. Make Prediction

**Endpoint:** `POST /predict`

**Description:** Generate a prediction using the specified model.

**Request Body:**
```json
{
  "model": "linear",
  "data": {
    "square_feet": 2000,
    "location": "urban"
  }
}
```

**Response:**
```json
{
  "model": "linear",
  "prediction": 350000.5
}
```

**Status Codes:**
- `200`: Prediction successful
- `400`: Invalid request
- `404`: Model not found
- `500`: Prediction error

**Error Examples:**

Missing model:
```json
{
  "error": "Model 'invalid' not found or not loaded",
  "status_code": 404
}
```

Prediction error:
```json
{
  "error": "Prediction failed for model 'linear': Invalid input",
  "status_code": 500
}
```

---

### 5. Get Chart Data

**Endpoint:** `GET /chart-data/{model_id}`

**Description:** Retrieve pre-computed chart data for model visualization.

**Parameters:**
- `model_id` (path): Model identifier

**Response:**
```json
{
  "type": "scatter",
  "data": [
    {"x": 1000, "y": 150000},
    {"x": 2000, "y": 300000},
    {"x": 3000, "y": 450000}
  ],
  "labels": ["Square Feet", "Price"]
}
```

**Status Codes:**
- `200`: Success
- `404`: Model not found or no chart data available

---

## Model IDs

Use these identifiers in API requests:

| ID | Model | Algorithm |
|----|-------|-----------|
| `adaboost` | Insurance Charges Predictor | AdaBoost Regressor |
| `catboost` | House Price Estimator | CatBoost |
| `xgboost` | Podcast Engagement Predictor | XGBoost |
| `bayesian` | Energy Efficiency Predictor | Bayesian Ridge |
| `elasticnet` | Bike Rental Forecaster | ElasticNet |
| `lasso` | Video Game Sales Predictor | Lasso |
| `linear` | House Price Calculator | Linear Regression |
| `polynomial` | Manufacturing Quality Predictor | Polynomial |
| `randomforest` | Used Car Price Estimator | Random Forest |
| `ridge` | Cryptocurrency Price Predictor | Ridge |
| `svr` | Stock Price Forecaster | Support Vector |

## Feature Types

### Number Type
```json
{
  "label": "Feature Name",
  "type": "number",
  "default": 0
}
```

Input: Any numeric value

### Select Type
```json
{
  "label": "Feature Name",
  "type": "select",
  "options": ["option1", "option2", "option3"],
  "default": "option1"
}
```

Input: One of the provided options

## Error Handling

The API returns appropriate HTTP status codes:

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (invalid input) |
| 404 | Not Found (model or resource) |
| 500 | Internal Server Error |

## Rate Limiting

Currently, no rate limiting is implemented. This may be added in future versions.

## CORS

The API supports CORS requests from all origins by default. Configure `CORS_ORIGINS` in `.env` to restrict.

## Examples

### Using cURL

**Get all models:**
```bash
curl http://localhost:8000/api/models
```

**Make a prediction:**
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "linear",
    "data": {"square_feet": 2000}
  }'
```

### Using Python (requests)

```python
import requests

# Get models
response = requests.get('http://localhost:8000/api/models')
models = response.json()

# Make prediction
prediction_data = {
    "model": "linear",
    "data": {"square_feet": 2000}
}
response = requests.post(
    'http://localhost:8000/api/predict',
    json=prediction_data
)
result = response.json()
print(f"Prediction: {result['prediction']}")
```

### Using JavaScript (fetch)

```javascript
// Get models
fetch('http://localhost:8000/api/models')
  .then(res => res.json())
  .then(data => console.log(data));

// Make prediction
const predictionData = {
  model: 'linear',
  data: { square_feet: 2000 }
};

fetch('http://localhost:8000/api/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(predictionData)
})
  .then(res => res.json())
  .then(data => console.log(`Prediction: ${data.prediction}`));
```

## Versioning

Current API Version: **1.0.0**

The version is returned in the health check response and can be used for compatibility checks.

## Support

For issues or questions about the API, please refer to the main README or open an issue on GitHub.
