# ML-Algorithm-Explorer

A comprehensive machine learning platform featuring 11 supervised learning algorithms with pre-trained models, an interactive web interface, and a robust REST API backend.

## 🎯 Overview

**MLVerse** is an educational and practical platform that demonstrates the application of various machine learning algorithms across different real-world datasets. Each algorithm is pre-trained and ready for predictions through an intuitive web interface or REST API.

### Featured Algorithms

| Algorithm | Model | Dataset | Target |
|-----------|-------|---------|--------|
| AdaBoost | Insurance Charges Predictor | Medical Insurance Costs | Medical Insurance Charges ($) |
| CatBoost | House Price Estimator | Ames Housing (80+ features) | House Sale Price ($) |
| XGBoost | Podcast Engagement Predictor | Podcast Episodes | Listening Time (minutes) |
| Bayesian Ridge | Energy Efficiency Predictor | ENB2012 Dataset | Heating Load (kWh) |
| ElasticNet | Bike Rental Forecaster | UCI Bike Sharing | Total Rentals (count) |
| Lasso | Video Game Sales Predictor | VGSales Dataset | Global Sales (million units) |
| Linear Regression | House Price Calculator | Simple Housing Data | House Price ($) |
| Polynomial | Manufacturing Quality Predictor | Manufacturing Process | Quality Rating Score |
| Random Forest | Used Car Price Estimator | CarDekho Prices | Selling Price (₹ Lakhs) |
| Ridge | Cryptocurrency Price Predictor | Crypto-to-USD Prices | Closing Price (USD) |
| Support Vector | Stock Price Forecaster | Google Stock (2020-2025) | Adjusted Closing Price (USD) |

## 📋 Project Structure

```
ML-Algorithm-Explorer/
├── backend/                    # FastAPI backend server
│   ├── api/                   # API routes and models
│   ├── models/                # ML model management
│   ├── utils/                 # Utilities and helpers
│   ├── tests/                 # Unit tests
│   ├── config.py              # Configuration management
│   ├── main.py                # FastAPI application
│   └── requirements.txt        # Python dependencies
│
├── frontend/                   # React frontend application
│   ├── src/
│   │   ├── pages/             # Page components
│   │   ├── components/        # Reusable components
│   │   ├── services/          # API client services
│   │   ├── hooks/             # Custom React hooks
│   │   └── utils/             # Utility functions
│   ├── package.json           # Node dependencies
│   └── vite.config.js         # Vite configuration
│
├── ml_models/                 # Pre-trained ML models
│   └── Supervised Learning/   # Model files (.pkl)
│
├── docs/                      # Documentation
│   ├── API.md                # API documentation
│   ├── SETUP.md              # Setup instructions
│   └── ARCHITECTURE.md       # Architecture overview
│
└── scripts/                   # Utility scripts
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/mani-191-21/ML-Algorithm-Explorer.git
   cd ML-Algorithm-Explorer
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r backend/requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env if needed
   ```

5. **Start backend server**
   ```bash
   python -m backend.main
   ```
   
   The API will be available at `http://localhost:8000`
   - API Documentation: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```
   
   The frontend will be available at `http://localhost:5173`

4. **Build for production**
   ```bash
   npm run build
   ```

## 📚 API Documentation

### Endpoints

#### Health Check
```
GET /api/health
```
Returns API status and model information.

#### List All Models
```
GET /api/models
```
Returns information about all available models.

#### Get Model Details
```
GET /api/models/{model_id}
```
Returns detailed information about a specific model.

#### Make Prediction
```
POST /api/predict
Content-Type: application/json

{
  "model": "linear",
  "data": {
    "feature1": 10.5,
    "feature2": "category_a"
  }
}
```

Returns prediction result.

#### Get Chart Data
```
GET /api/chart-data/{model_id}
```
Returns pre-computed chart data for visualization.

### Response Examples

**Prediction Response:**
```json
{
  "model": "linear",
  "prediction": 150000.5
}
```

**Models Response:**
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
      "loaded": true,
      "features": {
        "feature1": {
          "label": "Feature 1",
          "type": "number",
          "default": 0
        }
      }
    }
  }
}
```

## 🏗️ Architecture

### Backend Architecture

The backend follows a modular, layered architecture:

- **API Layer** (`api/`): FastAPI routes and request/response models
- **Model Layer** (`models/`): ML model management, loading, and prediction
- **Utility Layer** (`utils/`): Logging, error handling, validation
- **Configuration**: Centralized settings management

### Frontend Architecture

The frontend uses a component-based architecture with:

- **Pages**: Top-level page components
- **Components**: Reusable UI components organized by feature
- **Services**: API client and data fetching
- **Hooks**: Custom React hooks for logic reuse
- **Utils**: Helper functions and constants

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=False

# CORS
CORS_ORIGINS=*

# Logging
LOG_LEVEL=INFO

# Models
MODEL_LOAD_TIMEOUT=30
```

## 📦 Dependencies

### Backend
- **FastAPI**: Modern web framework
- **Uvicorn**: ASGI server
- **scikit-learn**: Machine learning library
- **CatBoost**: Gradient boosting
- **XGBoost**: Extreme gradient boosting
- **Pandas**: Data manipulation
- **Pydantic**: Data validation

### Frontend
- **React**: UI framework
- **Vite**: Build tool
- **React Router**: Client-side routing
- **Axios**: HTTP client
- **Recharts**: Charting library
- **Lucide React**: Icon library

## 🧪 Testing

### Backend Tests

```bash
cd backend
pytest tests/
```

### Frontend Tests

```bash
cd frontend
npm test
```

## 📖 Documentation

See the `docs/` directory for detailed documentation:

- **API.md**: Complete API reference
- **SETUP.md**: Detailed setup instructions
- **ARCHITECTURE.md**: System architecture overview
- **MODELS.md**: Algorithm descriptions and model details

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Manikanta Viswanadhapalli**
- GitHub: [@mani-191-21](https://github.com/mani-191-21)
- Email: manikantaviswanadhapalli123@gmail.com

## 🙏 Acknowledgments

- Machine learning algorithms from scikit-learn, CatBoost, and XGBoost
- Datasets from UCI Machine Learning Repository and Kaggle
- Frontend framework: React and Vite

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub or contact the author.

---

**Last Updated**: December 2024
**Version**: 1.0.0
