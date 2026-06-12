# Deployment Guide for ML-Algorithm-Explorer

## Overview

This guide covers deploying ML-Algorithm-Explorer to Vercel, a serverless platform perfect for full-stack applications.

## Prerequisites

- GitHub account with the repository
- Vercel account (free tier available)
- Node.js and npm installed locally

## Vercel Deployment Steps

### 1. Connect Repository to Vercel

1. Go to [vercel.com](https://vercel.com)
2. Click "New Project"
3. Import your GitHub repository (mani-191-21/ML-Algorithm-Explorer)
4. Select the repository and click "Import"

### 2. Configure Project Settings

**Framework Preset:** Other
**Root Directory:** ./
**Build Command:** `cd frontend && npm run build`
**Output Directory:** `frontend/dist`
**Install Command:** `npm install && pip install -r requirements.txt`

### 3. Environment Variables

Add the following environment variables in Vercel dashboard:

```
ENVIRONMENT=production
DEBUG=false
CORS_ORIGINS=*
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=3000
```

### 4. Deploy

Click "Deploy" and wait for the deployment to complete.

## Project Structure for Vercel

```
ML-Algorithm-Explorer/
├── api/
│   └── index.py              # Serverless API entry point
├── frontend/                 # React frontend
│   ├── src/
│   ├── dist/                 # Build output
│   └── package.json
├── backend/                  # FastAPI backend
│   ├── main.py
│   ├── api/
│   ├── models/
│   └── utils/
├── ml_models/                # Pre-trained models
├── vercel.json               # Vercel configuration
├── requirements.txt          # Python dependencies
└── README.md
```

## How It Works

### Frontend
- Built with Vite and React
- Deployed as static files to Vercel CDN
- Automatically served from `frontend/dist`

### Backend
- FastAPI application
- Runs as serverless function on Vercel
- Handles all API requests to `/api/*`

### Models
- Pre-trained pickle files included in repository
- Loaded on first request (cold start)
- Cached in memory for subsequent requests

## API Endpoints

After deployment, your API will be available at:
```
https://your-project-name.vercel.app/api/
```

### Available Endpoints

- `GET /api/health` - Health check
- `GET /api/models` - List all models
- `GET /api/models/{model_id}` - Get model details
- `GET /api/categories` - List model categories
- `GET /api/category/{category_name}` - Models in category
- `POST /api/predict` - Make prediction
- `GET /api/chart-data/{model_id}` - Chart data

## Frontend URL

After deployment, your frontend will be available at:
```
https://your-project-name.vercel.app/
```

## Monitoring

### Vercel Dashboard
- Monitor deployments
- View logs
- Check performance metrics
- Manage environment variables

### API Logs
View logs in Vercel dashboard under "Functions" tab.

## Performance Optimization

### Cold Start Optimization
- Models are lazy-loaded on first request
- Consider pre-warming by calling `/api/health` after deployment

### Caching
- Frontend assets cached by CDN
- API responses can be cached by frontend

### Model Loading
- Models loaded once on startup
- Kept in memory for fast predictions

## Troubleshooting

### Models Not Loading
1. Check if model files are in repository
2. Verify paths in `backend/models/registry.py`
3. Check Vercel function logs

### CORS Issues
1. Verify `CORS_ORIGINS` environment variable
2. Check browser console for specific error
3. Update `vercel.json` if needed

### Build Failures
1. Check build logs in Vercel dashboard
2. Ensure `frontend/package.json` is correct
3. Verify Python dependencies in `requirements.txt`

## Custom Domain

To add a custom domain:
1. Go to project settings in Vercel
2. Click "Domains"
3. Add your custom domain
4. Follow DNS configuration instructions

## Continuous Deployment

- Automatic deployments on push to `main` branch
- Preview deployments for pull requests
- Rollback to previous versions if needed

## Production Checklist

- [ ] Environment variables configured
- [ ] Custom domain set up (optional)
- [ ] CORS settings verified
- [ ] Models loading correctly
- [ ] API endpoints responding
- [ ] Frontend loading properly
- [ ] Performance acceptable
- [ ] Error handling working

## Advanced Configuration

### Increasing Function Memory

Edit `vercel.json`:
```json
"functions": {
  "api/index.py": {
    "memory": 3008,
    "maxDuration": 60
  }
}
```

### Custom Build Scripts

Edit `vercel.json`:
```json
"buildCommand": "npm run build:all"
```

## Support

For issues:
1. Check Vercel documentation
2. Review project logs
3. Open GitHub issue
4. Contact support

## Additional Resources

- [Vercel Documentation](https://vercel.com/docs)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [React Deployment](https://react.dev/learn/start-a-new-react-project)

---

**Deployment Date:** December 2024
**Version:** 1.0.0
