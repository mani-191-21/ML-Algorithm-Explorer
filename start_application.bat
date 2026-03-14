@echo off
echo Starting Supervised Learning Predictor Backend...
cd /d "c:\Users\Vaka Bela Jithendra\OneDrive\Desktop\ML Algorithm Explorer"
start cmd /k ".\venv\Scripts\activate & cd backend & python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

echo Starting MLVerse React Frontend...
cd /d "c:\Users\Vaka Bela Jithendra\OneDrive\Desktop\ML Algorithm Explorer\frontend"
start cmd /k "npm run dev"

echo Both servers are starting up!
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:3000
