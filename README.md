# World Population Dashboard — Full Source (React + Vite frontend + Dash backend)

This package contains:
- frontend/ : React + Vite source (port 3000 for dev)
- backend/  : Dash + Flask backend (port 8050), serves API and Dash UI
- data/     : CSV files (World Bank population data)

## Run backend (in VS Code terminal)
cd backend
python -m venv venv
# activate venv
source venv/bin/activate   # macOS / Linux
# or venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py

## Run frontend (in separate terminal)
cd frontend
npm install
npm run dev

Frontend dev server runs on http://localhost:3000 and will call backend API at http://127.0.0.1:8050/api


## Enhancements applied

- Enabled `flask-cors` to avoid CORS issues when running React dev server on port 3000 and backend on 8050.
- Choropleth now intelligently detects whether Country Codes are ISO-3; if not, it uses country names as fallback.
- Regions are bucketed into top N (default 8) and 'Other' to keep stacked area charts readable.
