#!/usr/bin/env bash
set -e
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Start backend
cd "$ROOT_DIR/backend"
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt
# run backend in background
python app.py &
BACK_PID=$!

# Start frontend
cd "$ROOT_DIR/frontend"
if [ ! -d "node_modules" ]; then
  npm install
fi
npm run dev &
FRONT_PID=$!

echo "Backend PID: $BACK_PID | Frontend PID: $FRONT_PID"
wait
