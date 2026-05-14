#!/bin/bash
set -e

echo "=== LeafGuard AI — Starting Services ==="

# 1. Start the Python FastAPI ML backend on port 8000 (internal)
echo "[1/2] Starting Python ML backend on port 8000..."
cd /app/python_backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 &
PYTHON_PID=$!

# 2. Wait for Python backend to be ready
echo "[1/2] Waiting for Python backend to be healthy..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo "[1/2] Python backend is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "[1/2] WARNING: Python backend did not respond in 30s, starting Node anyway..."
    fi
    sleep 1
done

# 3. Start the Node.js Express server on port 7860 (public)
echo "[2/2] Starting Node.js server on port 7860..."
cd /app
exec node dist/server.cjs
