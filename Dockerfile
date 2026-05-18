# ==============================================================
# LeafGuard AI — Hugging Face Spaces Dockerfile
# Runs both the Node.js Express server (React SPA + API proxy)
# and the Python FastAPI ML backend in a single container.
# ==============================================================

# ---- Stage 1: Build the React frontend ----
FROM node:20-slim AS frontend-build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts

COPY index.html tsconfig.json vite.config.ts ./
COPY src/ src/

# Build the Vite React app into dist/
RUN npx vite build

# ---- Stage 2: Build the Node.js server bundle ----
FROM node:20-slim AS server-build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts

COPY server.ts tsconfig.json ./
RUN npx esbuild server.ts --bundle --platform=node --format=cjs --packages=external --sourcemap --outfile=dist/server.cjs

# ---- Stage 3: Runtime (Python + Node.js) ----
FROM python:3.11-slim

# Install Node.js 20 + utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl sed libgl1-mesa-glx libglib2.0-0 && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python dependencies (CPU-only PyTorch for smaller image) ----
COPY python_backend/requirements.txt /app/python_backend/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /app/python_backend/requirements.txt

# ---- Pre-download ML models to avoid cold-start ----
RUN python -c "\
from transformers import CLIPProcessor, CLIPModel; \
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
print('CLIP model cached'); \
"

RUN python -c "\
from transformers import pipeline; \
p = pipeline('image-classification', model='linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification'); \
print('PlantVillage classifier cached'); \
"

RUN python -c "\
from ultralytics import YOLO; \
model = YOLO('yolov8s-seg.pt'); \
print('YOLOv8s-seg model cached'); \
"

# ---- Node.js production dependencies ----
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts

# ---- Copy built artifacts ----
# React frontend build
COPY --from=frontend-build /app/dist ./dist

# Node.js server bundle
COPY --from=server-build /app/dist/server.cjs ./dist/server.cjs
COPY --from=server-build /app/dist/server.cjs.map ./dist/server.cjs.map

# Python backend
COPY python_backend/main.py /app/python_backend/main.py

# Startup script — strip Windows CRLF line endings then make executable
COPY start.sh /app/start.sh
RUN sed -i 's/\r//' /app/start.sh && chmod +x /app/start.sh

# ---- Environment ----
ENV NODE_ENV=production
ENV PORT=7860
ENV RENDER_VISION_API_URL=http://localhost:8000

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["/app/start.sh"]
