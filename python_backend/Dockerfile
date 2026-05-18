FROM python:3.11-slim

# Install OpenCV system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Pre-download ML models
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

# Copy the actual application
COPY main.py .

# Expose port
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
