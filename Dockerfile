FROM python:3.11-slim

# System libraries needed by OpenCV / DeepFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Pre-warm DeepFace model weights at build time so first request is fast
RUN python -c "\
import numpy as np; \
from deepface import DeepFace; \
img = np.zeros((100,100,3), dtype=np.uint8); \
DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, silent=True)" || true

COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]