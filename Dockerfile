FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create static models directory
RUN mkdir -p /app/models

# Download official BirdNET V2.4 FP32 ONNX model & labels directly into /app/models
RUN python3 -c "import urllib.request; \
    print('Downloading BirdNET V2.4 ONNX weights...', flush=True); \
    urllib.request.urlretrieve('https://github.com/kahst/BirdNET-Analyzer/raw/main/birdnet_analyzer/model/BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx', '/app/models/model.onnx'); \
    urllib.request.urlretrieve('https://github.com/kahst/BirdNET-Analyzer/raw/main/birdnet_analyzer/model/labels.txt', '/app/models/labels.txt'); \
    print('Model download complete!', flush=True)"

COPY . .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir bottle onnxruntime "numpy<2.0.0" scipy

EXPOSE 10000
CMD ["python", "server.py"]
