FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /app/models

# Download official BirdNET V2.4 FP32 ONNX model & labels from the correct repository path
RUN python3 -c "import urllib.request, os; \
    print('Downloading BirdNET V2.4 ONNX model...', flush=True); \
    urllib.request.urlretrieve('https://github.com/birdnet-team/BirdNET-Analyzer/raw/main/birdnet_analyzer/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx', '/app/models/model.onnx'); \
    print('Downloading species labels...', flush=True); \
    urllib.request.urlretrieve('https://github.com/birdnet-team/BirdNET-Analyzer/raw/main/birdnet_analyzer/labels/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels.txt', '/app/models/labels.txt'); \
    print('Model download size:', os.path.getsize('/app/models/model.onnx'), 'bytes', flush=True)"

COPY . .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir bottle onnxruntime "numpy<2.0.0" scipy

EXPOSE 10000
CMD ["python", "server.py"]
