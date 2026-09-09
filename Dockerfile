FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

ENV OMP_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download the BirdNET ONNX model weights directly during build
RUN mkdir -p /app/models && \
    curl -L -o /app/models/BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx \
    https://github.com/kahst/BirdNET-Analyzer/raw/main/birdnet_analyzer/model/BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx || true

EXPOSE 10000
CMD ["python", "server.py"]
