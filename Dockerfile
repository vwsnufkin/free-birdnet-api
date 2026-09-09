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
RUN pip install --no-cache-dir --no-deps -r requirements.txt
RUN pip install --no-cache-dir bottle onnxruntime "numpy<2.0.0" scipy

# Download direct binary model weights & labels into /app/models/
RUN mkdir -p /app/models && \
    curl -sL "https://github.com/birdnet-team/BirdNET-Analyzer/releases/download/v2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx" -o /app/models/BirdNET_Model.onnx && \
    curl -sL "https://raw.githubusercontent.com/birdnet-team/BirdNET-Analyzer/main/birdnet_analyzer/model/labels.txt" -o /app/models/labels.txt && \
    test -s /app/models/BirdNET_Model.onnx

EXPOSE 10000
CMD ["python", "server.py"]
