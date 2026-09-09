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

# Download official BirdNET V2.4 FP32 ONNX model & labels from HuggingFace CDN
RUN mkdir -p /app/models && \
    curl -fL "https://huggingface.co/mcguirep/birdnet-analyzer/resolve/main/BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx" -o /app/models/model.onnx && \
    curl -fL "https://huggingface.co/mcguirep/birdnet-analyzer/resolve/main/BirdNET_GLOBAL_6K_V2.4_Labels.txt" -o /app/models/labels.txt

# Verify the ONNX model binary downloaded successfully and is larger than 100MB
RUN test -s /app/models/model.onnx && test $(wc -c < /app/models/model.onnx) -gt 100000000

COPY . .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir bottle onnxruntime "numpy<2.0.0" scipy

EXPOSE 10000
CMD ["python", "server.py"]
