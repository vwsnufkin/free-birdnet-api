FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

# Install git, ffmpeg, and ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone only the main branch (shallow clone) to get the latest model weights & labels
RUN git clone --depth 1 https://github.com/birdnet-team/BirdNET-Analyzer.git /tmp/birdnet-repo && \
    mkdir -p /app/models && \
    cp /tmp/birdnet-repo/birdnet_analyzer/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx /app/models/model.onnx && \
    cp /tmp/birdnet-repo/birdnet_analyzer/labels/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels.txt /app/models/labels.txt && \
    rm -rf /tmp/birdnet-repo

COPY . .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir bottle onnxruntime "numpy<2.0.0" scipy

EXPOSE 10000
CMD ["python", "server.py"]
