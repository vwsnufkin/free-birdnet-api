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

# Download verified binary ONNX model & full labels file from Hugging Face mirror
RUN mkdir -p /app/models && \
    curl -fL "https://huggingface.co/birdnet-team/BirdNET-Analyzer/resolve/main/birdnet_analyzer/model/BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx" -o /app/models/BirdNET_Model.onnx && \
    curl -fL "https://huggingface.co/birdnet-team/BirdNET-Analyzer/resolve/main/birdnet_analyzer/model/labels.txt" -o /app/models/labels.txt && \
    test $(wc -c < /app/models/BirdNET_Model.onnx) -gt 100000000

EXPOSE 10000
CMD ["python", "server.py"]
