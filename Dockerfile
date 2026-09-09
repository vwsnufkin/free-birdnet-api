FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone repo and explicitly run git lfs pull to download binary LFS files
RUN git lfs install && \
    git clone https://github.com/birdnet-team/BirdNET-Analyzer.git /tmp/birdnet-repo && \
    cd /tmp/birdnet-repo && git lfs pull && cd /app && \
    mkdir -p /app/models && \
    find /tmp/birdnet-repo -name "*.onnx" -exec cp {} /app/models/model.onnx \; && \
    find /tmp/birdnet-repo -name "*label*.txt" -exec cp {} /app/models/labels.txt \; && \
    rm -rf /tmp/birdnet-repo && \
    test -s /app/models/model.onnx && \
    test $(wc -c < /app/models/model.onnx) -gt 100000000

COPY . .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir bottle onnxruntime "numpy<2.0.0" scipy

EXPOSE 10000
CMD ["python", "server.py"]
