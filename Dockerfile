FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir bottle onnxruntime "numpy<2.0.0" scipy birdnet-analyzer

# Use birdnet_analyzer's official internal model loader to fetch weights into local cache
RUN python3 -c "from birdnet_analyzer import model, labels; model.load_model(); labels.load_labels()" || true

EXPOSE 10000
CMD ["python", "server.py"]
