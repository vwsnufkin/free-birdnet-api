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

# Download TFLite model weights and labels directly from your vwsnufkin/free-birdnet-api Release
RUN mkdir -p /app/models && \
    curl -fL "https://github.com/vwsnufkin/free-birdnet-api/releases/download/v1.0.0/model.tflite" -o /app/models/model.tflite && \
    curl -fL "https://github.com/vwsnufkin/free-birdnet-api/releases/download/v1.0.0/labels.txt" -o /app/models/labels.txt

# Sanity check to confirm model downloaded successfully (>30MB)
RUN test -s /app/models/model.tflite && test $(wc -c < /app/models/model.tflite) -gt 30000000

COPY . .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir bottle "numpy<2.0.0" scipy tflite-runtime

EXPOSE 10000
CMD ["python", "server.py"]
