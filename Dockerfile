FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Permanent model storage directory inside the container
ENV BIRDNET_MODEL_PATH=/app/birdnet_models

# Memory isolation
ENV OMP_NUM_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy application files
COPY . .

# Upgrade pip and install requirements
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Create directory and download model files into /app/birdnet_models during build phase
RUN mkdir -p /app/birdnet_models && \
    python -c "from birdnet_analyzer import model; model.load_model('/app/birdnet_models')" || true

EXPOSE 10000
CMD ["python", "server.py"]
