FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Permanent model cache path inside container
ENV BIRDNET_MODEL_PATH=/app/model_cache
ENV XDG_CACHE_HOME=/app/model_cache
ENV TORCH_HOME=/app/model_cache
ENV HF_HOME=/app/model_cache

# Force ONNX runtime to save memory
ENV BIRDNET_USE_ONNX=1

# Strict single-threading
ENV OMP_NUM_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /app/model_cache

COPY . .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models directly into cache
RUN python -c "import os; os.environ['XDG_CACHE_HOME']='/app/model_cache'; from birdnet_analyzer import model, species; model.load_model(); species.get_species_list(50.85, 4.35, 0.15)" || true

EXPOSE 10000
CMD ["python", "server.py"]
