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

# Force birdnet_analyzer to explicitly download the ONNX model and labels during build
RUN python3 -c "from birdnet_analyzer import model, labels; model.load_model(use_onnx=True); labels.load_labels()" || true

EXPOSE 10000
CMD ["python", "server.py"]
