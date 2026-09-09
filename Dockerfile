FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install huggingface_hub to reliably download LFS binary files
RUN pip install --no-cache-dir huggingface_hub

# Download official BirdNET weights & labels via HuggingFace Hub API
RUN mkdir -p /app/models && python3 -c "\
from huggingface_hub import hf_hub_download; \
import shutil, os; \
print('Downloading BirdNET V2.4 ONNX model...', flush=True); \
model_path = hf_hub_download(repo_id='birdnet-team/BirdNET-Analyzer', filename='birdnet_analyzer/model/BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx'); \
shutil.copy(model_path, '/app/models/model.onnx'); \
label_path = hf_hub_download(repo_id='birdnet-team/BirdNET-Analyzer', filename='birdnet_analyzer/model/labels.txt'); \
shutil.copy(label_path, '/app/models/labels.txt'); \
print('Downloaded model size:', os.path.getsize('/app/models/model.onnx'), 'bytes', flush=True)"

# Verify the binary is larger than 100MB
RUN test -s /app/models/model.onnx && test $(wc -c < /app/models/model.onnx) -gt 100000000

COPY . .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir bottle onnxruntime "numpy<2.0.0" scipy

EXPOSE 10000
CMD ["python", "server.py"]
