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
RUN pip install --no-cache-dir bottle onnxruntime "numpy<2.0.0" scipy huggingface_hub

# Download official model weights and labels using Python's verified downloader
RUN mkdir -p /app/models && python3 -c "\
import urllib.request, os; \
from huggingface_hub import hf_hub_download; \
print('Downloading BirdNET ONNX model...', flush=True); \
model_path = hf_hub_download(repo_id='birdnet-team/BirdNET-Analyzer', filename='birdnet_analyzer/model/BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx'); \
os.system(f'cp {model_path} /app/models/BirdNET_Model.onnx'); \
label_path = hf_hub_download(repo_id='birdnet-team/BirdNET-Analyzer', filename='birdnet_analyzer/model/labels.txt'); \
os.system(f'cp {label_path} /app/models/labels.txt'); \
print('Downloaded model size:', os.path.getsize('/app/models/BirdNET_Model.onnx'), 'bytes', flush=True); \
" && test $(wc -c < /app/models/BirdNET_Model.onnx) -gt 100000000

EXPOSE 10000
CMD ["python", "server.py"]
