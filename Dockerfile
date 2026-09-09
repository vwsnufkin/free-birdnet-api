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

# Force-download the ONNX model and labels file directly into /app/models during build
RUN mkdir -p /app/models && python3 -c "\
import os, glob, shutil; \
import birdnet_analyzer.config as cfg; \
import birdnet_analyzer.model as bmodel; \
import birdnet_analyzer.labels as blabels; \
cfg.MODEL_TYPE = 'onnx'; \
bmodel.load_model(); \
blabels.load_labels(); \
onnx_files = glob.glob('/root/.cache/**/*.onnx', recursive=True) + glob.glob('/usr/local/lib/python3.11/site-packages/**/*.onnx', recursive=True); \
label_files = glob.glob('/root/.cache/**/*label*.txt', recursive=True) + glob.glob('/usr/local/lib/python3.11/site-packages/**/*label*.txt', recursive=True); \
if onnx_files: shutil.copy(onnx_files[0], '/app/models/model.onnx'); \
if label_files: shutil.copy(label_files[0], '/app/models/labels.txt'); \
print('Baking complete. Models in /app/models:', os.listdir('/app/models'), flush=True)" || true

EXPOSE 10000
CMD ["python", "server.py"]
