FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: Copy ALL repository files FIRST (so requirements.txt and server.py exist!)
COPY . .

# Step 2: Upgrade pip and install requirements
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Environment variables for RAM safety
ENV OMP_NUM_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1

EXPOSE 10000
CMD ["python", "server.py"]
