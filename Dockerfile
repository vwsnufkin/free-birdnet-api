FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy repository files
COPY . .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Environment variables for memory management
ENV OMP_NUM_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1

EXPOSE 10000
CMD ["python", "server.py"]
