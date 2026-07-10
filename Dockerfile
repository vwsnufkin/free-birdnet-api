# Use a lightweight Python environment
FROM python:3.11-slim

# Install system audio libraries required to read sound waves
RUN apt-get update && apt-get install -y git ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

# Download the Official Cornell BirdNET-Analyzer brain
RUN git clone https://github.com/birdnet-team/BirdNET-Analyzer.git /app
WORKDIR /app

# Install lightweight AI dependencies so it easily fits on Render's Free Tier
RUN pip install --no-cache-dir tflite-runtime librosa bottle resampy scipy

# Open the web port
EXPOSE 8080

# Start the API server to listen for file uploads from your website
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8080"]
