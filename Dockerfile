FROM python:3.10-slim

# Install system dependencies like ffmpeg
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement files and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code (including preload_models.py and server.py)
COPY . .

# 💡 ADD THIS LINE: Downloads AI models directly into the Docker image layers at build time
RUN python preload_models.py

# Expose port and start your server
EXPOSE 10000
CMD ["python", "server.py"]
