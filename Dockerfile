

# 1. Base Image
FROM python:3.10-slim

# 2. Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 3. Install System Dependencies
# - Install Tesseract OCR with Japanese language support
# - Install required libraries for OpenCV
# - Clean up apt cache to keep the image lightweight
RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-jpn \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 4. Set Working Directory
WORKDIR /app

# 5. Copy and Install Python Dependencies
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code
COPY . .

# 7. Expose Port and Define Startup Command
# Railway provides the PORT environment variable
CMD uvicorn main:app --host 0.0.0.0 --port $PORT

