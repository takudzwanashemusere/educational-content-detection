FROM python:3.10-slim

# Install ffmpeg and OpenCV system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 10000

# Use gunicorn for production
#CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--timeout", "300", "--workers", "1", "api:app"]
CMD gunicorn --bind 0.0.0.0:$PORT --timeout 300 --workers 1 api:app
