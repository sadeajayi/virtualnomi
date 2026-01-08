FROM python:3.11-slim

WORKDIR /app

# Install pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .

# Expose port
EXPOSE $PORT

# Start command
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
