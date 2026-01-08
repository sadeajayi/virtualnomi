FROM python:3.11-slim

WORKDIR /app

# Install only system dependencies needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies and clean up
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    rm -rf /root/.cache/pip

# Copy application code
COPY app.py .

# Don't download models at build time - download at runtime
# This significantly reduces image size
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/hf_home

# Expose port
EXPOSE $PORT

# Run the application
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
