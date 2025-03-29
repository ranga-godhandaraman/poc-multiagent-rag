# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/data/vector_db /app/data/temp_uploads /app/data/embeddings_cache

# Copy the application code
COPY . .

# Expose the Streamlit default port
EXPOSE 8501

# Set environment variables
ENV DATA_DIR=/app/data
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Command to run the application with CORS enabled
CMD ["streamlit", "run", "multi_agent_rag.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.enableCORS=true"]
