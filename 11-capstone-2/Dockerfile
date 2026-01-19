FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (optional but safe)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and app
COPY predict.py .
COPY model.bin .

# Expose FastAPI port
EXPOSE 9696

# Run the app
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
