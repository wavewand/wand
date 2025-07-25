FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make main.py executable
RUN chmod +x main.py

# Expose API port
EXPOSE 8000

# Default command
CMD ["python", "main.py", "--api-host", "0.0.0.0", "--api-port", "8000"]