# Builder stage
FROM python:3.11-slim AS builder


LABEL maintainer="Kaothar Adedeji" 
LABEL created="2025-07-30"

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy requirements first for better caching
COPY requirements.txt .

# Install packages to default location (/usr/local)
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

LABEL maintainer="Kaothar Adedeji" 
LABEL created="2025-07-30"

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy the installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create models directory if it doesn't exist
RUN mkdir -p /app/models

EXPOSE 8000

# Use python -m instead of direct gunicorn call for better compatibility
CMD ["python", "-m", "gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "app.main:app", "--bind", "0.0.0.0:8000"]