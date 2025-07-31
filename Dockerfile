# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src ./src
COPY artifacts ./artifacts

# Default command – run inference
CMD ["python", "-m", "src.predict"]
