FROM python:3.11-slim

LABEL org.opencontainers.image.title="ClinicalTrialRecruiter"
LABEL org.opencontainers.image.description="OpenEnv environment for AI clinical trial patient recruitment"
LABEL org.opencontainers.image.version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY data/ ./data/
COPY app.py .
COPY inference.py .
COPY openenv.yaml .

# Create non-root user (security best practice)
RUN useradd -m -u 1000 openenv
RUN chown -R openenv:openenv /app
USER openenv

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the app server
CMD ["python", "app.py"]
