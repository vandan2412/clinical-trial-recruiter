FROM python:3.11-slim

LABEL org.opencontainers.image.title="ClinicalTrialRecruiter"
LABEL org.opencontainers.image.description="OpenEnv environment for AI clinical trial patient recruitment"
LABEL org.opencontainers.image.version="1.0.0"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY server/ ./server/
COPY app.py .
COPY inference.py .
COPY openenv.yaml .
COPY pyproject.toml .

RUN useradd -m -u 1000 openenv
RUN chown -R openenv:openenv /app
USER openenv

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "app.py"]
