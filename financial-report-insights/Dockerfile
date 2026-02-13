# ---- Stage 1: Build dependencies ----
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: Runtime ----
FROM python:3.12-slim

# Prevent .pyc files and enable real-time log output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install tini for proper signal handling (graceful SIGTERM)
RUN apt-get update && \
    apt-get install -y --no-install-recommends tini && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY *.py ./

# Create directories with correct permissions
RUN mkdir -p /app/documents /app/.cache /app/.streamlit && \
    chown -R appuser:appuser /app

# Create minimal Streamlit config
RUN printf '[server]\nheadless = true\nmaxUploadSize = 200\n' > /app/.streamlit/config.toml && \
    chown appuser:appuser /app/.streamlit/config.toml

# Switch to non-root user
USER appuser

# Expose Streamlit + API ports
EXPOSE 8501 8504

# Health check using Streamlit's built-in health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Use tini as init process for proper signal forwarding
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sh", "-c", \
    "uvicorn api:app --host 0.0.0.0 --port ${RAG_API_PORT:-8504} --log-level info & \
     streamlit run streamlit_app_local.py \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --browser.gatherUsageStats=false"]
