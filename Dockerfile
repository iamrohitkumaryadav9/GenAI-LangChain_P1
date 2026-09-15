# ==============================================================================
# DocQA — Dockerfile
# Builds and runs the FastAPI + Streamlit application
# ==============================================================================

FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Generate sample doc if not present
RUN python eval/generate_sample_doc.py || true

# Create chroma_db directory
RUN mkdir -p chroma_db

# Supervisor config to run both services
RUN cat > /etc/supervisor/conf.d/docqa.conf <<EOF
[supervisord]
nodaemon=true

[program:fastapi]
command=python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

[program:streamlit]
command=python -m streamlit run frontend/streamlit_app.py --server.port 8501 --server.headless true --server.address 0.0.0.0
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
EOF

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run supervisor
CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]
