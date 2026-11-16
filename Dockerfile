# Parakeet-TDT-0.6B-v3 behind an OpenAI-compatible /v1/audio/transcriptions API.
# GPU: run with --gpus all (NVIDIA driver + NVIDIA Container Toolkit on the host).
# The full (non-slim) Python image is used because some NeMo dependencies may need a compiler.
FROM python:3.12-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/data/hf \
    TORCH_HOME=/data/torch \
    PORT=8000

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ /app/
RUN chmod +x /app/start.sh

# The downloaded model (~2.5 GB) lives here; mount a volume so it survives restarts.
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4)"

CMD ["/app/start.sh"]
