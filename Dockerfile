FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir laion-clap

WORKDIR /search
COPY server.py search_core.py /search/

ENV HF_HOME=/models/huggingface \
    PYTHONUNBUFFERED=1 \
    PARSEC_SEARCH_UI_ENABLED=true
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1
ENTRYPOINT ["python", "/search/server.py"]
