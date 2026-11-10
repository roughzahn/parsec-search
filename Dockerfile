FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.source="https://github.com/roughzahn/parsec-search"

WORKDIR /search
COPY requirements.txt /search/requirements.txt
RUN pip install --no-cache-dir -r /search/requirements.txt

COPY server.py search_core.py embedding_store.py /search/

ENV HF_HOME=/models/huggingface \
    PYTHONUNBUFFERED=1 \
    PARSEC_SEARCH_UI_ENABLED=false
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1
ENTRYPOINT ["python", "/search/server.py"]
