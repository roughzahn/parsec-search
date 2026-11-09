FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir laion-clap

WORKDIR /search
COPY server.py /search/server.py
COPY embedding_store.py /search/embedding_store.py

ENV HF_HOME=/models/huggingface
EXPOSE 8000
ENTRYPOINT ["python", "/search/server.py"]
