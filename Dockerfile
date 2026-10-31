FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

RUN pip install --no-cache-dir laion-clap

WORKDIR /search
COPY server.py /search/server.py

ENV HF_HOME=/models/huggingface
EXPOSE 8000
ENTRYPOINT ["python", "/search/server.py"]
