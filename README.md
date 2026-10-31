# parsec-search

AI-powered semantic music search service used by parsec-media-gateway.

## Responsibilities

- Browser UI: GET /
- Search API: POST /search
- Readiness endpoint: GET /health
- Natural-language query embeddings with LAION-CLAP and cosine-similarity ranking.
- Command-line search implementation in clap-search.py.

## API

POST /search accepts JSON:

~~~json
{"query":"dark electronic music with heavy bass"}
~~~

The response remains compatible with the gateway integration:

~~~json
{
  "query": "dark electronic music with heavy bass",
  "seconds": 0.42,
  "results": [
    {"rank": 1, "score": 0.71, "path": "Artist/Album/track.flac"}
  ]
}
~~~

## Runtime data

This repository contains application code only. Large/generated runtime assets remain outside Git in C:\\music-ai:

- C:\\music-ai\\model-cache\\embedding-index.npz
- C:\\music-ai\\model-cache\\630k-audioset-best.pt
- C:\\music-ai\\model-cache\\huggingface\\

The Docker container mounts C:\\music-ai\\model-cache at /models. The embedding index stores relative music-library paths; those paths are returned unchanged by the API. Music files and embeddings are not copied into this repository.

## Docker

Build from this repository root:

~~~powershell
docker build -t music-ai-search:latest .
~~~

The image listens on container port 8000. The gateway reaches it on the shared Docker network as http://music-ai-search:8000; the local browser endpoint is normally http://127.0.0.1:8765.

The model and vector cache must be mounted at /models when creating the container. Do not add model files, generated indexes, embeddings, or music files to Git.

## Local Python

For a non-Docker environment, install dependencies with:

~~~powershell
python -m pip install -r requirements.txt
~~~

The CLI script currently expects data paths configured at the top of clap-search.py (/data, /embeddings, /models), matching its Linux-container execution environment.
