# parsec-search

AI-powered semantic music search service used by `parsec-media-gateway`.

## Responsibilities

- Browser UI: `GET /`
- Search API: `POST /search`
- Readiness endpoint: `GET /health`
- Natural-language query embeddings with LAION-CLAP and cosine-similarity ranking.
- Command-line search implementation in `search-cli.py`.

## API

`POST /search` accepts JSON:

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

The repository contains application code and deployment configuration only. Large/generated runtime assets remain outside Git:

- `C:\\music-ai\\model-cache\\embedding-index.npz`
- `C:\\music-ai\\model-cache\\630k-audioset-best.pt`
- `C:\\music-ai\\model-cache\\huggingface\\`

Compose mounts `C:\\music-ai\\model-cache` at `/models`. The embedding index stores relative music-library paths; those paths are returned unchanged by the API. Music files and embeddings are not copied into this repository.

## Docker Compose deployment

Start from this repository:

~~~powershell
.\start-parsec-search.cmd
~~~

Stop:

~~~powershell
.\stop-parsec-search.cmd
~~~

Compose builds the image `parsec-search:latest`, creates the container `parsec-search`, and joins the external Docker network `music-ai-network`.

- Container endpoint for the gateway: `http://parsec-search:8000`
- Local browser endpoint: `http://127.0.0.1:8765`
- Container port: `8000`
- Host port: `8765` (bound to loopback only)

The NVIDIA GPU is made available to the container. The CLAP model and vector cache are mounted from the external runtime directory; they are never stored in Git.

## Local Python

For a non-Docker environment, install dependencies with:

~~~powershell
python -m pip install -r requirements.txt
~~~

`search-cli.py` currently expects data paths configured at the top of the script (`/data`, `/embeddings`, `/models`), matching its Linux-container execution environment.
