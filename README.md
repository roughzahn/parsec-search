# parsec-search

AI-powered semantic music search service used by `parsec-media-gateway`.

## Responsibilities

- Browser UI: `GET /`
- Search API: `POST /search`
- Readiness endpoint: `GET /health`
- Natural-language query embeddings with LAION-CLAP and cosine-similarity ranking.
- Command-line index builder and search implementation in `search-cli.py`.

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

## Runtime data and Docker volumes

The repository contains application code and deployment configuration only. Runtime data is stored on the Docker host and mounted into the container:

| Container path | Purpose | Windows development default | Linux/vServer example |
|---|---|---|---|
| `/data` | `embedding-index.npz` | `C:\music-ai\search-data` | `/opt/parsec-search/data` |
| `/models` | CLAP checkpoint and Hugging Face cache | `C:\music-ai\model-cache` | `/opt/parsec-search/models` |

The server reads the index from `/data/embedding-index.npz` and the CLAP checkpoint from `/models/630k-audioset-best.pt`. Both paths can be overridden with `PARSEC_SEARCH_DATA_DIR` and `PARSEC_SEARCH_MODEL_DIR` inside the container.

The host directories are configured through `PARSEC_SEARCH_DATA_PATH` and `PARSEC_SEARCH_MODEL_PATH`. Windows development uses the defaults in `compose.yaml`. On the vServer, copy `.env.example` to `.env` and adjust the two Linux paths. The `.env` file is intentionally excluded from Git.

Replacing the Docker image or container does not modify either host directory. The search index therefore survives code changes, image rebuilds, container removal and recreation.

## GPU and CPU deployment

The base `compose.yaml` does not request a GPU and is suitable for a CPU-only host such as the vServer:

~~~bash
docker compose up -d --build
~~~

Windows development starts the GPU-enabled variant through:

~~~powershell
.\\start-parsec-search.cmd
~~~

This adds `compose.gpu.yaml`, which requests all available GPUs. Both variants use the same image, application code, API and persistent data layout. PyTorch selects CUDA when available and otherwise uses the CPU.

Stop the Windows development deployment:

~~~powershell
.\\stop-parsec-search.cmd
~~~

## Index update procedure

The index is a generated artifact. To replace it on the vServer:

1. Stop `parsec-search`.
2. Upload the new `embedding-index.npz` to the host data directory.
3. Replace `/opt/parsec-search/data/embedding-index.npz` and verify the file.
4. Start the service again and check `GET /health`.

The index is not stored in the image or in the Git repository. No index backup is maintained by this service.

## Endpoints

- Container endpoint for the gateway: `http://parsec-search:8000`
- Local browser endpoint: `http://127.0.0.1:8765`
- Container port: `8000`
- Host port: `8765` (bound to loopback only)

## CLI index builder

`search-cli.py` can rebuild the cached vector index from the worker's CSV and individual embedding files. Its paths are configurable with:

- `PARSEC_SEARCH_SOURCE_INDEX` (default `/data/clap-index.csv`)
- `PARSEC_SEARCH_EMBEDDINGS_DIR` (default `/embeddings`)
- `PARSEC_SEARCH_DATA_DIR` (output directory for `embedding-index.npz`)
- `PARSEC_SEARCH_MODEL_DIR` (CLAP checkpoint directory)

The worker's source CSV and individual embeddings are not required by the running search server once `embedding-index.npz` has been generated and transferred.
