# parsec-search

AI-powered semantic music search service used by `parsec-media-gateway`.

## Responsibilities

- Browser UI: `GET /`
- Search API: `POST /search`
- Internal embedding ingestion: `POST /internal/v1/embeddings` (Bearer token)
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

## Browser playback

The browser UI uses the public Parsec Media Gateway API for searches and playback. Search results contain opaque track IDs; the UI requests a short-lived stream ticket before playing a track in the built-in audio player. The browser never receives NAS filesystem paths.

The gateway address is configured with `PARSEC_GATEWAY_BASE_URL` (Windows development default: `http://127.0.0.1:8766`). The gateway must allow the UI origin through `PARSEC_SEARCH_UI_ORIGINS` (default: `http://127.0.0.1:8765,http://localhost:8765`).

## Runtime data and Docker volumes

The repository contains application code and deployment configuration only. Runtime data is stored on the Docker host and mounted into the container:

| Container path | Purpose | Windows development default | Linux/vServer example |
|---|---|---|---|
| `/data` | `embeddings.db` (SQLite; legacy NPZ imported once) | `C:\music-ai\search-data` | `/opt/parsec-search/data` |
| `/models` | CLAP checkpoint and Hugging Face cache | `C:\music-ai\model-cache` | `/opt/parsec-search/models` |

The server stores each normalized 512-dimensional embedding in `/data/embeddings.db` and loads an immutable NumPy matrix into memory for fast cosine search. The database is the durable source of truth; the in-memory matrix is refreshed after ingestion. On the first start only, an existing `/data/embedding-index.npz` is imported automatically. The CLAP checkpoint is loaded from `/models/630k-audioset-best.pt`. Data and model directories can be overridden with `PARSEC_SEARCH_DATA_DIR` and `PARSEC_SEARCH_MODEL_DIR` inside the container.

The host directories are configured through `PARSEC_SEARCH_DATA_PATH` and `PARSEC_SEARCH_MODEL_PATH`. Windows development uses the defaults in `compose.yaml`. On the vServer, copy `.env.example` to `.env` and adjust the two Linux paths. Set `PARSEC_SEARCH_INTERNAL_TOKEN` to the same secret in Gateway and Search; it protects the internal ingestion endpoint. `PARSEC_SEARCH_EMBEDDING_MODEL_ID` and `PARSEC_SEARCH_EMBEDDING_VERSION` must match the worker output. The `.env` file is intentionally excluded from Git.

Replacing the Docker image or container does not modify either host directory. The SQLite database therefore survives code changes, image rebuilds, container removal and recreation. SQLite uses WAL mode. Back up the database with SQLite backup tooling rather than copying only the main `.db` file while the service is running.

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

## Embedding ingestion

`POST /internal/v1/embeddings` is an internal, authenticated endpoint. It accepts the completed worker result (model ID, embedding version, 512 float values and source audio metadata) with `Authorization: Bearer <PARSEC_SEARCH_INTERNAL_TOKEN>`. The endpoint validates the model/version, vector dimensions, finite values and relative source path, then performs an idempotent upsert keyed by music path. The stored vector is normalized before persistence. A successful write returns HTTP 202. The active in-memory matrix is refreshed lazily on the next search request, so a batch of worker completions does not rebuild it once per track.

## Legacy index migration

The old NPZ file is supported as a one-time migration source only. A newly initialized, empty database imports it automatically. Uploading a replacement NPZ does not overwrite an existing database. To deliberately rebuild from a new legacy NPZ, stop the service, make a verified backup of `embeddings.db`, remove the database and its `-wal`/`-shm` companions, replace the NPZ, then start the service to trigger the one-time import. Do not do this for normal worker updates; use the ingestion API instead.

The SQLite database is not stored in the image or in the Git repository. No database backup is maintained by this service. The old NPZ file is retained as a migration source and is imported only when a new, empty database is initialized.

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

This remains a legacy offline tool. Its NPZ output is imported only into a new, empty SQLite database; it does not update an existing live database.
## Quality checks and releases

Pull requests run the GitHub Actions workflow in `.github/workflows/ci.yml`. It validates the Docker build definition, compiles the Python application, runs unit tests without downloading the CLAP model, and builds the complete Docker image.

Stable releases are created manually from the GitHub Actions tab using `Release`. Choose `major`, `minor` or `patch`; the workflow calculates the next semantic version and publishes an immutable version tag to GHCR. Deployments must pin the exact version; no mutable `latest` or `stable` tag is published.

The Docker image disables the browser UI by default (`PARSEC_SEARCH_UI_ENABLED=false`). The local `compose.yaml` enables it for Windows development; production deployments should keep it disabled. With the UI disabled, `GET /` returns 404 while `/search` and `/health` remain available.