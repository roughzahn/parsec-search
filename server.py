import hmac
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np
import torch
import laion_clap

from embedding_store import EmbeddingStore
from search_core import rank_vectors

DATA_DIR = os.environ.get("PARSEC_SEARCH_DATA_DIR", "/data")
MODEL_DIR = os.environ.get("PARSEC_SEARCH_MODEL_DIR", "/models")
LEGACY_CACHE = os.path.join(DATA_DIR, "embedding-index.npz")
DATABASE = os.path.join(DATA_DIR, "embeddings.db")
CHECKPOINT = os.path.join(MODEL_DIR, "630k-audioset-best.pt")
PORT = int(os.environ.get("PARSEC_SEARCH_PORT", "8000"))
INTERNAL_TOKEN = os.environ.get("PARSEC_SEARCH_INTERNAL_TOKEN", "").strip()
EMBEDDING_MODEL_ID = os.environ.get("PARSEC_SEARCH_EMBEDDING_MODEL_ID", "laion-clap-630k-audioset-v1")
EMBEDDING_VERSION = int(os.environ.get("PARSEC_SEARCH_EMBEDDING_VERSION", "1"))

if not os.path.isfile(CHECKPOINT):
    raise FileNotFoundError("CLAP checkpoint not found: %s (set PARSEC_SEARCH_MODEL_DIR to the model directory)" % CHECKPOINT)

STORE = EmbeddingStore(DATABASE, LEGACY_CACHE)
STORE.initialize(EMBEDDING_MODEL_ID, EMBEDDING_VERSION)
VECTORS, PATHS = STORE.snapshot(EMBEDDING_MODEL_ID, EMBEDDING_VERSION)
print("Loaded %d durable embeddings from %s" % (len(PATHS), DATABASE), flush=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading CLAP model on %s..." % DEVICE, flush=True)
MODEL = laion_clap.CLAP_Module(enable_fusion=False, device=DEVICE)
MODEL.load_ckpt(ckpt=CHECKPOINT, verbose=False)
print("MODEL READY. API-only mode.", flush=True)

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            count = STORE.count(EMBEDDING_MODEL_ID, EMBEDDING_VERSION)
            self._send(200, json.dumps({
                "ready": True, "vectors": count, "device": DEVICE,
                "embeddingModelId": EMBEDDING_MODEL_ID,
                "embeddingVersion": EMBEDDING_VERSION,
            }), "application/json")
        else:
            self._send(404, "Not found", "text/plain; charset=utf-8")

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/search", "/internal/v1/embeddings", "/internal/v1/embeddings/delete"):
            self._send(404, json.dumps({"error": "Not found"}), "application/json")
            return
        if path.startswith("/internal/"):
            if not INTERNAL_TOKEN:
                self._send(503, json.dumps({"error": "Embedding ingestion is not configured"}), "application/json")
                return
            supplied = self.headers.get("Authorization", "")
            expected = "Bearer " + INTERNAL_TOKEN
            if not hmac.compare_digest(supplied, expected):
                self._send(401, json.dumps({"error": "Unauthorized"}), "application/json")
                return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            max_length = 256 * 1024 if path == "/internal/v1/embeddings" else 10000
            if length < 1 or length > max_length:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON object required")
            if path == "/internal/v1/embeddings":
                result = STORE.upsert(payload, EMBEDDING_MODEL_ID, EMBEDDING_VERSION)
                self._send(202, json.dumps({"accepted": True, **result}), "application/json; charset=utf-8")
                return
            if path == "/internal/v1/embeddings/delete":
                result = STORE.delete(
                    payload.get("path"),
                    payload.get("expectedSizeBytes"),
                    payload.get("expectedLastModifiedUtc"),
                )
                self._send(202, json.dumps({"accepted": True, **result}), "application/json; charset=utf-8")
                return

            query = str(payload.get("query", "")).strip()
            if not query or len(query) > 500:
                raise ValueError("Enter a query between 1 and 500 characters")
            started = time.time()
            q = np.asarray(MODEL.get_text_embedding([query], use_tensor=False), dtype=np.float32).reshape(-1)
            vectors, paths = STORE.snapshot(EMBEDDING_MODEL_ID, EMBEDDING_VERSION)
            ranked = rank_vectors(vectors, q, limit=10)
            matches = [{"rank": rank, "score": score, "path": paths[index]}
                       for rank, (index, score) in enumerate(ranked, 1)]
            result = {"query": query, "seconds": round(time.time() - started, 2), "results": matches}
            self._send(200, json.dumps(result, ensure_ascii=True), "application/json; charset=utf-8")
        except ValueError as exc:
            self._send(400, json.dumps({"error": str(exc)}, ensure_ascii=True), "application/json; charset=utf-8")
        except Exception as exc:
            self._send(500, json.dumps({"error": "Internal search error"}, ensure_ascii=True), "application/json; charset=utf-8")
    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)

print("Listening on 0.0.0.0:%d" % PORT, flush=True)
ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
