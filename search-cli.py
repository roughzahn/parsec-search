import csv
import os
import sys
import time
import numpy as np
import torch
import laion_clap

INDEX = "/data/clap-index.csv"
EMBED_DIR = "/embeddings"
CACHE = "/models/embedding-index.npz"
if len(sys.argv) < 2:
    raise SystemExit('Usage: python /search.py "describe the music"')
query = " ".join(sys.argv[1:]).strip()
if not query:
    raise SystemExit("Query is empty")

t0 = time.time()
index_mtime = os.path.getmtime(INDEX)
vectors = None
paths = None
ids = None
if os.path.isfile(CACHE):
    try:
        saved = np.load(CACHE, allow_pickle=False)
        if float(saved["index_mtime"]) == index_mtime:
            vectors = saved["vectors"]
            paths = saved["paths"].tolist()
            ids = saved["ids"].tolist()
            print("Loaded cached vectors: %d" % len(paths), flush=True)
        saved.close()
    except Exception:
        vectors = None

if vectors is None:
    print("Building local vector cache from existing embeddings...", flush=True)
    rows_vec = []
    paths = []
    ids = []
    missing = 0
    with open(INDEX, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for n, row in enumerate(reader, 1):
            if len(row) < 5:
                continue
            track_id, relpath, length, modified, emb_name = row[:5]
            emb_path = os.path.join(EMBED_DIR, os.path.basename(emb_name))
            if not os.path.isfile(emb_path):
                missing += 1
                continue
            try:
                vec = np.asarray(np.load(emb_path, allow_pickle=False), dtype=np.float32).reshape(-1)
                if vec.size != 512:
                    continue
                norm = np.linalg.norm(vec)
                if norm == 0:
                    continue
                rows_vec.append(vec / norm)
                paths.append(relpath)
                ids.append(track_id)
            except Exception:
                continue
            if n % 2000 == 0:
                print("Index rows checked: %d; usable vectors: %d" % (n, len(rows_vec)), flush=True)
    if not rows_vec:
        raise RuntimeError("No usable embeddings found")
    vectors = np.stack(rows_vec).astype(np.float32)
    np.savez_compressed(CACHE, vectors=vectors, paths=np.asarray(paths), ids=np.asarray(ids), index_mtime=np.float64(index_mtime))
    print("Cached %d vectors; missing embedding files: %d" % (len(paths), missing), flush=True)

print("Loading CLAP model...", flush=True)
model = laion_clap.CLAP_Module(enable_fusion=False, device="cuda" if torch.cuda.is_available() else "cpu")
model.load_ckpt(ckpt="/models/630k-audioset-best.pt", verbose=False)
print("Model ready in %.1f s; device=%s" % (time.time() - t0, "cuda" if torch.cuda.is_available() else "cpu"), flush=True)

q = np.asarray(model.get_text_embedding([query], use_tensor=False), dtype=np.float32).reshape(-1)
qnorm = np.linalg.norm(q)
if qnorm == 0:
    raise RuntimeError("CLAP returned a zero query vector")
q /= qnorm
scores = vectors @ q
top = np.argsort(scores)[-10:][::-1]
print("")
print("Query: %s" % query)
print("Usable embeddings: %d" % len(paths))
print("Top 10 matches (cosine similarity):")
for rank, i in enumerate(top, 1):
    print("%2d. %.4f | %s" % (rank, float(scores[i]), paths[int(i)]))
print("Search finished in %.1f s" % (time.time() - t0), flush=True)
