import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import numpy as np
import torch
import laion_clap

CACHE = "/models/embedding-index.npz"
CHECKPOINT = "/models/630k-audioset-best.pt"
PORT = 8000

print("Loading cached music vectors...", flush=True)
saved = np.load(CACHE, allow_pickle=False)
VECTORS = saved["vectors"]
PATHS = saved["paths"].tolist()
IDS = saved["ids"].tolist()
saved.close()
print("Vectors ready: %d" % len(PATHS), flush=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading CLAP model on %s..." % DEVICE, flush=True)
MODEL = laion_clap.CLAP_Module(enable_fusion=False, device=DEVICE)
MODEL.load_ckpt(ckpt=CHECKPOINT, verbose=False)
print("MODEL READY. Open http://127.0.0.1:8765", flush=True)

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>parsec-search</title>
<style>
:root{color-scheme:dark}
body{font:16px/1.5 system-ui,Segoe UI,sans-serif;background:#101318;color:#edf0f6;margin:0}
main{max-width:920px;margin:0 auto;padding:36px 20px}
h1{font-size:28px;margin:0 0 8px}
.sub{color:#aeb7c7;margin:0 0 24px}
form{display:flex;gap:10px;flex-wrap:wrap}
input{flex:1;min-width:240px;padding:14px 16px;border:1px solid #3b4351;border-radius:10px;background:#1b2029;color:#fff;font-size:16px}
button{padding:14px 20px;border:0;border-radius:10px;background:#8cb8ff;color:#111722;font-weight:700;font-size:16px;cursor:pointer}
button:disabled{opacity:.6;cursor:wait}
.examples{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 22px}
.examples button{padding:8px 11px;background:#252d3a;color:#dbe5f5;font-size:13px;font-weight:500}
#status{color:#b8c4d6;min-height:26px;margin:12px 0}
.result{padding:14px 16px;margin:8px 0;background:#1a1f28;border:1px solid #2d3440;border-radius:10px}
.top{display:flex;gap:12px;align-items:baseline}
.rank{color:#91baff;font-weight:700;min-width:28px}
.score{margin-left:auto;color:#aab6c8;font-variant-numeric:tabular-nums}
.path{overflow-wrap:anywhere;color:#e4e9f2;margin-top:5px}
.note{font-size:13px;color:#8995a7;margin-top:26px}
</style>
</head>
<body><main>
<h1>parsec-search</h1>
<p class="sub">Search your existing music by sound, style, mood or instrumentation. The music files are not changed.</p>
<form id="form"><input id="query" maxlength="500" placeholder="e.g. dark electronic music with heavy bass" required><button id="go" type="submit">Search</button></form>
<div class="examples">
<button type="button" data-q="energetic electronic dance music with a strong bass">Electronic / bass</button>
<button type="button" data-q="dark aggressive industrial rock with distorted guitars">Dark industrial rock</button>
<button type="button" data-q="relaxed soulful trip hop with female vocals">Soulful trip hop</button>
</div>
<div id="status" role="status"></div>
<div id="results"></div>
<p class="note">Runs locally on this PC. Similarity scores are ranking hints, not percentages or probabilities.</p>
</main>
<script>
const form=document.getElementById('form'), input=document.getElementById('query'), go=document.getElementById('go'), status=document.getElementById('status'), results=document.getElementById('results');
document.querySelectorAll('[data-q]').forEach(b=>b.addEventListener('click',()=>{input.value=b.dataset.q;input.focus()}));
form.addEventListener('submit',async e=>{
 e.preventDefault(); const query=input.value.trim(); if(!query)return;
 go.disabled=true; status.textContent='Searching...'; results.replaceChildren();
 try{
  const r=await fetch('/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query})});
  const data=await r.json(); if(!r.ok)throw new Error(data.error||'Search failed');
  status.textContent='Found '+data.results.length+' matches in '+data.seconds+' s';
  data.results.forEach(item=>{
   const card=document.createElement('div');card.className='result';
   const top=document.createElement('div');top.className='top';
   const rank=document.createElement('span');rank.className='rank';rank.textContent='#'+item.rank;
   const score=document.createElement('span');score.className='score';score.textContent='similarity '+item.score.toFixed(4);
   top.append(rank,score);
   const path=document.createElement('div');path.className='path';path.textContent=item.path;
   card.append(top,path);results.append(card);
  });
 }catch(err){status.textContent='Error: '+err.message}
 finally{go.disabled=false}
});
</script>
</body></html>"""

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
        if path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/health":
            self._send(200, json.dumps({"ready": True, "vectors": len(PATHS), "device": DEVICE}), "application/json")
        else:
            self._send(404, "Not found", "text/plain; charset=utf-8")

    def do_POST(self):
        if urlparse(self.path).path != "/search":
            self._send(404, json.dumps({"error": "Not found"}), "application/json")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 10000:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            query = str(payload.get("query", "")).strip()
            if not query or len(query) > 500:
                raise ValueError("Enter a query between 1 and 500 characters")
            started = time.time()
            q = np.asarray(MODEL.get_text_embedding([query], use_tensor=False), dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(q)
            if norm == 0:
                raise ValueError("The model returned an empty vector")
            q /= norm
            scores = VECTORS @ q
            top = np.argsort(scores)[-10:][::-1]
            matches = [{"rank": n + 1, "score": float(scores[int(i)]), "path": PATHS[int(i)]}
                       for n, i in enumerate(top)]
            result = {"query": query, "seconds": round(time.time() - started, 2), "results": matches}
            self._send(200, json.dumps(result, ensure_ascii=True), "application/json; charset=utf-8")
        except Exception as exc:
            self._send(400, json.dumps({"error": str(exc)}, ensure_ascii=True), "application/json; charset=utf-8")

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)

print("Listening on 0.0.0.0:%d" % PORT, flush=True)
ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()

