"""
Benchmark the pipeline: embedding throughput on GPU vs CPU, retrieval latency
broken down by stage, and peak memory per model.

    python benchmark.py                 # both devices
    python benchmark.py --device cpu    # one device

Numbers in docs/REPORT.md section 5.2 come from this script.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHUNKS = "sample_v3.json"
COLLECTION = "max77751_v3"
EMBED_MODEL = "google/embeddinggemma-300m"

QUERIES = [
    "What battery chemistries are supported by this charger?",
    "What resistor value sets the top-off current?",
    "What happens if the chip gets too hot?",
    "What is VCHGIN_OVLO?",
    "How can an MCU enable and disable battery charging?",
    "Which inductors are recommended?",
    "What is the pin configuration of the package?",
    "How does the charger decide that a power source is valid?",
]


def _vram():
    import torch
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 2**30


def bench_embedding(device: str, texts: list[str]) -> dict:
    import torch
    from sentence_transformers import SentenceTransformer

    if device == "cuda":
        if not torch.cuda.is_available():
            return {"device": device, "skipped": "no CUDA"}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    model = SentenceTransformer(EMBED_MODEL, device=device)
    load = time.time() - t0

    model.encode(texts[:4], batch_size=4, normalize_embeddings=True)   # warm-up

    t0 = time.time()
    vecs = model.encode(texts, batch_size=8, normalize_embeddings=True,
                        show_progress_bar=False)
    enc = time.time() - t0

    out = {"device": device, "load_s": round(load, 1), "encode_s": round(enc, 2),
           "chunks": len(texts), "per_chunk_ms": round(enc / len(texts) * 1000, 1),
           "chunks_per_s": round(len(texts) / enc, 1), "dim": int(vecs.shape[1])}
    if device == "cuda":
        out["peak_vram_gib"] = round(_vram(), 2)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def bench_retrieval(device: str) -> dict:
    from hybrid_search import minmax
    from section_search import SectionRetriever

    r = SectionRetriever("./chroma", COLLECTION, device=device)
    r.dense_scores(QUERIES[0], "query")            # warm-up

    dense, sparse, fused, total = [], [], [], []
    for q in QUERIES:
        t0 = time.time(); d = r.dense_scores(q, "query"); dense.append(time.time() - t0)
        t0 = time.time(); b = r.bm25_scores(q);           sparse.append(time.time() - t0)
        t0 = time.time()
        nd, nb = minmax(d), minmax(b)
        [0.6 * x + 0.4 * y for x, y in zip(nd, nb)]
        fused.append(time.time() - t0)
        t0 = time.time(); r.fused_scores(q, 0.6, 0.5, 4); total.append(time.time() - t0)

    ms = lambda v: round(statistics.median(v) * 1000, 1)
    return {"device": device, "queries": len(QUERIES), "chunks": len(r.ids),
            "dense_ms": ms(dense), "bm25_ms": ms(sparse), "fuse_ms": ms(fused),
            "end_to_end_ms": ms(total)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--chunks", default=CHUNKS)
    args = ap.parse_args()

    import torch
    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        free, tot = torch.cuda.mem_get_info()
        print(f"gpu: {torch.cuda.get_device_name(0)}  {tot/2**30:.1f} GiB total, "
              f"{free/2**30:.2f} free")
    print(f"cpu: {os.cpu_count()} logical cores\n")

    payload = json.load(open(args.chunks, encoding="utf-8"))
    texts = [f"title: {c['metadata'].get('title') or 'none'} | text: {c['text']}"
             for c in payload]

    devices = [args.device] if args.device else (
        ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])

    print("=" * 74)
    print("EMBEDDING  (EmbeddingGemma-300m, 104 chunks)")
    print("=" * 74)
    print(f"{'device':<7}{'load':>8}{'encode':>9}{'per chunk':>12}{'rate':>14}{'peak VRAM':>12}")
    for dev in devices:
        r = bench_embedding(dev, texts)
        if r.get("skipped"):
            print(f"{dev:<7}  skipped: {r['skipped']}"); continue
        print(f"{r['device']:<7}{r['load_s']:>7.1f}s{r['encode_s']:>8.2f}s"
              f"{r['per_chunk_ms']:>10.1f}ms{r['chunks_per_s']:>11.1f}/s"
              f"{(str(r.get('peak_vram_gib','-'))+' GiB'):>12}")

    print()
    print("=" * 74)
    print("RETRIEVAL  (median over 8 queries, whole collection scored)")
    print("=" * 74)
    print(f"{'device':<7}{'vector':>10}{'bm25':>10}{'fuse':>10}{'end-to-end':>14}")
    for dev in devices:
        r = bench_retrieval(dev)
        print(f"{r['device']:<7}{r['dense_ms']:>8.1f}ms{r['bm25_ms']:>8.1f}ms"
              f"{r['fuse_ms']:>8.1f}ms{r['end_to_end_ms']:>12.1f}ms")


if __name__ == "__main__":
    main()
