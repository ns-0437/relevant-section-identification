"""
Hybrid retrieval: dense vectors (EmbeddingGemma) fused with sparse BM25.

    python hybrid_search.py "what resistor value sets the top-off current?" -k 5
    python hybrid_search.py "STAT pin" --alpha 0.6 --mode hybrid

Fusion is a weighted sum of min-max normalised scores:

    score = alpha * norm(cosine) + (1 - alpha) * norm(bm25)

Both score families are computed over the *entire* collection rather than over a
truncated candidate list. With ~10^2 chunks that is essentially free and avoids the
usual hybrid artefact where a document ranked highly by one retriever is missing
from the other's top-N and gets silently treated as a zero.

Min-max normalisation is per-query: BM25 is unbounded and cosine sits in a narrow
band (~0.35-0.65 on this corpus), so raw scores are not comparable and a weighted
sum over them would be dominated by BM25's scale, not by alpha.
"""

from __future__ import annotations

import argparse
import re
import textwrap
from typing import Any

DEFAULT_MODEL = "google/embeddinggemma-300m"
DEFAULT_ALPHA = 0.6          # weight on the dense/vector side

_TOKEN = re.compile(r"[a-z0-9_]+")

# Trimmed stoplist: keeps domain tokens that a generic list would drop.
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "how", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to",
    "was", "what", "when", "where", "which", "who", "why", "will", "with", "does",
    "do", "did", "can", "should", "would", "there", "these", "those",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower())
            if len(t) > 1 and t not in _STOP]


def minmax(scores: list[float]) -> list[float]:
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


class HybridRetriever:
    """Loads the whole collection into memory: fine at this scale, not at 10^6."""

    def __init__(self, db: str, collection: str, model_name: str = DEFAULT_MODEL,
                 device: str = "cuda", model=None) -> None:
        import chromadb
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self.coll = chromadb.PersistentClient(path=db).get_collection(collection)
        got = self.coll.get(include=["documents", "metadatas", "embeddings"])

        self.ids: list[str] = got["ids"]
        self.docs: list[str] = got["documents"]
        self.metas: list[dict[str, Any]] = got["metadatas"]
        self.embs = got["embeddings"]

        # BM25 sees title + body, mirroring what the dense side embeds
        corpus = [tokenize(f"{m.get('title','')} {d}")
                  for m, d in zip(self.metas, self.docs)]
        self.bm25 = BM25Okapi(corpus)

        # a shared model can be injected so several documents don't each load one
        self.model = model or SentenceTransformer(model_name, device=device)

    def dense_scores(self, text: str, prompt_name: str = "query") -> list[float]:
        """Cosine of every chunk against `text`, encoded with the given prompt.

        EmbeddingGemma is asymmetric: a real query uses the `query` prompt, but a
        HyDE passage is pretending to BE a document, so it is encoded with the
        `document` prompt instead.
        """
        import numpy as np

        v = self.model.encode(text, prompt_name=prompt_name, normalize_embeddings=True)
        return (np.asarray(self.embs) @ np.asarray(v)).tolist()

    def bm25_scores(self, text: str) -> list[float]:
        return self.bm25.get_scores(tokenize(text)).tolist()

    def rank(self, fused: list[float], k: int,
             chunk_types: list[str] | None = None) -> list[dict[str, Any]]:
        order = sorted(range(len(fused)), key=lambda i: -fused[i])
        out = []
        for i in order:
            if chunk_types and self.metas[i]["chunk_type"] not in chunk_types:
                continue
            out.append({"id": self.ids[i], "score": fused[i],
                        "meta": self.metas[i], "doc": self.docs[i]})
            if len(out) >= k:
                break
        return out

    def search(self, query: str, k: int = 5, alpha: float = DEFAULT_ALPHA,
               mode: str = "hybrid",
               chunk_types: list[str] | None = None) -> list[dict[str, Any]]:
        import numpy as np

        qv = self.model.encode(query, prompt_name="query", normalize_embeddings=True)
        cos = (np.asarray(self.embs) @ np.asarray(qv)).tolist()
        bm = self.bm25.get_scores(tokenize(query)).tolist()

        n_cos, n_bm = minmax(cos), minmax(bm)
        if mode == "vector":
            fused = n_cos
        elif mode == "bm25":
            fused = n_bm
        else:
            fused = [alpha * c + (1 - alpha) * b for c, b in zip(n_cos, n_bm)]

        order = sorted(range(len(fused)), key=lambda i: -fused[i])
        out = []
        for i in order:
            if chunk_types and self.metas[i]["chunk_type"] not in chunk_types:
                continue
            out.append({
                "id": self.ids[i], "score": fused[i],
                "cos": cos[i], "bm25": bm[i],
                "norm_cos": n_cos[i], "norm_bm25": n_bm[i],
                "meta": self.metas[i], "doc": self.docs[i],
            })
            if len(out) >= k:
                break
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid dense+BM25 search.")
    ap.add_argument("query", nargs="+")
    ap.add_argument("--db", default="./chroma")
    ap.add_argument("--collection", default="max77751")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help="weight on the vector side (default 0.6)")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "vector", "bm25"])
    ap.add_argument("--chars", type=int, default=300)
    args = ap.parse_args()

    q = " ".join(args.query)
    r = HybridRetriever(args.db, args.collection, args.model, args.device)
    hits = r.search(q, k=args.k, alpha=args.alpha, mode=args.mode)

    print(f'\nquery: "{q}"')
    print(f"mode={args.mode}  alpha={args.alpha}  (vector {args.alpha:.0%} / bm25 {1-args.alpha:.0%})")
    print("=" * 78)
    for rank, h in enumerate(hits, 1):
        m = h["meta"]
        body = m.get("core_text") or h["doc"]
        prev = textwrap.shorten(" ".join(body.split()), args.chars, placeholder=" ...")
        print(f"\n[{rank}] {m['chunk_type'].upper():5} page {m['page_number']:>3}  "
              f"score={h['score']:.3f}  (cos={h['cos']:.3f}→{h['norm_cos']:.2f}, "
              f"bm25={h['bm25']:.2f}→{h['norm_bm25']:.2f})")
        print(f"    title: {m.get('title','')[:66]}")
        print(textwrap.fill(prev, 74, initial_indent="    ", subsequent_indent="    "))


if __name__ == "__main__":
    main()
