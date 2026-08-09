"""
Metadata as a third retrieval channel.

Content retrieval matches the query against the chunk body. This adds a parallel
match against the chunk's *descriptors* — section heading, table caption, OCR
keywords — which carry different information:

  * a heading states what a region of the document is ABOUT, compactly and without
    the surrounding prose diluting it. The task literally asks for section
    headings, so matching them directly is on-target rather than incidental.
  * OCR keywords are the only trustworthy signal on figure pages, where the LLaVA
    caption is unreliable — but in the content channel they are ~8% of the text
    and get swamped by the caption.

    score = (1-gamma) * content_hybrid + gamma * metadata_hybrid

Both channels are min-max normalised before fusing, as elsewhere.
"""

from __future__ import annotations

import argparse
import textwrap
from typing import Any

from hybrid_search import HybridRetriever, tokenize, minmax, DEFAULT_ALPHA

DEFAULT_GAMMA = 0.3


class MetaRetriever(HybridRetriever):
    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        from rank_bm25 import BM25Okapi
        import numpy as np

        self.meta_docs: list[str] = []
        for m in self.metas:
            parts = [
                m.get("title") or "",
                m.get("table_caption") or "",
                (m.get("ocr_keywords") or "").replace(",", " "),
                m.get("chunk_type") or "",
            ]
            self.meta_docs.append(" ".join(p for p in parts if p).strip())

        n_empty = sum(1 for d in self.meta_docs if not d)
        print(f"      metadata channel: {len(self.meta_docs)} descriptors "
              f"({n_empty} empty)")

        self.meta_bm25 = BM25Okapi([tokenize(d) for d in self.meta_docs])
        # descriptors are short label-like strings; embed them as documents
        self.meta_embs = np.asarray(self.model.encode(
            [f"title: {d} | text: {d}" for d in self.meta_docs],
            normalize_embeddings=True, batch_size=16, show_progress_bar=False))

    def meta_dense_scores(self, query: str) -> list[float]:
        import numpy as np

        v = self.model.encode(query, prompt_name="query", normalize_embeddings=True)
        return (self.meta_embs @ np.asarray(v)).tolist()

    def meta_bm25_scores(self, query: str) -> list[float]:
        return self.meta_bm25.get_scores(tokenize(query)).tolist()

    def fused_scores(self, query: str, alpha: float = DEFAULT_ALPHA,
                     gamma: float = DEFAULT_GAMMA) -> list[float]:
        content = [alpha * c + (1 - alpha) * b for c, b in zip(
            minmax(self.dense_scores(query, "query")),
            minmax(self.bm25_scores(query)))]
        if gamma <= 0:
            return content
        meta = [alpha * c + (1 - alpha) * b for c, b in zip(
            minmax(self.meta_dense_scores(query)),
            minmax(self.meta_bm25_scores(query)))]
        return [(1 - gamma) * c + gamma * m for c, m in zip(content, meta)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+")
    ap.add_argument("--db", default="./chroma")
    ap.add_argument("--collection", default="max77751_v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    q = " ".join(args.query)
    r = MetaRetriever(args.db, args.collection, device=args.device)
    for hit in r.rank(r.fused_scores(q, args.alpha, args.gamma), args.k):
        m = hit["meta"]
        body = m.get("core_text") or hit["doc"]
        print(f"\n[{m['chunk_type'].upper():5}] p{m['page_number']:>3} "
              f"score={hit['score']:.3f}  title={m.get('title','')[:50]!r}")
        print(textwrap.fill(" ".join(body.split())[:260], 74,
                            initial_indent="    ", subsequent_indent="    "))


if __name__ == "__main__":
    main()
