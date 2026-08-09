"""
Cross-encoder reranking over a fixed page pool.

Retrieval gets ~95% of gold pages into the top 20 but only ~79% into the top 5, so
the remaining gap is ordering, not recall. A cross-encoder reads (query, chunk)
jointly instead of comparing two independently-produced vectors, which is what a
bi-encoder cannot do.

Pipeline:
  1. hybrid + section expansion  -> ranked chunks
  2. collapse to pages, keep the top `pool_pages`
  3. score every chunk on those pages with the cross-encoder
  4. page score = max over its chunks   (a page is relevant if ANY part answers)
  5. optionally blend with the retrieval score, since the reranker sees only 512
     tokens and can misjudge a page whose evidence is spread across chunks
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import textwrap
from collections import defaultdict
from typing import Any, Optional

from hybrid_search import minmax, DEFAULT_ALPHA
from section_search import SectionRetriever

DEFAULT_RERANKER = "BAAI/bge-reranker-base"
DEFAULT_POOL_PAGES = 20


class RerankRetriever(SectionRetriever):
    def __init__(self, *a, reranker: str = DEFAULT_RERANKER,
                 rerank_device: Optional[str] = None, **kw) -> None:
        super().__init__(*a, **kw)
        from sentence_transformers import CrossEncoder

        dev = rerank_device or self._device_guess()
        self.ce = CrossEncoder(reranker, max_length=512, device=dev)
        print(f"      reranker: {reranker} on {dev}")

    @staticmethod
    def _device_guess() -> str:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _pair_text(self, i: int) -> str:
        m = self.metas[i]
        body = m.get("core_text") or self.docs[i]
        title = m.get("title") or ""
        return f"{title}\n{body}" if title else body

    def search_pages(self, query: str, alpha: float = DEFAULT_ALPHA,
                     decay: float = 0.5, max_group: int = 4,
                     pool_pages: int = DEFAULT_POOL_PAGES,
                     rerank: bool = True, blend: float = 0.0
                     ) -> list[tuple[int, float, dict[str, Any]]]:
        """Return [(page, score, best_chunk_meta), ...] ranked."""
        base = self.fused_scores(query, alpha, decay, max_group)

        # retrieval-ranked pages, and the chunks that live on them
        order = sorted(range(len(base)), key=lambda i: -base[i])
        page_rank: list[int] = []
        for i in order:
            p = self.metas[i]["page_number"]
            if p not in page_rank:
                page_rank.append(p)
        pool = page_rank[:pool_pages]
        pool_set = set(pool)

        retr_page_score = {}
        for i in order:
            p = self.metas[i]["page_number"]
            if p in pool_set and p not in retr_page_score:
                retr_page_score[p] = base[i]

        if not rerank:
            return [(p, retr_page_score[p], {}) for p in pool]

        idxs = [i for i in range(len(self.metas))
                if self.metas[i]["page_number"] in pool_set]
        scores = self.ce.predict([(query, self._pair_text(i)) for i in idxs],
                                 show_progress_bar=False)

        best: dict[int, tuple[float, int]] = {}
        for i, s in zip(idxs, scores):
            p = self.metas[i]["page_number"]
            if p not in best or s > best[p][0]:
                best[p] = (float(s), i)

        pages = list(best)
        ce_norm = dict(zip(pages, minmax([best[p][0] for p in pages])))
        rt_norm = dict(zip(pages, minmax([retr_page_score[p] for p in pages])))
        final = {p: (1 - blend) * ce_norm[p] + blend * rt_norm[p] for p in pages}

        return sorted(((p, final[p], self.metas[best[p][1]]) for p in pages),
                      key=lambda x: -x[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+")
    ap.add_argument("--db", default="./chroma")
    ap.add_argument("--collection", default="max77751_v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool-pages", type=int, default=DEFAULT_POOL_PAGES)
    ap.add_argument("--blend", type=float, default=0.0)
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()

    q = " ".join(args.query)
    r = RerankRetriever(args.db, args.collection, device=args.device)
    for rank, (p, s, m) in enumerate(
            r.search_pages(q, pool_pages=args.pool_pages, blend=args.blend)[:args.k], 1):
        print(f"\n[{rank}] page {p:>3}  score={s:.3f}  "
              f"section={(m.get('title') or '')[:46]!r}  ({m.get('chunk_type')})")
        body = m.get("core_text") or ""
        print(textwrap.fill(" ".join(body.split())[:200], 74,
                            initial_indent="    ", subsequent_indent="    "))


if __name__ == "__main__":
    main()
