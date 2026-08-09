"""
Hybrid search plus section expansion via the `title` metadata field.

A section's strongest chunk usually ranks fine; its siblings on other pages often
do not, even though the section as a whole is what answers the query. Since the
task asks for *sections and page numbers*, a chunk that belongs to a matched
section is evidence for that section's other pages.

    score[i] = max( own_score[i],  decay * best_score_in_its_title_group )

so a sibling rides just below the best chunk of its section instead of being
scored purely on its own text.

Two guards, both learned from the data:

  * `max_group` — `Typical Operating Characteristics` covers 27 chunks over 5
    pages, a quarter of the index. Expanding a group that large lets one weak hit
    inject five pages, so oversized groups are skipped.
  * 40 of 56 titles are singletons, so expansion is a no-op for most of the index
    and only fires where a section genuinely spans chunks.
"""

from __future__ import annotations

import argparse
import textwrap
from collections import defaultdict

from hybrid_search import HybridRetriever, minmax, DEFAULT_ALPHA

DEFAULT_DECAY = 0.85
DEFAULT_MAX_GROUP = 8


class SectionRetriever(HybridRetriever):
    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.groups: dict[str, list[int]] = defaultdict(list)
        for i, m in enumerate(self.metas):
            t = (m.get("title") or "").strip()
            if t:
                self.groups[t].append(i)
        multi = {t: g for t, g in self.groups.items() if len(g) > 1}
        print(f"      sections: {len(self.groups)} titles, {len(multi)} span >1 chunk")

    def fused_scores(self, query: str, alpha: float = DEFAULT_ALPHA,
                     decay: float = DEFAULT_DECAY,
                     max_group: int = DEFAULT_MAX_GROUP) -> list[float]:
        base = [alpha * c + (1 - alpha) * b for c, b in zip(
            minmax(self.dense_scores(query, "query")),
            minmax(self.bm25_scores(query)))]
        if decay <= 0:
            return base

        out = list(base)
        for title, idxs in self.groups.items():
            if len(idxs) < 2 or len(idxs) > max_group:
                continue
            best = max(base[i] for i in idxs)
            lifted = decay * best
            for i in idxs:
                if lifted > out[i]:
                    out[i] = lifted
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+")
    ap.add_argument("--db", default="./chroma")
    ap.add_argument("--collection", default="max77751_v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--decay", type=float, default=DEFAULT_DECAY)
    ap.add_argument("--max-group", type=int, default=DEFAULT_MAX_GROUP)
    ap.add_argument("-k", type=int, default=8)
    args = ap.parse_args()

    q = " ".join(args.query)
    r = SectionRetriever(args.db, args.collection, device=args.device)
    scores = r.fused_scores(q, args.alpha, args.decay, args.max_group)
    for hit in r.rank(scores, args.k):
        m = hit["meta"]
        body = m.get("core_text") or hit["doc"]
        print(f"\n[{m['chunk_type'].upper():5}] p{m['page_number']:>3} "
              f"score={hit['score']:.3f}  section={m.get('title','')[:46]!r}")
        print(textwrap.fill(" ".join(body.split())[:220], 74,
                            initial_indent="    ", subsequent_indent="    "))


if __name__ == "__main__":
    main()
