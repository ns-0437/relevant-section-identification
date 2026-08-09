"""
Compare vector-only, BM25-only and hybrid retrieval on the labelled query set.

    python eval_hybrid.py
    python eval_hybrid.py --sweep        # alpha 0.0 .. 1.0
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from evaluation.eval_retrieval import CASES
from hybrid_search import HybridRetriever, DEFAULT_ALPHA


def score(r: HybridRetriever, mode: str, alpha: float, k_max: int = 5):
    hits = {1: 0, 3: 0, 5: 0}
    detail = []
    for q, gold in CASES:
        res = r.search(q, k=k_max, alpha=alpha, mode=mode)
        pages = [h["meta"]["page_number"] for h in res]
        types = [h["meta"]["chunk_type"] for h in res]
        rank = next((i for i, p in enumerate(pages, 1) if p in gold), None)
        for k in (1, 3, 5):
            if rank and rank <= k:
                hits[k] += 1
        detail.append((q, gold, pages, types, rank))
    n = len(CASES)
    return {k: v / n for k, v in hits.items()}, detail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./chroma")
    ap.add_argument("--collection", default="max77751")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    r = HybridRetriever(args.db, args.collection, device=args.device)
    n = len(CASES)
    print(f"\n{n} labelled queries, {len(r.ids)} chunks\n")

    rows = []
    for label, mode, a in (
        ("vector only", "vector", 1.0),
        ("bm25 only", "bm25", 0.0),
        (f"hybrid a={args.alpha}", "hybrid", args.alpha),
    ):
        rec, detail = score(r, mode, a)
        rows.append((label, rec))
        if mode == "hybrid":
            hybrid_detail = detail
        elif mode == "vector":
            vector_detail = detail

    print(f"{'method':<18} {'recall@1':>9} {'recall@3':>9} {'recall@5':>9}")
    print("-" * 48)
    for label, rec in rows:
        print(f"{label:<18} {rec[1]:>8.0%} {rec[3]:>9.0%} {rec[5]:>9.0%}")

    print("\nper-query rank (gold page found at position; '-' = not in top 5)")
    print("-" * 78)
    print(f"{'query':<52} {'vec':>5} {'hyb':>5}")
    for (q, gold, _, _, rv), (_, _, ph, th, rh) in zip(vector_detail, hybrid_detail):
        mark = lambda x: str(x) if x else "-"
        flag = ""
        if (rv or 99) != (rh or 99):
            flag = "  <-- changed"
        print(f"{q[:52]:<52} {mark(rv):>5} {mark(rh):>5}{flag}")

    if args.sweep:
        print("\nalpha sweep (1.0 = pure vector, 0.0 = pure bm25)")
        print("-" * 48)
        print(f"{'alpha':>6} {'recall@1':>9} {'recall@3':>9} {'recall@5':>9}")
        for i in range(11):
            a = i / 10
            rec, _ = score(r, "hybrid", a)
            star = "  *" if abs(a - args.alpha) < 1e-9 else ""
            print(f"{a:>6.1f} {rec[1]:>8.0%} {rec[3]:>9.0%} {rec[5]:>9.0%}{star}")


if __name__ == "__main__":
    main()
