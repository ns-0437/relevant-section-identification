"""Evaluate cross-encoder reranking over a top-N page pool."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from collections import defaultdict

from evaluation.eval_v2 import load_gold, ndcg_at
from experiments.rerank_search import RerankRetriever, DEFAULT_POOL_PAGES


def evaluate(r, queries, pool_pages, rerank, blend, ks=(1, 3, 5, 10)):
    acc = defaultdict(list)
    detail = []
    for q in queries:
        gold = {int(k): int(v) for k, v in q["relevant"].items()}
        ranked = r.search_pages(q["query"], pool_pages=pool_pages,
                                rerank=rerank, blend=blend)
        pages = [p for p, _, _ in ranked]
        for k in ks:
            top = pages[:k]
            acc[f"hit@{k}"].append(1.0 if any(p in gold for p in top) else 0.0)
            acc[f"R@{k}"].append(len([p for p in top if p in gold]) / len(gold))
        acc["nDCG@5"].append(ndcg_at(pages, gold, 5))
        first = next((i for i, p in enumerate(pages, 1) if p in gold), None)
        detail.append((q, pages[:10], first))
    return {k: sum(v) / len(v) for k, v in acc.items()}, detail


HDR = (f"{'setting':<22} {'hit@3':>7} {'hit@5':>7} {'hit@10':>7} "
       f"{'R@3':>7} {'R@5':>7} {'R@10':>7} {'nDCG@5':>8}")


def row(label, s):
    print(f"{label:<22} {s['hit@3']:>7.0%} {s['hit@5']:>7.0%} {s['hit@10']:>7.0%} "
          f"{s['R@3']:>7.0%} {s['R@5']:>7.0%} {s['R@10']:>7.0%} {s['nDCG@5']:>8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./chroma")
    ap.add_argument("--collection", default="max77751_v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool-pages", type=int, default=DEFAULT_POOL_PAGES)
    ap.add_argument("--blend", type=float, default=0.0)
    ap.add_argument("--by-kind", action="store_true")
    ap.add_argument("--blend-sweep", action="store_true")
    ap.add_argument("--failures", action="store_true")
    args = ap.parse_args()

    qs = load_gold()
    tune = [q for q in qs if not q.get("holdout")]
    hold = [q for q in qs if q.get("holdout")]
    r = RerankRetriever(args.db, args.collection, device=args.device)

    for name, split in (("TUNING", tune), ("HOLDOUT", hold), ("ALL", qs)):
        print(f"\n===== {name} ({len(split)}) =====")
        print(HDR)
        print("-" * 78)
        s0, _ = evaluate(r, split, args.pool_pages, False, 0.0)
        row("retrieval only", s0)
        s1, d1 = evaluate(r, split, args.pool_pages, True, args.blend)
        row(f"+rerank blend={args.blend}", s1)
        if name == "ALL":
            all_detail = d1

    if args.by_kind:
        print(f"\n===== by kind: retrieval vs +rerank (pool={args.pool_pages}) =====")
        print(f"{'kind':<8} {'n':>3} | {'hit@3':>14} | {'R@5':>14} | {'nDCG@5':>15}")
        print("-" * 62)
        for kind in ("word", "symbol", "table", "figure", "multi"):
            sub = [q for q in qs if q.get("kind") == kind]
            if not sub:
                continue
            a, _ = evaluate(r, sub, args.pool_pages, False, 0.0)
            b, _ = evaluate(r, sub, args.pool_pages, True, args.blend)
            f = lambda x, y: f"{x:>5.0%} ->{y:>5.0%}"
            print(f"{kind:<8} {len(sub):>3} | {f(a['hit@3'], b['hit@3']):>14} | "
                  f"{f(a['R@5'], b['R@5']):>14} | "
                  f"{a['nDCG@5']:.3f} ->{b['nDCG@5']:.3f}")

    if args.blend_sweep:
        print("\n===== blend sweep (0 = pure reranker, 1 = pure retrieval) =====")
        print(f"{'blend':>6} " + HDR[22:])
        for i in range(0, 11, 2):
            b = i / 10
            s, _ = evaluate(r, tune, args.pool_pages, True, b)
            print(f"{b:>6.1f} {s['hit@3']:>7.0%} {s['hit@5']:>7.0%} {s['hit@10']:>7.0%} "
                  f"{s['R@3']:>7.0%} {s['R@5']:>7.0%} {s['R@10']:>7.0%} {s['nDCG@5']:>8.3f}")

    if args.failures:
        print("\n===== worst after rerank =====")
        for q, pages, first in sorted(all_detail, key=lambda x: -(x[2] or 99))[:8]:
            gold = sorted({int(k) for k in q["relevant"]})
            print(f"  first_gold={first}  [{q.get('kind')}] {q['query'][:52]}")
            print(f"     gold {gold}  got {pages[:8]}")


if __name__ == "__main__":
    main()
