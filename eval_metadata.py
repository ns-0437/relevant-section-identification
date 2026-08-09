"""Evaluate the metadata channel against the 45-query gold set."""

from __future__ import annotations

import argparse
from collections import defaultdict

from eval_v2 import load_gold, ranked_pages, ndcg_at
from metadata_search import MetaRetriever, DEFAULT_GAMMA
from hybrid_search import DEFAULT_ALPHA


def evaluate(r, queries, alpha, gamma, ks=(1, 3, 5, 10, 20), pool=104):
    acc = defaultdict(list)
    detail = []
    for q in queries:
        gold = {int(k): int(v) for k, v in q["relevant"].items()}
        pages = ranked_pages(r.rank(r.fused_scores(q["query"], alpha, gamma), pool))
        for k in ks:
            top = pages[:k]
            acc[f"hit@{k}"].append(1.0 if any(p in gold for p in top) else 0.0)
            acc[f"R@{k}"].append(len([p for p in top if p in gold]) / len(gold))
        acc["nDCG@5"].append(ndcg_at(pages, gold, 5))
        first = next((i for i, p in enumerate(pages, 1) if p in gold), None)
        detail.append((q, pages[:10], first))
    return {k: sum(v) / len(v) for k, v in acc.items()}, detail


HDR = f"{'setting':<18} {'hit@1':>7} {'hit@3':>7} {'R@5':>7} {'R@10':>7} {'R@20':>7} {'nDCG@5':>8}"


def row(label, s):
    print(f"{label:<18} {s['hit@1']:>7.0%} {s['hit@3']:>7.0%} {s['R@5']:>7.0%} "
          f"{s['R@10']:>7.0%} {s['R@20']:>7.0%} {s['nDCG@5']:>8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./chroma")
    ap.add_argument("--collection", default="max77751_v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--by-kind", action="store_true")
    args = ap.parse_args()

    qs = load_gold()
    tune = [q for q in qs if not q.get("holdout")]
    hold = [q for q in qs if q.get("holdout")]
    r = MetaRetriever(args.db, args.collection, device=args.device)

    for name, split in (("TUNING", tune), ("HOLDOUT", hold), ("ALL", qs)):
        print(f"\n===== {name} ({len(split)}) =====")
        print(HDR)
        print("-" * 68)
        for label, g in (("content only", 0.0),
                         (f"+metadata g={args.gamma}", args.gamma),
                         ("metadata only", 1.0)):
            s, _ = evaluate(r, split, args.alpha, g)
            row(label, s)

    if args.by_kind:
        print(f"\n===== by kind: content vs +metadata(g={args.gamma}) =====")
        print(f"{'kind':<8} {'n':>3} | {'hit@1':>14} | {'R@5':>14} | {'nDCG@5':>15}")
        print("-" * 62)
        for kind in ("word", "symbol", "table", "figure", "multi"):
            sub = [q for q in qs if q.get("kind") == kind]
            if not sub:
                continue
            a, _ = evaluate(r, sub, args.alpha, 0.0)
            b, _ = evaluate(r, sub, args.alpha, args.gamma)
            f = lambda x, y: f"{x:>5.0%} ->{y:>5.0%}"
            print(f"{kind:<8} {len(sub):>3} | {f(a['hit@1'], b['hit@1']):>14} | "
                  f"{f(a['R@5'], b['R@5']):>14} | "
                  f"{a['nDCG@5']:.3f} ->{b['nDCG@5']:.3f}")

    if args.sweep:
        print("\n===== gamma sweep (tuning split) =====")
        print(f"{'gamma':>6} {'hit@1':>7} {'hit@3':>7} {'R@5':>7} {'R@10':>7} {'nDCG@5':>8}")
        for i in range(0, 11):
            g = i / 10
            s, _ = evaluate(r, tune, args.alpha, g)
            print(f"{g:>6.1f} {s['hit@1']:>7.0%} {s['hit@3']:>7.0%} {s['R@5']:>7.0%} "
                  f"{s['R@10']:>7.0%} {s['nDCG@5']:>8.3f}")


if __name__ == "__main__":
    main()
