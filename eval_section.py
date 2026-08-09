"""Evaluate title-based section expansion against the 45-query gold set."""

from __future__ import annotations

import argparse
from collections import defaultdict

from eval_v2 import load_gold, ranked_pages, ndcg_at
from section_search import SectionRetriever, DEFAULT_DECAY, DEFAULT_MAX_GROUP
from hybrid_search import DEFAULT_ALPHA


def evaluate(r, queries, alpha, decay, max_group, ks=(1, 3, 5, 10, 20)):
    acc = defaultdict(list)
    detail = []
    for q in queries:
        gold = {int(k): int(v) for k, v in q["relevant"].items()}
        pages = ranked_pages(r.rank(
            r.fused_scores(q["query"], alpha, decay, max_group), 104))
        for k in ks:
            top = pages[:k]
            acc[f"hit@{k}"].append(1.0 if any(p in gold for p in top) else 0.0)
            acc[f"R@{k}"].append(len([p for p in top if p in gold]) / len(gold))
        acc["nDCG@5"].append(ndcg_at(pages, gold, 5))
        first = next((i for i, p in enumerate(pages, 1) if p in gold), None)
        detail.append((q, pages[:10], first))
    return {k: sum(v) / len(v) for k, v in acc.items()}, detail


HDR = f"{'setting':<22} {'hit@1':>7} {'hit@3':>7} {'R@5':>7} {'R@10':>7} {'R@20':>7} {'nDCG@5':>8}"


def row(label, s):
    print(f"{label:<22} {s['hit@1']:>7.0%} {s['hit@3']:>7.0%} {s['R@5']:>7.0%} "
          f"{s['R@10']:>7.0%} {s['R@20']:>7.0%} {s['nDCG@5']:>8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./chroma")
    ap.add_argument("--collection", default="max77751_v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--decay", type=float, default=DEFAULT_DECAY)
    ap.add_argument("--max-group", type=int, default=DEFAULT_MAX_GROUP)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--by-kind", action="store_true")
    args = ap.parse_args()

    qs = load_gold()
    tune = [q for q in qs if not q.get("holdout")]
    hold = [q for q in qs if q.get("holdout")]
    r = SectionRetriever(args.db, args.collection, device=args.device)

    for name, split in (("TUNING", tune), ("HOLDOUT", hold), ("ALL", qs)):
        print(f"\n===== {name} ({len(split)}) =====")
        print(HDR)
        print("-" * 72)
        s, _ = evaluate(r, split, args.alpha, 0.0, args.max_group)
        row("hybrid (no expand)", s)
        s, _ = evaluate(r, split, args.alpha, args.decay, args.max_group)
        row(f"+section d={args.decay} g<={args.max_group}", s)

    if args.by_kind:
        print(f"\n===== by kind: hybrid vs +section(d={args.decay}) =====")
        print(f"{'kind':<8} {'n':>3} | {'hit@1':>14} | {'R@5':>14} | {'nDCG@5':>15}")
        print("-" * 62)
        for kind in ("word", "symbol", "table", "figure", "multi"):
            sub = [q for q in qs if q.get("kind") == kind]
            if not sub:
                continue
            a, _ = evaluate(r, sub, args.alpha, 0.0, args.max_group)
            b, _ = evaluate(r, sub, args.alpha, args.decay, args.max_group)
            f = lambda x, y: f"{x:>5.0%} ->{y:>5.0%}"
            print(f"{kind:<8} {len(sub):>3} | {f(a['hit@1'], b['hit@1']):>14} | "
                  f"{f(a['R@5'], b['R@5']):>14} | "
                  f"{a['nDCG@5']:.3f} ->{b['nDCG@5']:.3f}")

    if args.sweep:
        print("\n===== decay x max_group sweep (tuning split, R@5 / nDCG@5) =====")
        print(f"{'decay':>6} " + " ".join(f"{'g<=' + str(g):>16}" for g in (4, 8, 16, 104)))
        for i in range(0, 11, 2):
            d = i / 10
            cells = []
            for g in (4, 8, 16, 104):
                s, _ = evaluate(r, tune, args.alpha, d, g)
                cells.append(f"{s['R@5']:>7.0%}/{s['nDCG@5']:.3f}")
            print(f"{d:>6.1f} " + " ".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
