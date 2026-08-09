"""
Evaluate HyDE against the 45-query gold set.

Three score components, each min-max normalised over the whole collection:

    n_cos_q  cosine( query,     chunk )   -- query prompt
    n_bm_q   BM25  ( query,     chunk )
    n_cos_h  cosine( hyde-doc,  chunk )   -- DOCUMENT prompt, it is pretending to
                                             be a passage, not a question

    score = (1-beta) * [ alpha*n_cos_q + (1-alpha)*n_bm_q ]  +  beta * n_cos_h

beta = 0 reproduces the hybrid baseline; beta = 1 is HyDE-dense only.

    python eval_hyde.py --sweep
    python eval_hyde.py --beta 0.5 --by-kind
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from eval_v2 import load_gold, ranked_pages, ndcg_at
from hybrid_search import HybridRetriever, minmax, DEFAULT_ALPHA
from hyde import HydeGenerator, DEFAULT_LLM


def evaluate(r, gen, queries, alpha, beta, ks=(1, 3, 5, 10), pool=40):
    acc = defaultdict(list)
    detail = []
    for q in queries:
        gold = {int(k): int(v) for k, v in q["relevant"].items()}
        text = q["query"]

        n_cos_q = minmax(r.dense_scores(text, "query"))
        n_bm_q = minmax(r.bm25_scores(text))
        if beta > 0:
            n_cos_h = minmax(r.dense_scores(gen.generate(text), "document"))
        else:
            n_cos_h = [0.0] * len(n_cos_q)

        fused = [(1 - beta) * (alpha * c + (1 - alpha) * b) + beta * h
                 for c, b, h in zip(n_cos_q, n_bm_q, n_cos_h)]
        pages = ranked_pages(r.rank(fused, pool))

        for k in ks:
            top = pages[:k]
            acc[f"hit@{k}"].append(1.0 if any(p in gold for p in top) else 0.0)
            acc[f"R@{k}"].append(len([p for p in top if p in gold]) / len(gold))
        acc["nDCG@5"].append(ndcg_at(pages, gold, 5))
        first = next((i for i, p in enumerate(pages, 1) if p in gold), None)
        detail.append((q, pages[:10], first))
    return {k: sum(v) / len(v) for k, v in acc.items()}, detail


HDR = f"{'setting':<20} {'hit@1':>7} {'hit@3':>7} {'R@5':>7} {'R@10':>7} {'nDCG@5':>8}"


def row(label, s):
    print(f"{label:<20} {s['hit@1']:>7.0%} {s['hit@3']:>7.0%} "
          f"{s['R@5']:>7.0%} {s['R@10']:>7.0%} {s['nDCG@5']:>8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./chroma")
    ap.add_argument("--collection", default="max77751_v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--llm", default=DEFAULT_LLM)
    ap.add_argument("--llm-device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--by-kind", action="store_true")
    ap.add_argument("--show", type=int, default=0, help="print N sample HyDE passages")
    args = ap.parse_args()

    qs = load_gold()
    tune = [q for q in qs if not q.get("holdout")]
    hold = [q for q in qs if q.get("holdout")]
    r = HybridRetriever(args.db, args.collection, device=args.device)
    gen = HydeGenerator(args.llm, args.llm_device)

    print(f"\ngold: {len(qs)} queries ({len(tune)} tuning / {len(hold)} holdout)")
    print(f"retriever collection: {args.collection}   LLM: {args.llm}")

    for name, split in (("TUNING", tune), ("HOLDOUT", hold), ("ALL", qs)):
        print(f"\n===== {name} ({len(split)}) =====")
        print(HDR)
        print("-" * 60)
        for label, b in (("hybrid (beta=0)", 0.0),
                         (f"+hyde beta={args.beta}", args.beta),
                         ("hyde only (beta=1)", 1.0)):
            s, _ = evaluate(r, gen, split, args.alpha, b)
            row(label, s)

    if args.by_kind:
        print(f"\n===== by kind: hybrid vs +hyde(beta={args.beta}) =====")
        print(f"{'kind':<8} {'n':>3} | {'hit@1':>13} | {'R@5':>13} | {'R@10':>13}")
        print("-" * 60)
        for kind in ("word", "symbol", "table", "figure", "multi"):
            sub = [q for q in qs if q.get("kind") == kind]
            if not sub:
                continue
            a, _ = evaluate(r, gen, sub, args.alpha, 0.0)
            b, _ = evaluate(r, gen, sub, args.alpha, args.beta)
            f = lambda x, y: f"{x:>5.0%} ->{y:>5.0%}"
            print(f"{kind:<8} {len(sub):>3} | {f(a['hit@1'], b['hit@1']):>13} | "
                  f"{f(a['R@5'], b['R@5']):>13} | {f(a['R@10'], b['R@10']):>13}")

    if args.sweep:
        print("\n===== beta sweep (tuning split) =====")
        print(f"{'beta':>5} {'hit@1':>7} {'hit@3':>7} {'R@5':>7} {'R@10':>7} {'nDCG@5':>8}")
        for i in range(0, 11, 2):
            b = i / 10
            s, _ = evaluate(r, gen, tune, args.alpha, b)
            print(f"{b:>5.1f} {s['hit@1']:>7.0%} {s['hit@3']:>7.0%} "
                  f"{s['R@5']:>7.0%} {s['R@10']:>7.0%} {s['nDCG@5']:>8.3f}")

    if args.show:
        print("\n===== sample HyDE passages =====")
        for q in qs[:args.show]:
            print(f"\nQ: {q['query']}")
            print(f"H: {gen.generate(q['query'])[:300]}")


if __name__ == "__main__":
    main()
