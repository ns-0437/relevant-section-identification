"""
Evaluation against eval_gold.yaml, scored the way the task is actually stated:
"identify all relevant pages", so this is set retrieval over PAGES, not chunks.

Chunk hits are collapsed to a ranked page list (first occurrence of each page
wins), then scored with:

  hit@k    - did ANY gold page appear in the top k          (the old, loose metric)
  recall@k - what FRACTION of the gold pages appeared       (what the task asks for)
  nDCG@5   - were complete answers (grade 2) ranked above
             supporting mentions (grade 1)

Tuning and holdout splits are reported separately; holdout queries were never
inspected while choosing anything.

    python eval_v2.py
    python eval_v2.py --sweep
    python eval_v2.py --by-kind
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import yaml

from hybrid_search import HybridRetriever, DEFAULT_ALPHA


def load_gold(path="eval_gold.yaml"):
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data["queries"]


def ranked_pages(hits) -> list[int]:
    """Collapse ranked chunks to ranked unique pages (first occurrence wins)."""
    seen, out = set(), []
    for h in hits:
        p = h["meta"]["page_number"]
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def ndcg_at(pages: list[int], gold: dict[int, int], k: int) -> float:
    gains = [gold.get(p, 0) for p in pages[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(gold.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def evaluate(r, queries, mode, alpha, ks=(1, 3, 5, 10), pool_chunks=40):
    acc = {"hit": defaultdict(list), "rec": defaultdict(list), "ndcg": []}
    per_query = []
    for q in queries:
        gold = {int(k): int(v) for k, v in q["relevant"].items()}
        hits = r.search(q["query"], k=pool_chunks, alpha=alpha, mode=mode)
        pages = ranked_pages(hits)
        for k in ks:
            top = pages[:k]
            acc["hit"][k].append(1.0 if any(p in gold for p in top) else 0.0)
            acc["rec"][k].append(len([p for p in top if p in gold]) / len(gold))
        nd = ndcg_at(pages, gold, 5)
        acc["ndcg"].append(nd)
        first = next((i for i, p in enumerate(pages, 1) if p in gold), None)
        per_query.append((q, pages[:10], first, nd))
    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    summary = {f"hit@{k}": avg(v) for k, v in acc["hit"].items()}
    summary.update({f"R@{k}": avg(v) for k, v in acc["rec"].items()})
    summary["nDCG@5"] = avg(acc["ndcg"])
    return summary, per_query


def show(name, s):
    print(f"{name:<16} "
          f"{s['hit@1']:>7.0%} {s['hit@3']:>7.0%} {s['hit@5']:>7.0%} "
          f"{s['R@5']:>8.0%} {s['R@10']:>8.0%} {s['nDCG@5']:>8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./chroma")
    ap.add_argument("--collection", default="max77751")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gold", default="eval_gold.yaml")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--by-kind", action="store_true")
    ap.add_argument("--failures", action="store_true")
    ap.add_argument("--missing", action="store_true",
                    help="analyse which gold pages fall outside the top 5")
    args = ap.parse_args()

    qs = load_gold(args.gold)
    tune = [q for q in qs if not q.get("holdout")]
    hold = [q for q in qs if q.get("holdout")]
    print(f"\ngold set: {len(qs)} queries  ({len(tune)} tuning / {len(hold)} holdout)")
    kinds = defaultdict(int)
    for q in qs:
        kinds[q.get("kind", "?")] += 1
    print("by kind:", dict(kinds))

    r = HybridRetriever(args.db, args.collection, device=args.device)

    hdr = f"\n{'method':<16} {'hit@1':>7} {'hit@3':>7} {'hit@5':>7} {'R@5':>8} {'R@10':>8} {'nDCG@5':>8}"
    for split_name, split in (("TUNING", tune), ("HOLDOUT", hold), ("ALL", qs)):
        print(f"\n===== {split_name} ({len(split)} queries) =====")
        print(hdr.strip("\n"))
        print("-" * 66)
        for label, mode, a in (("vector", "vector", 1.0),
                               ("bm25", "bm25", 0.0),
                               (f"hybrid a={args.alpha}", "hybrid", args.alpha)):
            s, detail = evaluate(r, split, mode, a)
            show(label, s)
            if split_name == "ALL" and mode == "hybrid":
                all_detail = detail

    if args.by_kind:
        print("\n===== by query kind (hybrid) =====")
        print(f"{'kind':<8} {'n':>3} {'hit@1':>7} {'hit@5':>7} {'R@5':>8} {'R@10':>8} {'nDCG@5':>8}")
        print("-" * 54)
        for kind in ("word", "symbol", "table", "figure", "multi"):
            sub = [q for q in qs if q.get("kind") == kind]
            if not sub:
                continue
            s, _ = evaluate(r, sub, "hybrid", args.alpha)
            print(f"{kind:<8} {len(sub):>3} {s['hit@1']:>7.0%} {s['hit@5']:>7.0%} "
                  f"{s['R@5']:>8.0%} {s['R@10']:>8.0%} {s['nDCG@5']:>8.3f}")

    if args.missing:
        # what does the index actually hold for each page?
        page_mix = defaultdict(lambda: defaultdict(int))
        for m in r.metas:
            page_mix[m["page_number"]][m["chunk_type"]] += 1

        def profile(p):
            mix = page_mix.get(p)
            if not mix:
                return "NOT INDEXED"
            return "/".join(f"{t[:3]}{n}" for t, n in sorted(mix.items()))

        print("\n===== gold pages MISSED at R@5 (hybrid) =====")
        miss_by_kind = defaultdict(list)
        miss_by_grade = defaultdict(int)
        miss_pages = defaultdict(int)
        found_by_grade = defaultdict(int)
        recovered_at_10 = 0
        total_missed = 0
        for q in qs:
            gold = {int(k): int(v) for k, v in q["relevant"].items()}
            hits = r.search(q["query"], k=40, alpha=args.alpha, mode="hybrid")
            pages = ranked_pages(hits)
            top5, top10 = pages[:5], pages[:10]
            missed = [p for p in gold if p not in top5]
            for p in gold:
                (found_by_grade if p in top5 else miss_by_grade)[gold[p]] += 1
            if not missed:
                continue
            total_missed += len(missed)
            recovered_at_10 += sum(1 for p in missed if p in top10)
            miss_by_kind[q.get("kind")].append((q["query"], missed, gold))
            for p in missed:
                miss_pages[p] += 1
            print(f"\n  [{q.get('kind')}] {q['query'][:60]}")
            print(f"     missed: " + ", ".join(
                f"p{p}(grade {gold[p]}, index has {profile(p)})"
                + ("  [recovered by 10]" if p in top10 else "")
                for p in missed))

        print("\n----- summary -----")
        print(f"total gold pages missed at 5: {total_missed}"
              f"   recovered by rank 10: {recovered_at_10}")
        print("missed by grade:", dict(miss_by_grade), " found by grade:", dict(found_by_grade))
        print("misses per kind:", {k: sum(len(m) for _, m, _ in v) for k, v in miss_by_kind.items()})
        print("\nmost-missed pages (page: times missed, index composition):")
        for p, n in sorted(miss_pages.items(), key=lambda x: -x[1])[:10]:
            print(f"  p{p:>2}: missed {n}x   index has {profile(p)}")

    if args.failures:
        print("\n===== worst queries (hybrid, by nDCG@5) =====")
        for q, pages, first, nd in sorted(all_detail, key=lambda x: x[3])[:10]:
            gold = sorted({int(k) for k in q["relevant"]})
            print(f"\n  nDCG={nd:.2f} first_gold_rank={first}  [{q.get('kind')}]")
            print(f"    {q['query']}")
            print(f"    gold {gold}   got {pages}")

    if args.sweep:
        print("\n===== alpha sweep (tuning split only) =====")
        print(f"{'alpha':>6} {'hit@1':>7} {'hit@5':>7} {'R@5':>8} {'R@10':>8} {'nDCG@5':>8}")
        for i in range(11):
            a = i / 10
            s, _ = evaluate(r, tune, "hybrid", a)
            print(f"{a:>6.1f} {s['hit@1']:>7.0%} {s['hit@5']:>7.0%} "
                  f"{s['R@5']:>8.0%} {s['R@10']:>8.0%} {s['nDCG@5']:>8.3f}")


if __name__ == "__main__":
    main()
