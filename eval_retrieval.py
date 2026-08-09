"""
Small retrieval eval over the ChromaDB index.

Each case names the page(s) that actually answer the question, taken from the
source datasheet. Reports recall@1/@3/@5 overall and split by the chunk type of
the top hit, so caption-driven false positives are visible rather than averaged away.

    python eval_retrieval.py
    python eval_retrieval.py --exclude-image-captions
"""

from __future__ import annotations

import argparse

DEFAULT_MODEL = "google/embeddinggemma-300m"

# (query, set of pages that genuinely answer it)
CASES: list[tuple[str, set[int]]] = [
    ("what is the absolute maximum voltage on the CHGIN pin?", {3}),
    ("what resistor value sets the top-off current?", {32}),
    ("what does the STAT pin indicate about charging status?", {37, 38}),
    ("smart power selector switches QCHGIN QHS QLS QBAT", {20}),
    ("USB BC1.2 detected charger type SDP CDP DCP", {29}),
    ("input self-discharge when the charge source is removed", {22}),
    ("package thermal resistance junction to ambient", {3, 4}),
    ("recommended capacitor selection for BYP SYS VDD PVL", {33}),
    ("what is the reverse boost OTG output voltage and current limit?", {21, 27, 31}),
    ("charge timer and safety features during fast charge", {20, 23}),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./chroma")
    ap.add_argument("--collection", default="max77751")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--exclude-image-captions", action="store_true",
                    help="filter image chunks out of the candidate pool")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    import chromadb

    model = SentenceTransformer(args.model, device=args.device)
    coll = chromadb.PersistentClient(path=args.db).get_collection(args.collection)
    where = {"chunk_type": {"$in": ["text", "table"]}} if args.exclude_image_captions else None

    hits = {1: 0, 3: 0, 5: 0}
    top_types: dict[str, int] = {}
    rows = []

    for q, gold in CASES:
        v = model.encode(q, prompt_name="query", normalize_embeddings=True)
        r = coll.query(query_embeddings=[v.tolist()], n_results=5, where=where,
                       include=["metadatas", "distances"])
        metas = r["metadatas"][0]
        pages = [m["page_number"] for m in metas]
        types = [m["chunk_type"] for m in metas]
        sims = [1 - d for d in r["distances"][0]]

        rank = next((i for i, p in enumerate(pages, 1) if p in gold), None)
        for k in (1, 3, 5):
            if rank and rank <= k:
                hits[k] += 1
        top_types[types[0]] = top_types.get(types[0], 0) + 1
        rows.append((q, gold, pages, types, sims, rank))

    n = len(CASES)
    print(f"\ncases: {n}   pool: {'text+table only' if args.exclude_image_captions else 'all types'}")
    print("=" * 78)
    for q, gold, pages, types, sims, rank in rows:
        mark = f"rank {rank}" if rank else "MISS"
        print(f"\n{mark:>7}  {q}")
        print(f"         gold pages {sorted(gold)}   got "
              + ", ".join(f"{t[:3]}/p{p}({s:.2f})" for t, p, s in zip(types, pages, sims)))
    print("\n" + "=" * 78)
    for k in (1, 3, 5):
        print(f"recall@{k}: {hits[k]}/{n} = {hits[k]/n:.0%}")
    print("chunk type of top-1 hit:", top_types)


if __name__ == "__main__":
    main()
