"""
Query the ChromaDB index built by embed_index.py.

    python query_index.py "what is the maximum CHGIN input voltage?" -k 5
    python query_index.py "charge current vs battery voltage" --type image

EmbeddingGemma is asymmetric: queries must use the `task: search result | query: `
prompt, documents the `title: ... | text: ` one. Mixing them up silently degrades
recall, so the query prompt is applied here via prompt_name="query".
"""

from __future__ import annotations

import argparse
import textwrap

DEFAULT_MODEL = "google/embeddinggemma-300m"


def main() -> None:
    ap = argparse.ArgumentParser(description="Search the chunk index.")
    ap.add_argument("query", nargs="+")
    ap.add_argument("--db", default="./chroma")
    ap.add_argument("--collection", default="max77751")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--type", default=None, choices=["text", "table", "image"],
                    help="restrict to one chunk type")
    ap.add_argument("--page", type=int, default=None, help="restrict to one page")
    ap.add_argument("--chars", type=int, default=420, help="preview length")
    args = ap.parse_args()

    query = " ".join(args.query)

    from sentence_transformers import SentenceTransformer
    import chromadb

    model = SentenceTransformer(args.model, device=args.device)
    vec = model.encode(query, prompt_name="query", normalize_embeddings=True)

    coll = chromadb.PersistentClient(path=args.db).get_collection(args.collection)

    where = None
    clauses = []
    if args.type:
        clauses.append({"chunk_type": args.type})
    if args.page is not None:
        clauses.append({"page_number": args.page})
    if len(clauses) == 1:
        where = clauses[0]
    elif clauses:
        where = {"$and": clauses}

    res = coll.query(
        query_embeddings=[vec.tolist()],
        n_results=args.k,
        where=where,
        include=["metadatas", "documents", "distances"],
    )

    print(f'\nquery: "{query}"   (collection={args.collection}, n={coll.count()})')
    print("=" * 78)
    for rank, (doc, meta, dist) in enumerate(
        zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), 1
    ):
        sim = 1 - dist
        body = meta.get("core_text") or doc
        preview = textwrap.shorten(" ".join(body.split()), args.chars, placeholder=" ...")
        print(f"\n[{rank}] {meta['chunk_type'].upper():5}  page {meta['page_number']:>3}  "
              f"cos={sim:.3f}")
        print(f"    title: {meta.get('title', '')[:70]}")
        if meta.get("ocr_keywords"):
            print(f"    keywords: {meta['ocr_keywords'][:100]}")
        print(textwrap.fill(preview, 74, initial_indent="    ", subsequent_indent="    "))


if __name__ == "__main__":
    main()
