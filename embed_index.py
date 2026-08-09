"""
Embed pdf_chunker.py output with EmbeddingGemma and load it into ChromaDB.

    python embed_index.py sample_full.json --db ./chroma --collection max77751

EmbeddingGemma is prompt-conditioned: documents are encoded as
`title: {title} | text: {content}` and queries as `task: search result | query: {q}`.
Our chunks already carry a section title, so we feed the real title rather than the
`none` placeholder the default prompt uses -- that is what the title slot is for.

The model is gated on HuggingFace. Accept the licence at
https://huggingface.co/google/embeddinggemma-300m and authenticate
(`huggingface-cli login`, or set HF_TOKEN) before running.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

DEFAULT_MODEL = "google/embeddinggemma-300m"

# Chroma metadata values must be scalars, so lists get flattened and anything
# that isn't a str/int/float/bool is dropped.
SCALAR = (str, int, float, bool)


def doc_input(chunk: dict[str, Any]) -> str:
    """Format a chunk the way EmbeddingGemma expects a document."""
    title = (chunk["metadata"].get("title") or "none").strip() or "none"
    return f"title: {title} | text: {chunk['text']}"


def flatten_metadata(chunk: dict[str, Any], keep_b64: bool) -> dict[str, Any]:
    m = chunk["metadata"]
    out: dict[str, Any] = {
        "chunk_type": chunk["chunk_type"],
        "page_number": m.get("page_number") or -1,
        "title": m.get("title") or "",
        "char_count": len(chunk["text"]),
    }

    for key in ("table_caption", "table_part", "table_parts_total",
                "caption", "image_mime_type", "overlap_prefix_chars"):
        v = m.get(key)
        if isinstance(v, SCALAR):
            out[key] = v

    if m.get("ocr_keywords"):
        out["ocr_keywords"] = ", ".join(m["ocr_keywords"])

    # core_text = the chunk without its carried overlap; useful when displaying a hit
    if m.get("core_text"):
        out["core_text"] = m["core_text"]

    if keep_b64 and m.get("image_base64"):
        out["image_base64"] = m["image_base64"]

    return out


def load_model(model_name: str, device: str | None, dim: int | None):
    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = {}
    if device:
        kwargs["device"] = device
    if dim:
        kwargs["truncate_dim"] = dim          # Matryoshka: 768 -> 512/256/128

    try:
        model = SentenceTransformer(model_name, **kwargs)
    except Exception as exc:
        msg = str(exc)
        if "gated" in msg.lower() or "401" in msg or "403" in msg:
            sys.exit(
                f"\n{model_name} is gated.\n"
                "  1. Accept the licence: https://huggingface.co/google/embeddinggemma-300m\n"
                "  2. Authenticate:       huggingface-cli login   (or set HF_TOKEN)\n"
            )
        raise

    print(f"      prompts advertised by the checkpoint: {getattr(model, 'prompts', {})}")
    # renamed in sentence-transformers 5.x; keep working on both
    get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    print(f"      max_seq_length={model.max_seq_length}  dim={get_dim()}")
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed chunks and load them into ChromaDB.")
    ap.add_argument("chunks_json")
    ap.add_argument("--db", default="./chroma", help="persistent Chroma directory")
    ap.add_argument("--collection", default="pdf_chunks")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--dim", type=int, default=None,
                    help="Matryoshka truncation (768 default, or 512/256/128)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--no-b64", action="store_true",
                    help="keep image_base64 out of the Chroma metadata")
    ap.add_argument("--reset", action="store_true", help="drop the collection first")
    args = ap.parse_args()

    chunks = json.load(open(args.chunks_json, encoding="utf-8"))
    print(f"[1/4] loaded {len(chunks)} chunks from {args.chunks_json}")

    print(f"[2/4] loading {args.model}...")
    model = load_model(args.model, args.device, args.dim)

    inputs = [doc_input(c) for c in chunks]
    print(f"[3/4] embedding {len(inputs)} documents...")
    vectors = model.encode(
        inputs,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"      vectors: {vectors.shape}")

    import chromadb

    client = chromadb.PersistentClient(path=args.db)
    if args.reset:
        try:
            client.delete_collection(args.collection)
            print(f"      dropped existing collection {args.collection!r}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine", "embedding_model": args.model},
    )

    ids = [c["chunk_id"] for c in chunks]
    if len(set(ids)) != len(ids):
        sys.exit("duplicate chunk_id values -- refusing to insert")

    metas = [flatten_metadata(c, keep_b64=not args.no_b64) for c in chunks]
    docs = [c["text"] for c in chunks]

    print(f"[4/4] upserting into {args.collection!r} at {os.path.abspath(args.db)}...")
    # Chroma caps payload size per call; base64 metadata makes these rows fat.
    step = 16
    for i in range(0, len(ids), step):
        collection.upsert(
            ids=ids[i:i + step],
            embeddings=[v.tolist() for v in vectors[i:i + step]],
            documents=docs[i:i + step],
            metadatas=metas[i:i + step],
        )

    print(f"\ncollection count: {collection.count()}")
    by_type: dict[str, int] = {}
    for m in metas:
        by_type[m["chunk_type"]] = by_type.get(m["chunk_type"], 0) + 1
    print("by type:", by_type)


if __name__ == "__main__":
    main()
