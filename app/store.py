"""
Document registry and retrieval service.

Holds the mapping from document id -> (pdf on disk, chunks json, chroma
collection), keeps one shared embedding model for every document, and runs the
indexing pipeline for uploads in a background thread.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "app_data")
UPLOADS = os.path.join(DATA, "uploads")
CHUNKS = os.path.join(DATA, "chunks")
CHROMA = os.path.join(ROOT, "chroma")
REGISTRY = os.path.join(DATA, "documents.json")

EMBED_MODEL = "google/embeddinggemma-300m"
CAPTION_MODEL = "llava-hf/llava-interleave-qwen-0.5b-hf"

# How many pages to return: keep everything scoring within CUTOFF_RELATIVE of the
# best page, capped. The task asks for "all relevant pages", so a fixed top-k
# would be wrong in both directions -- padding thin answers and truncating broad
# ones. A relative cutoff lets the answer size follow the query.
CUTOFF_RELATIVE = 0.55
MAX_PAGES = 8


@dataclass
class Document:
    doc_id: str
    name: str
    pdf_path: str
    chunks_path: str
    collection: str
    pages: int = 0
    chunks: int = 0
    status: str = "ready"        # ready | indexing | error
    message: str = ""
    progress: str = ""
    is_sample: bool = False
    created: float = field(default_factory=time.time)


class Store:
    def __init__(self) -> None:
        os.makedirs(UPLOADS, exist_ok=True)
        os.makedirs(CHUNKS, exist_ok=True)
        self.docs: dict[str, Document] = {}
        self._retrievers: dict[str, Any] = {}
        self._model = None
        self._lock = threading.Lock()
        self._load_registry()

    # ---------------------------------------------------------------- registry
    def _load_registry(self) -> None:
        if os.path.exists(REGISTRY):
            with open(REGISTRY, encoding="utf-8") as fh:
                for d in json.load(fh):
                    self.docs[d["doc_id"]] = Document(**d)

    def _save_registry(self) -> None:
        with open(REGISTRY, "w", encoding="utf-8") as fh:
            json.dump([asdict(d) for d in self.docs.values()], fh, indent=1)

    def register_sample(self, pdf_path: str, chunks_path: str, collection: str,
                        name: str) -> Optional[Document]:
        """Register the pre-indexed sample document if its files are present."""
        for d in self.docs.values():
            if d.is_sample:
                return d
        if not (os.path.exists(pdf_path) and os.path.exists(chunks_path)):
            print(f"[store] sample not found at {pdf_path}")
            return None
        chunks = json.load(open(chunks_path, encoding="utf-8"))
        pages = max((c["metadata"].get("page_number") or 0) for c in chunks)
        doc = Document(doc_id="sample", name=name, pdf_path=pdf_path,
                       chunks_path=chunks_path, collection=collection,
                       pages=pages, chunks=len(chunks), is_sample=True)
        self.docs[doc.doc_id] = doc
        self._save_registry()
        return doc

    # ------------------------------------------------------------------ models
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[store] loading {EMBED_MODEL} on {dev}")
            self._model = SentenceTransformer(EMBED_MODEL, device=dev)
        return self._model

    def retriever(self, doc_id: str):
        with self._lock:
            if doc_id not in self._retrievers:
                from section_search import SectionRetriever

                doc = self.docs[doc_id]
                self._retrievers[doc_id] = SectionRetriever(
                    CHROMA, doc.collection, model=self.model)
            return self._retrievers[doc_id]

    # ----------------------------------------------------------------- queries
    def search(self, doc_id: str, query: str, max_pages: int = MAX_PAGES,
               cutoff: float = CUTOFF_RELATIVE) -> list[dict[str, Any]]:
        r = self.retriever(doc_id)
        scores = r.fused_scores(query, alpha=0.6, decay=0.5, max_group=4)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])

        pages: dict[int, dict[str, Any]] = {}
        for i in order:
            m = r.metas[i]
            p = m["page_number"]
            entry = pages.setdefault(p, {"page": p, "score": scores[i],
                                         "sections": [], "evidence": []})
            title = (m.get("title") or "").strip()
            if title and title not in entry["sections"]:
                entry["sections"].append(title)
            if len(entry["evidence"]) < 3:
                body = m.get("core_text") or r.docs[i]
                entry["evidence"].append({
                    "type": m["chunk_type"],
                    "title": title,
                    "snippet": " ".join(body.split())[:400],
                    "score": round(float(scores[i]), 4),
                })

        ranked = sorted(pages.values(), key=lambda e: -e["score"])
        if not ranked:
            return []
        top = ranked[0]["score"]
        keep = [e for e in ranked[:max_pages] if e["score"] >= cutoff * top]
        for e in keep:
            e["score"] = round(float(e["score"]), 4)
        return keep

    def top_chunks(self, doc_id: str, query: str, k: int = 4) -> list[dict[str, Any]]:
        """Evidence chunks for the chat answer."""
        r = self.retriever(doc_id)
        scores = r.fused_scores(query, alpha=0.6, decay=0.5, max_group=4)
        out = []
        for hit in r.rank(scores, k):
            m = hit["meta"]
            out.append({
                "page": m["page_number"],
                "title": (m.get("title") or "").strip(),
                "type": m["chunk_type"],
                "text": m.get("core_text") or hit["doc"],
            })
        return out

    # ------------------------------------------------------------------ upload
    def start_upload(self, filename: str, blob: bytes) -> Document:
        doc_id = uuid.uuid4().hex[:12]
        pdf_path = os.path.join(UPLOADS, f"{doc_id}.pdf")
        with open(pdf_path, "wb") as fh:
            fh.write(blob)

        doc = Document(doc_id=doc_id, name=filename, pdf_path=pdf_path,
                       chunks_path=os.path.join(CHUNKS, f"{doc_id}.json"),
                       collection=f"doc_{doc_id}", status="indexing",
                       progress="queued")
        try:
            from pypdf import PdfReader
            doc.pages = len(PdfReader(pdf_path).pages)
        except Exception:
            pass

        self.docs[doc_id] = doc
        self._save_registry()
        threading.Thread(target=self._index, args=(doc_id,), daemon=True).start()
        return doc

    def _index(self, doc_id: str) -> None:
        doc = self.docs[doc_id]
        try:
            from pdf_chunker import chunk_pdf
            import chromadb
            from embed_index import doc_input, flatten_metadata

            doc.progress = "parsing pages and extracting tables/figures"
            chunks = chunk_pdf(doc.pdf_path, llava_model=CAPTION_MODEL,
                               cache_dir=os.path.join(DATA, "cache"))
            payload = [c.to_dict() for c in chunks]
            with open(doc.chunks_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            doc.chunks = len(payload)

            doc.progress = f"embedding {len(payload)} chunks"
            vectors = self.model.encode([doc_input(c) for c in payload],
                                        batch_size=8, normalize_embeddings=True)

            client = chromadb.PersistentClient(path=CHROMA)
            try:
                client.delete_collection(doc.collection)
            except Exception:
                pass
            coll = client.get_or_create_collection(
                name=doc.collection,
                metadata={"hnsw:space": "cosine", "embedding_model": EMBED_MODEL})
            step = 16
            ids = [c["chunk_id"] for c in payload]
            metas = [flatten_metadata(c, keep_b64=False) for c in payload]
            texts = [c["text"] for c in payload]
            for i in range(0, len(ids), step):
                coll.upsert(ids=ids[i:i + step],
                            embeddings=[v.tolist() for v in vectors[i:i + step]],
                            documents=texts[i:i + step],
                            metadatas=metas[i:i + step])

            doc.status, doc.progress, doc.message = "ready", "", ""
        except Exception as exc:
            doc.status = "error"
            doc.message = f"{type(exc).__name__}: {exc}"
            doc.progress = ""
            traceback.print_exc()
        finally:
            self._save_registry()


store = Store()
