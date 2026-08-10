"""
Relevant Section Identification — API + UI.

    uvicorn app.main:app --port 8000     (run from the pdf-chunker directory)

Endpoints
    GET  /                     the single-page UI
    GET  /api/documents        registered documents and their indexing status
    POST /api/upload           upload a PDF; indexing runs in the background
    GET  /api/documents/{id}   status of one document (poll while indexing)
    GET  /api/pdf/{id}         the PDF bytes, for the viewer
    POST /api/query            {doc_id, query} -> relevant pages + section headings
    POST /api/chat             {doc_id, query} -> grounded answer + citations
"""

from __future__ import annotations

import os
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from app.store import store, MAX_PAGES          # noqa: E402
from app.rag import answerer                    # noqa: E402

SAMPLE_PDF = os.environ.get(
    "RSI_SAMPLE_PDF",
    r"C:\Users\Navin Kumar\Downloads\relevant_section_identification-sample.pdf")
SAMPLE_CHUNKS = os.path.join(ROOT, "sample_v3.json")
SAMPLE_COLLECTION = "max77751_v3"

app = FastAPI(title="Relevant Section Identification")


@app.on_event("startup")
def _startup() -> None:
    doc = store.register_sample(SAMPLE_PDF, SAMPLE_CHUNKS, SAMPLE_COLLECTION,
                                "MAX77751 datasheet (sample)")
    print(f"[api] sample document: {'registered' if doc else 'MISSING'}")


class QueryIn(BaseModel):
    doc_id: str = "sample"
    query: str
    max_pages: int = MAX_PAGES


@app.get("/api/documents")
def list_documents():
    docs = sorted(store.docs.values(), key=lambda d: (not d.is_sample, d.created))
    return [
        {"doc_id": d.doc_id, "name": d.name, "pages": d.pages, "chunks": d.chunks,
         "status": d.status, "progress": d.progress, "message": d.message,
         "is_sample": d.is_sample}
        for d in docs
    ]


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str):
    d = store.docs.get(doc_id)
    if not d:
        raise HTTPException(404, "unknown document")
    return {"doc_id": d.doc_id, "name": d.name, "pages": d.pages,
            "chunks": d.chunks, "status": d.status, "progress": d.progress,
            "message": d.message, "is_sample": d.is_sample}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "please upload a .pdf file")
    blob = await file.read()
    if not blob:
        raise HTTPException(400, "empty file")
    doc = store.start_upload(file.filename, blob)
    return {"doc_id": doc.doc_id, "name": doc.name, "status": doc.status,
            "pages": doc.pages}


@app.get("/api/pdf/{doc_id}")
def serve_pdf(doc_id: str):
    d = store.docs.get(doc_id)
    if not d or not os.path.exists(d.pdf_path):
        raise HTTPException(404, "pdf not found")
    return FileResponse(d.pdf_path, media_type="application/pdf",
                        headers={"Content-Disposition": f'inline; filename="{d.name}"'})


@app.post("/api/query")
def query(body: QueryIn):
    d = store.docs.get(body.doc_id)
    if not d:
        raise HTTPException(404, "unknown document")
    if d.status != "ready":
        raise HTTPException(409, f"document is {d.status}: {d.progress or d.message}")
    q = body.query.strip()
    if not q:
        raise HTTPException(400, "empty query")
    results = store.search(body.doc_id, q, max_pages=body.max_pages)
    return {"query": q, "doc_id": body.doc_id, "results": results}


@app.post("/api/chat")
def chat(body: QueryIn):
    d = store.docs.get(body.doc_id)
    if not d:
        raise HTTPException(404, "unknown document")
    if d.status != "ready":
        raise HTTPException(409, f"document is {d.status}")
    q = body.query.strip()
    if not q:
        raise HTTPException(400, "empty query")
    chunks = store.top_chunks(body.doc_id, q, k=4)
    if not chunks:
        return {"answer": "Nothing in this document matched the question.",
                "citations": []}
    try:
        text = answerer.answer(q, chunks)
    except Exception as exc:
        return JSONResponse(status_code=500,
                            content={"detail": f"{type(exc).__name__}: {exc}"})
    seen, cites = set(), []
    for c in chunks:
        if c["page"] not in seen:
            seen.add(c["page"])
            cites.append({"page": c["page"], "title": c["title"]})
    return {"answer": text, "citations": cites}


app.mount("/", StaticFiles(directory=os.path.join(HERE, "static"), html=True),
          name="static")
