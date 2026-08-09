# Relevant Section Identification — container image
#
# Model weights are NOT baked in. They download on first use, which keeps the
# image to a manageable size and lets the container pass Cloud Run's startup
# probe immediately (uvicorn binds the port straight away; the embedding model
# loads lazily on the first query).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    RSI_SAMPLE_PDF=/app/data/sample.pdf

# tesseract  -> OCR of figures (no text layer to fall back on)
# poppler    -> pdftoppm, used by pdf2image during hi_res parsing
# libgl/glib -> opencv, pulled in by unstructured-inference
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first, from PyTorch's index, so the CUDA wheels are never pulled
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

COPY requirements.txt .
# torch/torchvision are already satisfied above; everything else resolves normally
RUN pip install -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/
COPY chroma/ ./chroma/
COPY sample_v3.json pdf_chunker.py embed_index.py hybrid_search.py \
     section_search.py query_index.py ./

RUN mkdir -p /models /app/app_data

EXPOSE 8080
# Cloud Run injects $PORT; default to 8080 for local runs.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
