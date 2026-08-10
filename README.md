# Relevant Section Identification

Given a PDF and a natural-language query, identify the **pages and section headings**
that contain the information needed to answer it.

**Live app:** <https://relevant-section-identification-qqgcuwkcmq-uc.a.run.app>

> **First load takes ~70 seconds.** The service scales to zero and downloads its
> embedding model on a cold start. After that, searches return in well under a
> second. If it looks stuck, it isn't — see [Performance](#performance).

The app puts a PDF viewer on the left and a query box on the right. Results are a
ranked list of pages with their section headings and the text that matched;
clicking one opens that page in the viewer. A bonus tab answers the question in
prose from the retrieved pages using a local open-weights model.

---

## Contents

- [Quickstart](#quickstart)
- [Using the app](#using-the-app)
- [Approach](#approach)
- [Results](#results)
- [Performance](#performance)
- [Command line](#command-line)
- [Deployment](#deployment)
- [Project layout](#project-layout)

---

## Quickstart

### 1. System binaries

Two non-Python dependencies must be on `PATH`.

| | Purpose | Install |
|---|---|---|
| **Tesseract** | OCR of figures | Windows: [UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki) · macOS: `brew install tesseract` · Debian: `apt install tesseract-ocr` |
| **Poppler** | PDF rasterisation | Windows: [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases) · macOS: `brew install poppler` · Debian: `apt install poppler-utils` |

Check both:

```bash
tesseract --version && pdftoppm -v
```

On Windows the app finds Tesseract automatically if it is installed but not on
`PATH`. Full per-OS notes are in [`docs/SETUP.md`](docs/SETUP.md).

### 2. Python environment

Python 3.10–3.11 is the safest range.

```bash
python -m venv .venv
```

Activate it — `.venv\Scripts\activate` on Windows, `source .venv/bin/activate` elsewhere.

**If you have an NVIDIA GPU, install torch first.** The default PyPI wheel is
CPU-only, and `pip install --upgrade torch --index-url .../cuXXX` silently does
nothing if that index's newest torch is older than what you already have. Pin the
exact build:

```bash
pip install --force-reinstall torch==2.13.0+cu126 torchvision --index-url https://download.pytorch.org/whl/cu126
```

Then everything else:

```bash
pip install -r requirements.txt
```

### 3. Model access

EmbeddingGemma is a gated model. Accept the licence at
<https://huggingface.co/google/embeddinggemma-300m>, then authenticate:

```bash
hf auth login
```

The other two models — LLaVA for captions and Qwen for the chat tab — are ungated
and download on first use. No API keys are needed anywhere.

### 4. Run

```bash
uvicorn app.main:app --port 8000
```

Open <http://localhost:8000>. The sample 40-page datasheet is pre-indexed and loads
by default, so you can query immediately.

To point the app at a different default document, set `RSI_SAMPLE_PDF=/path/to.pdf`.

### 5. Verify (optional)

```bash
python -m pytest tests/ -q
```

15 model-free tests covering table rendering, split overlap, boilerplate detection
and keyword extraction. They run in about a second.

---

## Using the app

**Find sections** — type a question and get back the pages that answer it, each with
its section headings and the matching evidence. Clicking a result jumps the viewer
to that page. This is the graded deliverable, and it responds in well under a second
once warm.

**Ask** *(bonus)* — the same retrieval, then `Qwen2.5-1.5B-Instruct` writes a short
answer from the retrieved pages with clickable page citations. It is slow on CPU
(30–60 s) and a model this small can misread a dense datasheet, so the cited pages
are shown alongside every answer.

**Upload PDF** — indexes a new document in the background with a progress indicator.
Expect several minutes: the layout pass is CPU-bound and every figure is captioned.
This works locally; on the deployed instance it will not finish (see
[Deployment](#deployment)).

---

## Approach

The document is parsed with `unstructured` using the `hi_res` strategy, which runs a
layout model over every page and types each block as a title, paragraph, table or
image. Those blocks are then chunked into **three kinds that never share a code
path**, so structure is never destroyed: prose is grouped by section title (minimum
1000 characters, soft limit 3000, hard limit 5000, with 500 characters of overlap);
tables are rendered to markdown with their caption first and, if oversized, split so
that every part repeats the header row; figures are stored as base64 alongside a
locally generated caption and stopword-filtered OCR keywords. Every chunk carries
its page number and section heading. Chunks are embedded with
**EmbeddingGemma-300m** and stored in **ChromaDB**. Retrieval is **hybrid** — dense
cosine fused 60/40 with BM25 over min-max normalised scores, since BM25 is unbounded
while cosine sits in a narrow band and a raw weighted sum would be dominated by
BM25's scale whatever the weight said. A small section-expansion step lets a chunk
inherit a score from the best chunk sharing its heading. Chunk hits are then
collapsed to a ranked list of pages, keeping everything within 55% of the top score
up to a maximum of eight; a fixed top-k would be wrong in both directions, because
the task asks for *all* relevant pages and how many exist varies per query.

The two changes that actually decided quality were both about extraction, not
ranking. First, `unstructured` infers table structure by **OCRing a rendered image of
the table**, which on a digital PDF throws away a perfect text layer — the entire
symbol column vanished, and `IBATT_Q` appeared nowhere in the index, so no amount of
retrieval tuning could ever surface it. Tables are now read with `pdfplumber` from
the embedded text layer, with the OCR path kept as a fallback. Second, every page
carries the running header `MAX77751 3.15A USB-C Autonomous Charger for 1-Cell Li+
Batteries`; left in, a query about battery chemistry matches all 40 pages equally,
and 57 of 118 chunks had a running-header line as their supposed section title.
Repeating lines are now detected generically and removed. Together these took
nDCG@5 from 0.645 to 0.710, and rebuilding the figure chunks — the document's own
`Figure N.` caption first, OCR keywords next, the generated caption last and trimmed
— added a further 0.018.

### Assumptions

- Page-level answers are the deliverable; section headings are attached to each page
  rather than being the primary unit, matching the task's wording.
- A page is relevant if **any** chunk on it is relevant, so a page scores as the
  maximum over its chunks.
- The document has a usable text layer for prose and tables. Scanned PDFs fall back
  to OCR automatically but will be noticeably worse.
- Open weights only, no API keys, so a reviewer can run everything locally.

### Challenges

- **OCR silently destroying table content.** The hardest failure to see, because
  retrieval looked merely mediocre rather than broken until a missing token was
  traced back to the extractor.
- **Boilerplate as a correctness problem**, not a tidiness one — a header on every
  page makes the retriever unable to discriminate between pages.
- **Hallucinated figure captions.** A 0.5B captioner is what 4 GB of VRAM allows, and
  it invents content. Mitigated by demoting the generated caption beneath text the
  authors actually wrote, rather than by a bigger model.
- **Four techniques that did not work**, each measured and rejected: HyDE with a 1.5B
  model (nDCG −0.026), a metadata scoring channel (−0.006), cross-encoder reranking
  (−0.001, tested on two index builds), and section expansion which was kept at
  +0.006 despite being marginal. All of them only reweight signal already present;
  every real gain came from repairing what the index contained.

Full method, measurements and limitations: **[`docs/REPORT.md`](docs/REPORT.md)**.

---

## Results

45 queries authored from the source PDF with graded relevance, 12 held out and never
used for any tuning decision. Scored over pages, as the task defines the output.

| | hit@1 | hit@3 | R@5 | R@10 | R@20 | nDCG@5 |
|---|---|---|---|---|---|---|
| **All 45** | 69% | 93% | 82% | 87% | 95% | **0.728** |
| Holdout (12) | 83% | 100% | 96% | 96% | 100% | 0.859 |

The task's own reference query — *"What battery chemistries are supported by this
charger?"* — returns pages **17 and 19** as its top two results, the pages naming
Li-ion and Li-Polymer, and the chat tab answers *"The charger supports Li-ion and
Li-Polymer batteries."*

These numbers come from a query set written by the same person who built the system.
It is a careful evaluation, not an independent benchmark.

---

## Performance

Measured with `benchmark.py` on an RTX 3050 Laptop (4 GB VRAM) with 16 CPU cores.

**Index building** — paid once per document, then cached.

| Stage | GPU | CPU |
|---|---|---|
| Layout parse, 40 pages | — | ~15 min (CPU-bound either way) |
| Captioning 46 figures | ~4 min | 193 s **per image** |
| Embedding 104 chunks | **9.9 s** | 71.3 s |

**Query time** — median over 8 queries, whole collection scored.

| Component | GPU | CPU |
|---|---|---|
| Vector search | 111.2 ms | 121.1 ms |
| Keyword search (BM25) | 0.0 ms | 1.0 ms |
| **End to end, in process** | **116.9 ms** | **119.2 ms** |

**A GPU speeds up indexing about 7× and does nothing measurable at query time.**
Encoding one short query dominates, and the cosine against 104 vectors is trivial on
either device. That is why the deployed service runs CPU-only.

EmbeddingGemma peaks at **1.89 GiB** of VRAM. LLaVA-1.5-7B would need roughly 14 GB
and does not fit on this card even in 4-bit, which is what forced the smaller
captioner.

```bash
python benchmark.py
```

---

## Command line

Everything the app does is available without it.

Chunk a PDF:

```bash
python pdf_chunker.py sample.pdf --device cuda --cache-dir .cache --out chunks.json
```

Embed and index:

```bash
python embed_index.py chunks.json --db ./chroma --collection mydoc --device cuda --reset
```

Search:

```bash
python hybrid_search.py "what does the STAT pin indicate?" --alpha 0.6
```

Evaluate:

```bash
python -m evaluation.eval_v2 --collection max77751_v3 --by-kind
```

Useful flags: `--cache-dir` reuses the expensive layout pass, `--keep-boilerplate`
and `--tables-from ocr` reproduce the earlier behaviour for comparison, and
`--skip-images` skips captioning entirely.

---

## Deployment

Deployed to **Google Cloud Run** (`us-central1`) from the `Dockerfile` in this repo,
with CI/CD in `.github/workflows/deploy.yml`: tests run on every push and pull
request, and deployment happens only on pushes to `main`, so a fork's pull request
can never reach the cloud credentials. The HuggingFace token is injected from Secret
Manager at runtime and appears in neither the image nor the repository.

The instance is 4 vCPU / 16 GiB and scales to zero.

Three limits belong to the deployment rather than the pipeline:

- **Cold starts.** Model weights are not baked into the image, so a reclaimed
  instance re-downloads them. Open the URL once before demonstrating it.
- **Chat is slow.** A 1.5B model in fp32 on 4 vCPUs — correct, but not fast.
- **Uploading a PDF will not finish in the cloud.** Parsing plus captioning is about
  20 minutes of CPU work on a background thread with no request holding the instance
  open, so Cloud Run may reclaim it mid-index. Upload works locally; the cloud demo
  path is the pre-indexed sample.

### Tearing it down

```bash
gcloud run services delete relevant-section-identification --region us-central1 --project rsi-demo-0437
```

```bash
gcloud artifacts repositories delete cloud-run-source-deploy --location us-central1 --project rsi-demo-0437
```

---

## Project layout

| Path | Purpose |
|---|---|
| `pdf_chunker.py` | PDF → text, table and figure chunks |
| `embed_index.py` | Embed chunks → ChromaDB |
| `hybrid_search.py` | Dense + BM25 retrieval with score fusion |
| `section_search.py` | Section expansion over the hybrid scores |
| `query_index.py` | Dense-only search, for comparison |
| `benchmark.py` | GPU/CPU throughput and latency measurements |
| `app/` | FastAPI backend, PDF.js frontend, grounded answering |
| `evaluation/` | 45-query graded gold set and the scoring harness |
| `experiments/` | HyDE, metadata channel, reranking — measured and rejected |
| `tests/` | Model-free unit tests, run by CI on every push |
| `docs/REPORT.md` | Full technical report |
| `docs/SETUP.md` | System binaries, CUDA wheels, model access |
| `data/`, `chroma/`, `sample_v3.json` | Sample PDF, prebuilt index, chunk output |
