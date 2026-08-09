# Relevant Section Identification

**Live app:** <https://relevant-section-identification-qqgcuwkcmq-uc.a.run.app>
**Source:** <https://github.com/ns-0437/relevant-section-identification>

> **First load may take ~75 seconds.** The service scales to zero and downloads
> the embedding model on a cold start; after that queries return in well under a
> second. The first chat answer additionally pulls a 3 GB model (~2 min). If it
> looks stuck, it isn't — see [Deployment](#deployment).

Given a PDF and a natural-language query, identify the **pages and section
headings** that contain the information needed to answer it.

Ships as a web app: PDF viewer on the left, query on the right. Results are a
ranked list of pages with their section headings; clicking one opens that page in
the viewer. A bonus tab answers the question in prose from the retrieved pages,
using a local open-weights model.

---

## Approach

The document is parsed with `unstructured` (`hi_res`) into typed elements, then
chunked into three separate kinds so structure is never destroyed: **prose**
chunked by section title (min 1000 / soft 3000 / hard 5000 characters, 500
characters of overlap), **tables** rendered to markdown with their caption first,
and **figures** stored as base64 with a locally generated caption plus OCR
keywords. Every chunk carries its page number and section heading. Chunks are
embedded with **EmbeddingGemma-300m** and stored in **ChromaDB**; retrieval is
**hybrid** — dense cosine fused 60/40 with BM25 over min-max normalised scores —
plus a small section-expansion step that lets a chunk inherit a score from the
best chunk sharing its heading. Chunk hits are then collapsed to a ranked page
list (first occurrence wins), and pages scoring within 55% of the best page are
returned, capped at 8. A fixed top-k would be wrong in both directions here: the
task asks for *all* relevant pages, and how many exist varies per query.

The two changes that actually decided quality were both about extraction, not
ranking. First, `unstructured` infers table structure by **OCRing a rendered image
of the table**, which on a digital PDF throws away a perfect text layer — the
symbol column vanished entirely, and `IBATT_Q` appeared nowhere in the index, so
no amount of retrieval tuning could surface it. Tables are now read with
`pdfplumber` from the text layer. Second, every page carries the running header
`MAX77751 3.15A USB-C Autonomous Charger for 1-Cell Li+ Batteries`; left in, a
query about battery chemistry matches all 40 pages equally, and 57 of 118 chunks
had a running-header line as their "section title". Repeating lines are now
detected generically (normalise, unify digits, drop anything on ≥60% of pages) and
removed. Together these took nDCG@5 from 0.645 to 0.710. Rebuilding the image
chunks — the document's own `Figure N.` caption first, OCR keywords next, the
generated caption last and trimmed, and dropped entirely where OCR found nothing
to anchor it — added another 0.018 and fixed figure-heavy pages outright.

### Assumptions

- Page-level answers are the deliverable; section headings are attached to each
  page rather than being the primary unit, matching the task's wording.
- A page is relevant if **any** chunk on it is relevant (page score = max over its
  chunks).
- The document has a usable text layer for prose and tables. Scanned PDFs fall
  back to OCR automatically but will be markedly worse.
- Open weights only, no API keys, so a reviewer can run everything locally.

### Challenges

- **OCR silently destroying table content** — the hardest failure to see, because
  retrieval looked merely mediocre rather than broken until the missing token was
  traced (§4.2 of `docs/REPORT.md`).
- **Boilerplate as an active correctness problem**, not cosmetic.
- **Hallucinated figure captions.** A 0.5B captioner is what 4 GB of VRAM allows,
  and it invents content. Mitigated by demoting the generated caption beneath
  real text rather than by a bigger model.
- **Things that did not work**, each measured and rejected: HyDE with a 1.5B model
  (nDCG −0.026), a metadata scoring channel (−0.006), and cross-encoder reranking
  (−0.001, tested on two index builds). All three only reweight signal already
  present; the wins all came from repairing what the index contained.

Full numbers, method and limitations: **`docs/REPORT.md`**.

---

## Results

45 hand-labelled queries with graded relevance, 12 held out and never used for any
tuning decision. Scored over pages, as the task defines the output.

| | hit@1 | hit@3 | R@5 | R@10 | R@20 | nDCG@5 |
|---|---|---|---|---|---|---|
| **All 45** | 69% | 93% | 82% | 87% | 95% | **0.728** |
| Holdout (12) | 83% | 100% | 96% | 96% | 100% | 0.859 |

The task's own reference query — *"What battery chemistries are supported by this
charger?"* — returns pages **17 and 19** as its top two, the pages naming Li-ion
and Li-Polymer, and the chat answers *"The charger supports Li-ion and Li-Polymer
batteries."*

---

## Quickstart

### 1. System binaries

`tesseract` (figure OCR) and `poppler` (PDF rasterisation) must be on `PATH`.
See `docs/SETUP.md` for per-OS instructions — the app locates Tesseract automatically
on Windows if it is installed but not on `PATH`.

### 2. Python

```bash
python -m venv .venv && .venv\Scripts\activate
```

GPU users: install a CUDA torch build **before** the rest, pinning the exact
variant (see `docs/SETUP.md` — `pip install --upgrade` silently no-ops here).

```bash
pip install -r requirements.txt
```

### 3. Model access

EmbeddingGemma is gated. Accept the licence at
<https://huggingface.co/google/embeddinggemma-300m>, then:

```bash
hf auth login
```

### 4. Run

```bash
uvicorn app.main:app --port 8000
```

Open <http://localhost:8000>. The sample 40-page datasheet is pre-indexed and
loads by default.

Point the app at a different sample with `RSI_SAMPLE_PDF=/path/to.pdf`.

---

## Using it

- **Find sections** — type a question, get ranked pages with their headings and
  the evidence that matched. Clicking a result opens that page in the viewer.
- **Ask** *(bonus)* — the same retrieval, then `Qwen/Qwen2.5-1.5B-Instruct`
  answers from the retrieved pages with page citations. Citations are clickable.
- **Upload PDF** — indexes a new document in the background. Expect several
  minutes: the `hi_res` layout pass is CPU-bound and every figure is captioned.
  The UI polls and reports progress.

First query after startup takes ~60 s while the embedding model loads; afterwards
queries run in **~0.2 s**. The first chat answer additionally loads the 1.5B model.

---

## Command line

```bash
python pdf_chunker.py sample.pdf --device cuda --cache-dir .cache --out chunks.json
```

```bash
python embed_index.py chunks.json --db ./chroma --collection mydoc --device cuda --reset
```

```bash
python hybrid_search.py "what does the STAT pin indicate?" --alpha 0.6
```

```bash
python -m evaluation.eval_v2 --collection max77751_v3 --by-kind
```

---

## Deployment

Deployed to **Google Cloud Run** (`us-central1`, project `rsi-demo-0437`) from the
`Dockerfile` in this repo, with CI/CD in `.github/workflows/deploy.yml`.

| | |
|---|---|
| URL | <https://relevant-section-identification-qqgcuwkcmq-uc.a.run.app> |
| Instance | 1 × 4 vCPU / 16 GiB, scales to zero |
| Secrets | `HF_TOKEN` injected from Secret Manager at runtime — never in the image or repo |
| CI | unit tests on every push and PR; deploy only on push to `main` |

Measured against the live service:

| | |
|---|---|
| Query, warm | **0.54 – 0.67 s** |
| Query, cold | 74 s (includes a 1.26 GB model download) |
| Chat, cold | 137 s (3.1 GB download + CPU generation) |
| `/api/pdf/sample` | 1,819,441 bytes, byte-identical to the source |

### Known limits of the deployed instance

These are properties of the deployment, not of the pipeline:

- **Cold starts.** Model weights are not baked into the image, so a reclaimed
  instance re-downloads them. Warm the URL before demoing.
- **Chat is slow.** A 1.5B model in fp32 on 4 vCPUs. Correct, not fast.
- **Uploading a new PDF will not finish in the cloud.** Parsing plus figure
  captioning is ~20 minutes of CPU on a background thread with no request holding
  the instance open, so Cloud Run can reclaim it mid-index. Upload works locally;
  the cloud demo path is the pre-indexed sample. Fixing this properly needs a job
  queue (Cloud Tasks + a Cloud Run Job), which is out of scope here.

### Tearing it down

To remove the deployed service and its image:

```bash
gcloud run services delete relevant-section-identification --region us-central1 --project rsi-demo-0437
```

```bash
gcloud artifacts repositories delete cloud-run-source-deploy --location us-central1 --project rsi-demo-0437
```

## Layout

| Path | Purpose |
|---|---|
| `app/main.py` | FastAPI routes |
| `app/store.py` | document registry, retrieval, background indexing |
| `app/rag.py` | grounded answering (bonus) |
| `app/static/` | UI (vanilla JS + PDF.js) |
| `pdf_chunker.py` | PDF → text/table/figure chunks |
| `embed_index.py` | embed → ChromaDB |
| `hybrid_search.py`, `section_search.py` | retrieval |
| `experiments/` | HyDE, metadata channel, cross-encoder rerank — measured and rejected |
| `evaluation/` | 45-query graded gold set and the scoring harness |
| `tests/` | model-free unit tests, run by CI on every push |
| `docs/REPORT.md` | full technical report |
| `docs/SETUP.md` | system binaries, CUDA wheels, model access |
