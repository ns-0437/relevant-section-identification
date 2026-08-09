# Setup

Python 3.10–3.11 is the safest range (`unstructured-inference` and `torch` wheels
are most reliable there).

## 1. System binaries

Two non-Python dependencies are required before `pip install`.

### Tesseract (OCR — used for image keywords, and by `hi_res` on scanned pages)

| OS | Command |
| --- | --- |
| Windows | Install from [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki), then add `C:\Program Files\Tesseract-OCR` to `PATH` |
| macOS | `brew install tesseract` |
| Debian/Ubuntu | `sudo apt-get install -y tesseract-ocr libtesseract-dev` |

If it isn't on `PATH`, point `pytesseract` at it explicitly near the top of
`pdf_chunker.py`:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### Poppler (PDF → image rasterization for `pdf2image`)

| OS | Command |
| --- | --- |
| Windows | Download [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases), unzip, add its `Library\bin` to `PATH` |
| macOS | `brew install poppler` |
| Debian/Ubuntu | `sudo apt-get install -y poppler-utils` |

Verify both:

```bash
tesseract --version && pdftoppm -v
```

## 2. Python environment

```bash
python -m venv .venv
```

Activate it — `.venv\Scripts\activate` on Windows, `source .venv/bin/activate` elsewhere.

**Install torch first if you want GPU.** The default PyPI wheel is CPU-only on
Windows/Linux, and CPU captioning is slow — measured at ~190 s for a single image
on the 0.5B model.

Check which CUDA builds exist for your torch version before installing, because
`pip install --upgrade torch --index-url .../cuXXX` silently does nothing if that
index's newest torch is older than what you already have:

```bash
pip index versions torch --index-url https://download.pytorch.org/whl/cu126
```

Then pin the exact `+cuXXX` build:

```bash
pip install --force-reinstall torch==2.13.0+cu126 torchvision --index-url https://download.pytorch.org/whl/cu126
```

Pick the CUDA series your driver supports — `nvidia-smi` reports the max CUDA
version. Any CUDA 12.x wheel runs on a 12.x driver (minor-version compatibility),
but a CUDA 13 wheel needs a 580+ driver.

Verify before going further:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Then the rest:

```bash
pip install -r requirements.txt
```

Optional, for `--load-in-4bit` (CUDA only):

```bash
pip install bitsandbytes
```

## 3. Pre-download models (optional but recommended)

The first run otherwise pulls ~14 GB of LLaVA weights mid-pipeline, plus the
`unstructured` layout model.

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('llava-hf/llava-1.5-7b-hf')"
```

```bash
python -c "import nltk; nltk.download('stopwords')"
```

Set `HF_HOME` if you want the cache somewhere other than `~/.cache/huggingface`.

## 4. Pick a LLaVA checkpoint for your hardware

| Hardware | Checkpoint | Flags |
| --- | --- | --- |
| 24 GB+ GPU | `llava-hf/llava-v1.6-mistral-7b-hf` | *(best on charts/diagrams)* |
| 16 GB GPU | `llava-hf/llava-1.5-7b-hf` | default |
| 8–12 GB GPU | `llava-hf/llava-1.5-7b-hf` | `--load-in-4bit` |
| 4–6 GB GPU | `llava-hf/llava-interleave-qwen-0.5b-hf` | *(7B won't fit even in 4-bit)* |
| CPU only | `llava-hf/llava-interleave-qwen-0.5b-hf` | `--device cpu` |
| No captioning needed | — | `--skip-images` |

## 4b. Re-running: cache the layout pass

`strategy="hi_res"` runs a layout model over every page and dominates wall-clock
time on CPU (~15 min for a 40-page datasheet). `--cache-dir` pickles the
partition result, keyed on the PDF's path + size + mtime, so you only pay it once
while iterating on captioning or chunk sizes:

```bash
python pdf_chunker.py report.pdf --cache-dir .cache --out chunks.json
```

## 5. Run

```bash
python pdf_chunker.py path/to/file.pdf --out chunks.json
```

Common variations:

```bash
python pdf_chunker.py report.pdf --load-in-4bit --out chunks.json
```

```bash
python pdf_chunker.py report.pdf --model llava-hf/llava-interleave-qwen-0.5b-hf --device cpu
```

```bash
python pdf_chunker.py report.pdf --skip-images --strip-base64
```

`--strip-base64` keeps the JSON readable while you inspect output; drop it when
you actually want the encoded images in the file.

## 6. Output shape

```json
[
  {
    "chunk_id": "9f2a1c04ab3d5e77",
    "chunk_type": "text",
    "text": "...",
    "metadata": {
      "page_number": 4,
      "title": "3.2 Evaluation Methodology",
      "char_count": 2871
    }
  },
  {
    "chunk_id": "c71b0e9a4d2f6188",
    "chunk_type": "image",
    "text": "A line chart comparing latency across three configurations...\n\nKeywords: latency, throughput, baseline, p95",
    "metadata": {
      "page_number": 5,
      "title": "3.2 Evaluation Methodology",
      "image_base64": "/9j/4AAQSkZJRgABAQ...",
      "image_mime_type": "image/jpeg",
      "caption": "A line chart comparing latency...",
      "ocr_keywords": ["latency", "throughput", "baseline", "p95"],
      "core_text": "A line chart comparing latency..."
    }
  },
  {
    "chunk_id": "3e5d81f0b9c4a262",
    "chunk_type": "table",
    "text": "Table 2: Ablation results (part 1/2)\n\n| Model | Acc | F1 |\n| --- | --- | --- |\n| ... |",
    "metadata": {
      "page_number": 6,
      "title": "4. Results",
      "table_caption": "Table 2: Ablation results",
      "table_part": 1,
      "table_parts_total": 2,
      "text_as_html": "<table>..."
    }
  }
]
```

`core_text` on image/table chunks is the chunk without its 500-char overlap
lead-in, in case you want to embed the clean version.

## Troubleshooting

**`OSError: [WinError 126]` / `detectron2` errors on import** — `unstructured`'s
hi-res path can fall back to ONNX; make sure `unstructured-inference` installed
cleanly. `pip install --force-reinstall unstructured-inference` usually fixes it.

**`TesseractNotFoundError`** — the binary isn't on `PATH`; set `tesseract_cmd` as
shown in step 1.

**`Unable to get page count. Is poppler installed?`** — poppler's `bin` isn't on
`PATH`.

**All tables come out as plain text, no markdown** — `infer_table_structure=True`
only populates `text_as_html` under `strategy="hi_res"`; confirm the layout model
actually loaded (the partition step prints element counts).

**Captions are generic ("a screenshot of a computer")** — the small interleave
checkpoint is weak on diagrams. Move up to `llava-v1.6-mistral-7b-hf` if you can
afford the VRAM.
