"""
Hi-res PDF -> chunk list, using `unstructured` for partitioning.

Produces three kinds of chunks, all in one flat list:

  * text   - `chunk_by_title` output (min 1000 / soft 3000 / hard 5000 chars, 500 overlap)
  * image  - LLaVA caption + OCR keywords, base64 of the image in metadata
  * table  - HTML table rendered to markdown, caption first, split with repeated headers

Every chunk carries `page_number` and `title` in its metadata.

Usage:
    python pdf_chunker.py /path/to/file.pdf --out chunks.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# Locate the Tesseract / poppler binaries.
#
# Both `pytesseract` and `unstructured`'s hi_res OCR shell out to `tesseract`,
# and pdf2image shells out to `pdftoppm`, so it isn't enough to point
# pytesseract at the exe -- the directory has to be on PATH for this process.
# ---------------------------------------------------------------------------

_TESSERACT_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR",
    r"C:\Program Files (x86)\Tesseract-OCR",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR"),
    os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR"),
    "/usr/local/bin",
    "/opt/homebrew/bin",
]


def ensure_binaries_on_path() -> None:
    """Add Tesseract's install dir to PATH if the binary isn't already there."""
    if shutil.which("tesseract"):
        return
    for d in _TESSERACT_CANDIDATES:
        exe = os.path.join(d, "tesseract.exe" if os.name == "nt" else "tesseract")
        if os.path.isfile(exe):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            try:
                import pytesseract

                pytesseract.pytesseract.tesseract_cmd = exe
            except ImportError:
                pass
            return
    print("[warn] tesseract not found -- OCR keywords will be empty")


ensure_binaries_on_path()

# ---------------------------------------------------------------------------
# Chunk sizing knobs (the numbers you asked for)
# ---------------------------------------------------------------------------

MIN_CHARS = 1000      # combine_text_under_n_chars
SOFT_LIMIT = 3000     # new_after_n_chars
HARD_LIMIT = 5000     # max_characters
OVERLAP = 500         # chars of overlap carried between chunks

DEFAULT_LLAVA_MODEL = "llava-hf/llava-1.5-7b-hf"

CAPTION_PROMPT = (
    "Describe this image in detail. If it is a chart, diagram or figure, "
    "state what it shows, the axes or labels, and the trend or relationship "
    "it conveys."
)

# Fallback stopword list, used if NLTK's corpus isn't downloadable.
_FALLBACK_STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "having", "he",
    "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
    "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will",
    "with", "you", "your", "yours", "yourself", "yourselves",
}


# ---------------------------------------------------------------------------
# Chunk container
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    chunk_id: str
    chunk_type: str            # "text" | "image" | "table"
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# Stopwords / keyword extraction
# ---------------------------------------------------------------------------


def _load_stopwords() -> set[str]:
    try:
        import nltk
        from nltk.corpus import stopwords

        try:
            words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = set(stopwords.words("english"))
        return words | _FALLBACK_STOPWORDS
    except Exception:
        return set(_FALLBACK_STOPWORDS)


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_/%\.]*")


def extract_keywords(text: str, stop: set[str], limit: int = 40) -> list[str]:
    """Tokenize OCR output, drop stopwords/noise, keep order of first appearance."""
    seen: dict[str, None] = {}
    for raw in _TOKEN_RE.findall(text or ""):
        tok = raw.strip(".-_/").lower()
        if len(tok) < 3 or tok in stop:
            continue
        if not any(c.isalpha() for c in tok):
            continue
        seen.setdefault(tok, None)
        if len(seen) >= limit:
            break
    return list(seen.keys())


# ---------------------------------------------------------------------------
# Local image captioning (LLaVA)
# ---------------------------------------------------------------------------


class ImageCaptioner:
    """Local LLaVA captioner. Model is loaded lazily on first use.

    Checkpoint guidance:
      * `llava-hf/llava-1.5-7b-hf`               - default, needs ~14GB fp16 GPU (or 4-bit)
      * `llava-hf/llava-interleave-qwen-0.5b-hf` - small, workable on CPU
      * `llava-hf/llava-v1.6-mistral-7b-hf`      - LLaVA-NeXT, better on charts/text
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LLAVA_MODEL,
        device: Optional[str] = None,
        load_in_4bit: bool = False,
        max_new_tokens: int = 120,
    ) -> None:
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self._device = device
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._processor = AutoProcessor.from_pretrained(self.model_name)

        # transformers renamed `torch_dtype` to `dtype` in 4.56 and dropped the
        # old spelling in 5.x, so pick the key the installed version accepts.
        import transformers

        dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"

        kwargs: dict[str, Any] = {}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = "auto"
        elif self._device == "cuda":
            kwargs[dtype_key] = torch.float16
            kwargs["device_map"] = "auto"
        else:
            kwargs[dtype_key] = torch.float32

        self._model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name, **kwargs
        )
        if not self.load_in_4bit and self._device == "cpu":
            self._model.to("cpu")
        self._model.eval()

    def _build_prompt(self) -> str:
        """Prefer the checkpoint's own chat template; fall back to LLaVA-1.5 format."""
        proc = self._processor
        try:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": CAPTION_PROMPT},
                    ],
                }
            ]
            return proc.apply_chat_template(conversation, add_generation_prompt=True)
        except Exception:
            return f"USER: <image>\n{CAPTION_PROMPT} ASSISTANT:"

    def caption(self, image) -> str:
        import torch

        self._ensure_loaded()
        prompt = self._build_prompt()

        inputs = self._processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        decoded = self._processor.batch_decode(out, skip_special_tokens=True)[0]
        return self._strip_prompt_echo(decoded)

    @staticmethod
    def _strip_prompt_echo(decoded: str) -> str:
        """LLaVA echoes the prompt; keep only what follows the assistant turn."""
        for marker in ("ASSISTANT:", "assistant\n", "[/INST]", "<|im_start|>assistant"):
            if marker in decoded:
                decoded = decoded.split(marker)[-1]
        decoded = decoded.replace(CAPTION_PROMPT, "")
        return decoded.strip().lstrip(":").strip()


def ocr_image(image) -> str:
    try:
        import pytesseract

        return pytesseract.image_to_string(image) or ""
    except Exception:
        return ""


def _decode_b64_image(b64: str):
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


# ---------------------------------------------------------------------------
# HTML table -> markdown
# ---------------------------------------------------------------------------


def _cell_text(cell) -> str:
    return " ".join(cell.get_text(" ", strip=True).split()).replace("|", "\\|")


def html_table_to_markdown(html: str) -> str:
    """Render an HTML table as a pipe-style markdown table.

    First row becomes the header (unstructured's hi_res tables often have no <th>).
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    table = soup.find("table") or soup

    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        rows.append([_cell_text(c) for c in cells])

    if not rows:
        return ""

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    header, body = rows[0], rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def rows_to_markdown(rows: list[list[Optional[str]]]) -> str:
    """Render pdfplumber's row/cell lists as a pipe table."""
    clean: list[list[str]] = []
    for row in rows:
        cells = [" ".join((c or "").split()).replace("|", "\\|") for c in row]
        if any(cells):
            clean.append(cells)
    if not clean:
        return ""

    width = max(len(r) for r in clean)
    clean = [r + [""] * (width - len(r)) for r in clean]
    header, body = clean[0], clean[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def extract_text_layer_tables(
    pdf_path: str, boilerplate: set[str]
) -> dict[int, list[str]]:
    """Pull tables straight from the PDF's embedded text layer.

    `unstructured`'s hi_res table inference OCRs a *rendered image* of the table
    region. On a digital PDF that throws away a perfectly good text layer: symbol
    columns come back empty and dot leaders turn into runs of 'e'. When the page
    has real text, reading it directly is strictly better.

    Returns page_number -> [markdown table, ...] in reading order.
    """
    try:
        import pdfplumber
    except ImportError:
        print("[warn] pdfplumber not installed -- falling back to OCR'd tables")
        return {}

    out: dict[int, list[str]] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for pageno, page in enumerate(pdf.pages, 1):
            mds: list[str] = []
            for rows in page.extract_tables():
                # running headers get detected as 2x2 "tables"; drop them
                flat = " ".join(" ".join((c or "") for c in r) for r in rows)
                if _bp_key(flat) in boilerplate or len(rows) < 2:
                    continue
                if all(_bp_key(" ".join((c or "") for c in r)) in boilerplate
                       for r in rows):
                    continue
                md = rows_to_markdown(rows)
                if md:
                    mds.append(md)
            if mds:
                out[pageno] = mds
    return out


def split_markdown_table(
    md_table: str,
    caption: str = "",
    hard_limit: int = HARD_LIMIT,
    overlap: int = OVERLAP,
) -> list[str]:
    """Split a markdown table into <= hard_limit pieces.

    Each piece repeats the header + separator row, is prefixed with the caption,
    and carries ~`overlap` chars of trailing rows from the previous piece.
    """
    lines = md_table.split("\n")
    if len(lines) < 2:
        return [f"{caption}\n\n{md_table}".strip()] if md_table else []

    header, sep, body = lines[0], lines[1], lines[2:]
    prefix = (caption.strip() + "\n\n" if caption.strip() else "") + header + "\n" + sep + "\n"

    whole = prefix + "\n".join(body)
    if len(whole) <= hard_limit:
        return [whole]

    def tail_rows(rows: list[str]) -> list[str]:
        """Last rows of a split, up to ~`overlap` chars, to repeat in the next split."""
        tail: list[str] = []
        size = 0
        for row in reversed(rows):
            if tail and size + len(row) + 1 > overlap:
                break
            tail.insert(0, row)
            size += len(row) + 1
        return tail

    splits: list[str] = []
    budget = hard_limit - len(prefix)
    current: list[str] = []
    fresh = 0            # rows added since the last flush (excludes carried-over rows)

    for row in body:
        if fresh and len("\n".join(current + [row])) > budget:
            splits.append(prefix + "\n".join(current))
            current = tail_rows(current)
            fresh = 0
        current.append(row)
        fresh += 1

    if fresh:
        splits.append(prefix + "\n".join(current))

    total = len(splits)
    return [
        s.replace(caption.strip(), f"{caption.strip()} (part {i + 1}/{total})", 1)
        if caption.strip()
        else s
        for i, s in enumerate(splits)
    ]


# ---------------------------------------------------------------------------
# Element helpers
# ---------------------------------------------------------------------------


_DIGITS = re.compile(r"\d+")


def _bp_key(text: str) -> str:
    """Normalise a line for boilerplate comparison (page numbers vary, text doesn't)."""
    t = " ".join((text or "").split()).lower()
    return _DIGITS.sub("#", t)


def detect_boilerplate(elements: list, ratio: float = 0.6,
                       min_pages: int = 3) -> set[str]:
    """Find running headers/footers: short text repeating across most pages.

    On this datasheet every page carries `MAX77751 3.15A USB-C Autonomous Charger
    for 1-Cell Li+ Batteries`. Left in, a query about battery chemistry matches all
    40 pages equally and the retriever has nothing to discriminate on -- so this is
    a correctness fix, not tidying.
    """
    from collections import defaultdict

    seen: dict[str, set[int]] = defaultdict(set)
    pages: set[int] = set()
    for el in elements:
        p = _page(el)
        if p is None:
            continue
        pages.add(p)
        if _category(el) in ("Table", "Image"):
            continue
        key = _bp_key(getattr(el, "text", "") or "")
        if not key or len(key) > 160:
            continue
        seen[key].add(p)

    if len(pages) < min_pages:
        return set()
    threshold = max(min_pages, ratio * len(pages))
    return {k for k, ps in seen.items() if len(ps) >= threshold}


def strip_boilerplate(elements: list, boilerplate: set[str]) -> list:
    """Drop boilerplate text elements; never drop Tables or Images."""
    kept = []
    for el in elements:
        if _category(el) in ("Table", "Image"):
            kept.append(el)
            continue
        if _bp_key(getattr(el, "text", "") or "") in boilerplate:
            continue
        kept.append(el)
    return kept


def _category(el) -> str:
    return getattr(el, "category", None) or type(el).__name__


def _page(el) -> Optional[int]:
    return getattr(getattr(el, "metadata", None), "page_number", None)


_TABLE_CAPTION_RE = re.compile(r"^\s*(table|tbl\.?|exhibit)\s*[\d ivxlc\-\.:]*", re.I)
_FIGURE_CAPTION_RE = re.compile(r"^\s*(figure|fig\.?)\s*[\d ivxlc\-\.:]*", re.I)


def find_figure_caption(elements: list, idx: int, window: int = 3) -> str:
    """Find the document's own caption for a figure, e.g. 'Figure 12. ...'.

    This is real text written by the authors, unlike the model-generated caption,
    so it is the most trustworthy thing an image chunk can carry.
    """
    page = _page(elements[idx])
    for offset in list(range(1, window + 1)) + [-o for o in range(1, window + 1)]:
        j = idx + offset if offset > 0 else idx - abs(offset)
        if not (0 <= j < len(elements)):
            continue
        el = elements[j]
        if _page(el) != page:
            continue
        text = (getattr(el, "text", "") or "").strip()
        if not text or len(text) > 300:
            continue
        if _category(el) == "FigureCaption" or _FIGURE_CAPTION_RE.match(text):
            return text
    return ""


def first_sentence(text: str, limit: int = 200) -> str:
    """Trim a generated caption to its opening claim.

    The model writes ~570 characters where OCR contributes ~44, so the caption is
    ~90% of an image chunk's embedded text even when it is wrong. Keeping only the
    first sentence preserves the gist and stops it dominating the vector.
    """
    t = " ".join((text or "").split())
    if not t:
        return ""
    cut = t.split(". ")[0].strip()
    return (cut[:limit].rstrip() + ".") if len(cut) > limit else (cut.rstrip(".") + ".")


def find_table_caption(elements: list, idx: int, window: int = 2) -> str:
    """Look just before/after a Table element for a caption-ish neighbor."""
    table_page = _page(elements[idx])

    for offset in list(range(1, window + 1)) + [-o for o in range(1, window + 1)]:
        j = idx - offset if offset > 0 else idx + abs(offset)
        if not (0 <= j < len(elements)):
            continue
        el = elements[j]
        if _page(el) != table_page:
            continue
        cat = _category(el)
        text = (getattr(el, "text", "") or "").strip()
        if not text or len(text) > 400:
            continue
        if cat == "FigureCaption" or _TABLE_CAPTION_RE.match(text):
            return text
    return ""


def running_titles(elements: list) -> dict[int, str]:
    """Map element index -> nearest preceding Title text (its section heading)."""
    titles: dict[int, str] = {}
    current = ""
    for i, el in enumerate(elements):
        if _category(el) == "Title":
            current = (getattr(el, "text", "") or "").strip() or current
        titles[i] = current
    return titles


def _chunk_title(chunk, fallback: str = "") -> str:
    """Pull the section heading out of a chunk's original elements."""
    orig = getattr(getattr(chunk, "metadata", None), "orig_elements", None) or []
    for el in orig:
        if _category(el) == "Title":
            text = (getattr(el, "text", "") or "").strip()
            if text:
                return text
    return fallback


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def partition(pdf_path: str, cache_dir: Optional[str] = None) -> list:
    """Partition the PDF, optionally reusing a cached result.

    hi_res runs a layout model over every page, which is minutes-per-document on
    CPU. The cache key is the file's path, size and mtime, so edits invalidate it.
    """
    from unstructured.partition.pdf import partition_pdf

    cache_path = None
    if cache_dir:
        import hashlib
        import pickle

        st = os.stat(pdf_path)
        key = hashlib.sha256(
            f"{os.path.abspath(pdf_path)}|{st.st_size}|{int(st.st_mtime)}".encode()
        ).hexdigest()[:16]
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"partition-{key}.pkl")

        if os.path.exists(cache_path):
            print(f"      reusing cached partition: {cache_path}")
            with open(cache_path, "rb") as fh:
                return pickle.load(fh)

    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",                      # layout model, needed for tables/images
        infer_table_structure=True,             # populates metadata.text_as_html
        extract_image_block_types=["Image"],    # pull image bytes out
        extract_image_block_to_payload=True,    # ...as base64 in metadata.image_base64
    )

    if cache_path:
        import pickle

        with open(cache_path, "wb") as fh:
            pickle.dump(elements, fh)
        print(f"      cached partition -> {cache_path}")

    return elements


def build_text_chunks(text_elements: list, doc_title_hint: str = "") -> list[Chunk]:
    from unstructured.chunking.title import chunk_by_title

    chunks = chunk_by_title(
        text_elements,
        combine_text_under_n_chars=MIN_CHARS,
        new_after_n_chars=SOFT_LIMIT,
        max_characters=HARD_LIMIT,
        overlap=OVERLAP,
        overlap_all=True,          # overlap between *all* chunks, not just hard splits
        multipage_sections=True,
    )

    out: list[Chunk] = []
    last_title = doc_title_hint
    for ch in chunks:
        title = _chunk_title(ch, fallback=last_title)
        last_title = title or last_title
        out.append(
            Chunk(
                chunk_id=_new_id(),
                chunk_type="text",
                text=ch.text,
                metadata={
                    "page_number": getattr(ch.metadata, "page_number", None),
                    "title": title,
                    "element_category": _category(ch),
                    "char_count": len(ch.text),
                },
            )
        )
    return out


def build_image_chunks(
    image_elements: list[tuple[int, Any]],
    titles: dict[int, str],
    captioner: ImageCaptioner,
    stop: set[str],
    elements: list,
    keywords_first: bool = True,
) -> list[Chunk]:
    out: list[Chunk] = []
    total = len(image_elements)
    for n, (idx, el) in enumerate(image_elements, 1):
        print(f"      image {n}/{total} (page {_page(el)})...", flush=True)
        b64 = getattr(el.metadata, "image_base64", None)
        if not b64:
            continue

        try:
            img = _decode_b64_image(b64)
        except Exception as exc:  # corrupt payload shouldn't kill the run
            print(f"[warn] could not decode image on page {_page(el)}: {exc}")
            continue

        try:
            caption = captioner.caption(img)
        except Exception as exc:
            print(f"[warn] captioning failed on page {_page(el)}: {exc}")
            caption = ""

        ocr_text = ocr_image(img)
        keywords = extract_keywords(ocr_text, stop)
        fig_caption = find_figure_caption(elements, idx)
        title = titles.get(idx, "")

        # Order and proportion both matter. Trustworthy text first (the document's
        # own figure caption, then OCR keywords read off the image), and the
        # generated caption last and trimmed. Where OCR found nothing there is
        # nothing to anchor the generated caption against, so it is dropped
        # entirely rather than left as the chunk's only content.
        if keywords_first:
            pieces = []
            if fig_caption:
                pieces.append(fig_caption)
            if keywords:
                pieces.append(f"Keywords: {', '.join(keywords)}")
            if pieces:
                if caption:
                    pieces.append(first_sentence(caption))
            else:
                pieces.append(f"Figure in section: {title}" if title else "Figure.")
            body = "\n\n".join(pieces)
            dropped = not (fig_caption or keywords)
        else:
            body = caption or "Image."
            if keywords:
                body = f"{body}\n\nKeywords: {', '.join(keywords)}"
            dropped = False

        out.append(
            Chunk(
                chunk_id=_new_id(),
                chunk_type="image",
                text=body,
                metadata={
                    "page_number": _page(el),
                    "title": title,
                    "image_base64": b64,
                    "image_mime_type": getattr(el.metadata, "image_mime_type", "image/jpeg"),
                    "caption": caption,                 # full generated caption kept
                    "figure_caption": fig_caption,
                    "generated_caption_used": not dropped and bool(caption),
                    "ocr_keywords": keywords,
                    "ocr_raw_char_count": len(ocr_text),
                    "char_count": len(body),
                },
            )
        )
    return out


def build_table_chunks(
    table_elements: list[tuple[int, Any]],
    elements: list,
    titles: dict[int, str],
    text_layer: Optional[dict[int, list[str]]] = None,
) -> list[Chunk]:
    """Build table chunks, preferring the PDF text layer over OCR'd HTML.

    Text-layer tables are matched to `unstructured`'s Table elements positionally
    within a page (nth table on page N). Any element with no counterpart falls back
    to the OCR'd HTML, and the choice is recorded in `metadata.table_source`.
    """
    from collections import defaultdict

    used: dict[int, int] = defaultdict(int)
    out: list[Chunk] = []
    for idx, el in table_elements:
        page = _page(el)
        source = "ocr_html"
        md = ""

        if text_layer:
            candidates = text_layer.get(page, [])
            n = used[page]
            if n < len(candidates):
                md = candidates[n]
                used[page] = n + 1
                source = "text_layer"

        if not md.strip():
            html = getattr(el.metadata, "text_as_html", None)
            md = html_table_to_markdown(html) if html else (getattr(el, "text", "") or "")
            source = "ocr_html"
        if not md.strip():
            continue

        caption = find_table_caption(elements, idx)
        parts = split_markdown_table(md, caption=caption)

        for i, part in enumerate(parts):
            out.append(
                Chunk(
                    chunk_id=_new_id(),
                    chunk_type="table",
                    text=part,
                    metadata={
                        "page_number": _page(el),
                        "title": titles.get(idx, ""),
                        "table_caption": caption,
                        "table_part": i + 1,
                        "table_parts_total": len(parts),
                        "table_source": source,
                        "text_as_html": getattr(el.metadata, "text_as_html", None),
                        "char_count": len(part),
                    },
                )
            )
    return out


def apply_cross_chunk_overlap(chunks: list[Chunk], overlap: int = OVERLAP) -> list[Chunk]:
    """Give image/table chunks the same ~500-char lead-in that text chunks get.

    `chunk_by_title(overlap_all=True)` already handles text-to-text overlap, and
    table splits carry their own row overlap, so this only fills the gaps: the
    first piece of a table and every image chunk.
    """
    for i, ch in enumerate(chunks):
        if i == 0:
            continue
        if ch.chunk_type == "text":
            continue
        if ch.chunk_type == "table" and ch.metadata.get("table_part", 1) > 1:
            continue

        prev_tail = chunks[i - 1].text[-overlap:].strip()
        if not prev_tail:
            continue

        ch.metadata["core_text"] = ch.text
        ch.metadata["overlap_prefix_chars"] = len(prev_tail)
        ch.text = f"{prev_tail}\n\n{ch.text}"
        ch.metadata["char_count"] = len(ch.text)
    return chunks


def chunk_pdf(
    pdf_path: str,
    llava_model: str = DEFAULT_LLAVA_MODEL,
    load_in_4bit: bool = False,
    device: Optional[str] = None,
    skip_images: bool = False,
    cache_dir: Optional[str] = None,
    strip_bp: bool = True,
    tables_from: str = "text-layer",
    keywords_first: bool = True,
) -> list[Chunk]:
    print(f"[1/5] partitioning {pdf_path} (hi_res)...")
    elements = partition(pdf_path, cache_dir=cache_dir)
    print(f"      {len(elements)} elements")

    boilerplate: set[str] = set()
    if strip_bp:
        boilerplate = detect_boilerplate(elements)
        before = len(elements)
        elements = strip_boilerplate(elements, boilerplate)
        print(f"      boilerplate: {len(boilerplate)} repeating lines, "
              f"dropped {before - len(elements)} elements")
        for k in list(boilerplate)[:4]:
            print(f"        - {k[:70]!r}")

    text_layer = None
    if tables_from == "text-layer":
        text_layer = extract_text_layer_tables(pdf_path, boilerplate)
        print(f"      text-layer tables found on {len(text_layer)} pages "
              f"({sum(len(v) for v in text_layer.values())} tables)")

    titles = running_titles(elements)
    doc_title_hint = next(
        (
            (getattr(e, "text", "") or "").strip()
            for e in elements
            if _category(e) == "Title"
        ),
        "",
    )

    text_elements: list = []
    image_elements: list[tuple[int, Any]] = []
    table_elements: list[tuple[int, Any]] = []

    for i, el in enumerate(elements):
        cat = _category(el)
        if cat == "Table":
            table_elements.append((i, el))
        elif cat == "Image":
            image_elements.append((i, el))
        else:
            text_elements.append(el)

    print(f"[2/5] text elements: {len(text_elements)} | tables: {len(table_elements)} "
          f"| images: {len(image_elements)}")

    print("[3/5] chunking text by title...")
    chunks = build_text_chunks(text_elements, doc_title_hint)

    print("[4/5] tables -> markdown...")
    chunks += build_table_chunks(table_elements, elements, titles, text_layer)

    if skip_images or not image_elements:
        print("[5/5] images skipped")
    else:
        print(f"[5/5] captioning {len(image_elements)} images with {llava_model}...")
        captioner = ImageCaptioner(
            model_name=llava_model, device=device, load_in_4bit=load_in_4bit
        )
        chunks += build_image_chunks(image_elements, titles, captioner,
                                     _load_stopwords(), elements,
                                     keywords_first=keywords_first)

    # Reading order: page, then text -> table -> image within a page.
    order = {"text": 0, "table": 1, "image": 2}
    chunks.sort(
        key=lambda c: (c.metadata.get("page_number") or 0, order.get(c.chunk_type, 9))
    )

    return apply_cross_chunk_overlap(chunks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _summarize(chunks: Iterable[Chunk]) -> None:
    chunks = list(chunks)
    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1
    sizes = [len(c.text) for c in chunks] or [0]
    print(f"\n{len(chunks)} chunks: " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    print(f"chars  min={min(sizes)}  max={max(sizes)}  avg={sum(sizes) // len(sizes)}")
    over = [c for c in chunks if len(c.text) > HARD_LIMIT + OVERLAP]
    if over:
        print(f"[warn] {len(over)} chunk(s) exceed the hard limit + overlap allowance")


def main() -> None:
    ap = argparse.ArgumentParser(description="Chunk a PDF into text/table/image chunks.")
    ap.add_argument("pdf_path", help="path to the PDF file")
    ap.add_argument("--out", default="chunks.json", help="output JSON path")
    ap.add_argument("--model", default=DEFAULT_LLAVA_MODEL, help="LLaVA checkpoint")
    ap.add_argument("--load-in-4bit", action="store_true", help="4-bit quantized load (needs bitsandbytes)")
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"], help="force device")
    ap.add_argument("--skip-images", action="store_true", help="don't load LLaVA / caption images")
    ap.add_argument("--cache-dir", default=None, help="reuse/store the hi_res partition result here")
    ap.add_argument("--keep-boilerplate", action="store_true",
                    help="don't strip repeating running headers/footers")
    ap.add_argument("--tables-from", default="text-layer", choices=["text-layer", "ocr"],
                    help="read tables from the PDF text layer (default) or unstructured's OCR")
    ap.add_argument("--caption-first", action="store_true",
                    help="old behaviour: generated caption leads the image chunk")
    ap.add_argument("--strip-base64", action="store_true", help="omit image_base64 from the JSON output")
    args = ap.parse_args()

    chunks = chunk_pdf(
        args.pdf_path,
        llava_model=args.model,
        load_in_4bit=args.load_in_4bit,
        device=args.device,
        skip_images=args.skip_images,
        cache_dir=args.cache_dir,
        strip_bp=not args.keep_boilerplate,
        tables_from=args.tables_from,
        keywords_first=not args.caption_first,
    )

    payload = []
    for c in chunks:
        d = c.to_dict()
        if args.strip_base64:
            d["metadata"].pop("image_base64", None)
        payload.append(d)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    _summarize(chunks)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
