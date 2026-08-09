"""
HyDE — Hypothetical Document Embeddings.

A small local LLM writes the passage it thinks would answer the query, and that
passage is embedded and searched instead of (or alongside) the query itself.

Why it should help *this* corpus specifically: the measured failure mode is a
vocabulary gap. Questions say "resistor", "voltage", "too hot"; the datasheet says
`RTOPOFF`, `VCHGIN_OVLO`, `TJREG`. A model that has seen datasheets writes the
symbol-flavoured sentence, giving both the dense and sparse sides something to
match. It cannot help where the page simply is not represented in the index.

Generations are cached to disk keyed by (model, prompt version, query) so an eval
sweep does not regenerate them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any, Optional

DEFAULT_LLM = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT_VERSION = "v1"

SYSTEM = (
    "You write short excerpts from electronic component datasheets. "
    "You imitate the register, terminology and symbol conventions of a real "
    "datasheet: parameter symbols (V_CHGIN, I_FAST, R_TOPOFF, T_JREG), units, "
    "thresholds and pin names."
)

USER_TMPL = (
    "Write a single short passage (about 70 words) that would plausibly appear in "
    "an integrated-circuit datasheet and that answers this question. Do not "
    "restate the question, do not use bullet points, and do not say you are "
    "uncertain — just write the passage as the datasheet would.\n\n"
    "Question: {q}"
)


class HydeGenerator:
    def __init__(self, model_name: str = DEFAULT_LLM, device: Optional[str] = None,
                 cache_path: str = "hyde_cache.json", max_new_tokens: int = 130) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as fh:
                self.cache = json.load(fh)
        self._model = None
        self._tok = None

    def _key(self, q: str) -> str:
        raw = f"{self.model_name}|{PROMPT_VERSION}|{q}"
        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        kw: dict[str, Any] = {"dtype": torch.float16} if self.device == "cuda" else {}
        try:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kw)
            self._model.to(self.device)
        except torch.cuda.OutOfMemoryError:
            # 1.5B in fp16 is ~3.1GB and this GPU has ~3.2GB free; CPU is slow but
            # this only runs once per query thanks to the cache.
            print("[warn] CUDA OOM loading the LLM -- falling back to CPU")
            self.device = "cpu"
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.to("cpu")
        self._model.eval()

    def generate(self, query: str) -> str:
        key = self._key(query)
        if key in self.cache:
            return self.cache[key]

        import torch

        self._ensure()
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TMPL.format(q=query)}]
        text = self._tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        inputs = self._tok([text], return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                       do_sample=False,
                                       pad_token_id=self._tok.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[-1]:]
        passage = self._tok.decode(gen, skip_special_tokens=True).strip()

        self.cache[key] = passage
        with open(self.cache_path, "w", encoding="utf-8") as fh:
            json.dump(self.cache, fh, ensure_ascii=False, indent=1)
        return passage


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a HyDE passage for a query.")
    ap.add_argument("query", nargs="+")
    ap.add_argument("--model", default=DEFAULT_LLM)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    args = ap.parse_args()
    q = " ".join(args.query)
    g = HydeGenerator(args.model, args.device)
    print(f"\nquery: {q}\n")
    print(g.generate(q))


if __name__ == "__main__":
    main()
