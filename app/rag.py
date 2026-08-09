"""
Grounded answering over retrieved chunks (the bonus chatbot).

Uses the same small open-weights model as the HyDE experiment
(`Qwen/Qwen2.5-1.5B-Instruct`, Apache 2.0, ungated) so the app needs no API key.

The prompt pins the model to the retrieved passages and requires page citations.
A 1.5B model will still get things wrong on a dense datasheet, so the UI shows the
cited pages alongside the answer and the retrieval result stays the primary
output -- the chat is an addition to it, not a replacement.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

LLM = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM = (
    "You answer questions about an electronic component datasheet using only the "
    "excerpts provided. Cite the page for every claim, like (p. 17). If the "
    "excerpts do not contain the answer, say so plainly instead of guessing. "
    "Be concise: three sentences at most."
)


class Answerer:
    def __init__(self, model_name: str = LLM) -> None:
        self.model_name = model_name
        self._model = None
        self._tok = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[rag] loading {self.model_name} on {dev}")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        kw = {"dtype": torch.float16} if dev == "cuda" else {}
        try:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kw)
            self._model.to(dev)
        except torch.cuda.OutOfMemoryError:
            print("[rag] CUDA OOM -- falling back to CPU")
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.to("cpu")
        self._model.eval()

    @staticmethod
    def _context(chunks: list[dict[str, Any]], budget: int = 4000) -> str:
        out, used = [], 0
        for c in chunks:
            head = f"[page {c['page']}" + (f" - {c['title']}" if c["title"] else "") + "]"
            body = " ".join(c["text"].split())
            piece = f"{head}\n{body}"
            if used + len(piece) > budget:
                piece = piece[: max(0, budget - used)]
            if not piece.strip():
                break
            out.append(piece)
            used += len(piece)
            if used >= budget:
                break
        return "\n\n".join(out)

    def answer(self, query: str, chunks: list[dict[str, Any]],
               max_new_tokens: int = 220) -> str:
        import torch

        with self._lock:
            self._ensure()
            ctx = self._context(chunks)
            msgs = [
                {"role": "system", "content": SYSTEM},
                {"role": "user",
                 "content": f"Excerpts from the datasheet:\n\n{ctx}\n\n"
                            f"Question: {query}"},
            ]
            text = self._tok.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True)
            inputs = self._tok([text], return_tensors="pt").to(self._model.device)
            with torch.inference_mode():
                out = self._model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self._tok.eos_token_id)
            gen = out[0][inputs["input_ids"].shape[-1]:]
            return self._tok.decode(gen, skip_special_tokens=True).strip()


answerer = Answerer()
