from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LocalGenerator:
    model_id: str
    cache_dir: Path
    max_new_tokens: int
    temperature: float
    top_p: float
    local_files_only: bool = False
    dtype: str = "auto"
    device_map: str = "auto"

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.cache_dir))
        torch_dtype = self._torch_dtype()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=str(self.cache_dir),
            local_files_only=self.local_files_only,
            trust_remote_code=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=str(self.cache_dir),
            local_files_only=self.local_files_only,
            torch_dtype=torch_dtype,
            device_map=self.device_map,
            trust_remote_code=False,
        )
        self.model.eval()

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        do_sample = self.temperature > 0
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        return strip_code_fences(decoded)

    def _torch_dtype(self):
        if self.dtype == "auto":
            if torch.cuda.is_available():
                return torch.bfloat16
            return torch.float32
        if self.dtype == "bfloat16":
            return torch.bfloat16
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "float32":
            return torch.float32
        raise ValueError(f"Unknown dtype: {self.dtype}")


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if "[BEGIN]" in stripped:
        stripped = stripped.split("[BEGIN]", 1)[1].strip()
    if "[DONE]" in stripped:
        stripped = stripped.split("[DONE]", 1)[0].strip()
    fenced = re.findall(
        r"```\s*([A-Za-z0-9_+.-]*)\s*\n?(.*?)```",
        stripped,
        flags=re.DOTALL,
    )
    if fenced:
        candidates = [
            {
                "language": language.lower(),
                "body": body.strip(),
            }
            for language, body in fenced
            if body.strip()
        ]
        implementations = [
            candidate
            for candidate in candidates
            if re.search(
                r"(?m)^\s*(?:(?:async\s+)?def|class)\s+[A-Za-z_]\w*",
                candidate["body"],
            )
        ]
        pool = implementations or candidates
        if pool:
            # Explanatory answers can contain example, regex, or assertion
            # fences before the final implementation. Prefer an actual Python
            # definition, then a Python-labelled and longer block.
            selected = max(
                pool,
                key=lambda candidate: (
                    candidate["language"] in {"python", "py"},
                    len(candidate["body"]),
                ),
            )
            return selected["body"]
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped
