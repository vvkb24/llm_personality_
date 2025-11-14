import logging
import random
from typing import Optional

_HAS_TRANSFORMERS = True
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except Exception:
    _HAS_TRANSFORMERS = False


class LLMEngine:
    """Simple LLM wrapper with a safe dummy fallback for local testing.

    If `transformers` or model weights aren't available, the engine
    returns predictable dummy outputs so the rest of the pipeline can be tested.
    """

    def __init__(self, model_name: str = None):
        # Default to a small HF model for local testing. This will attempt
        # to download the model on first use. You can override with an
        # environment variable LLM_MODEL or pass model_name explicitly.
        import os
        self.model_name = model_name or os.environ.get("LLM_MODEL", "sshleifer/tiny-gpt2")
        self.tokenizer: Optional[object] = None
        self.model: Optional[object] = None
        self.fallback_mode = "heuristic"

        if not self.model_name:
            # No model specified: use heuristic fallback only (no HF downloads)
            logging.info("No model_name provided; using heuristic fallback for LLM behavior")
            return

        if not _HAS_TRANSFORMERS:
            logging.warning("transformers not available; using heuristic fallback")
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # model loading may fail on machines without GPUs or auth
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name
            )
            # If model loads, clear fallback
            self.fallback_mode = None
        except Exception as e:
            logging.warning("Failed to load model '%s': %s. Falling back to heuristic.", self.model_name, e)
            self.tokenizer = None
            self.model = None
            self.fallback_mode = "heuristic"

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        if self.model is None or self.tokenizer is None:
            # deterministic, short dummy generation
            return f"[DUMMY OUTPUT] {prompt[:200]}"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        try:
            output = self.model.generate(**inputs, max_new_tokens=max_tokens)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception:
            return self.tokenizer.decode(self.model.generate(inputs['input_ids'], max_new_tokens=max_tokens)[0], skip_special_tokens=True)

    def score_prompt(self, prompt: str) -> str:
        """Return a Likert choice as a string '1'..'5'.

        If a real model is available, a simple heuristic is used; otherwise
        return a neutral '3' so downstream code can run during tests.
        """
        # If fallback mode is set to heuristic, use a simple lexical heuristic
        if self.fallback_mode == "heuristic" or self.model is None or self.tokenizer is None:
            return self._heuristic_score(prompt)

        instruction = prompt + "\nPlease answer with a single digit 1-5 indicating agreement (1=strongly disagree, 5=strongly agree).\nAnswer:"
        try:
            gen = self.generate(instruction, max_tokens=5)
            # find first digit 1-5 in the generated text
            for ch in gen:
                if ch in "12345":
                    return ch
            # fallback: take first token and try to parse digits
            toks = gen.strip().split()
            for t in toks:
                for ch in t:
                    if ch in "12345":
                        return ch
        except Exception:
            pass

        return "3"

    def _heuristic_score(self, prompt: str) -> str:
        """A lightweight heuristic to score Likert 1-5 from prompt text.

        This is not a real LLM — it inspects sentiment and presence of
        positive/negative words in the quoted item text inside the prompt
        to produce a 1-5 estimate centered at 3.
        """
        import re
        # try to extract the item text within quotes (last quoted string is likely the item)
        matches = re.findall(r'"([^"]+)"', prompt)
        item_text = matches[-1] if matches else prompt
        tokens = item_text.lower().split()
        pos = sum(1 for t in tokens if t.strip('.,!?') in ("good","great","happy","joy","love","excellent","wonderful","positive","awesome","amazing","like","enjoy","prefer"))
        neg = sum(1 for t in tokens if t.strip('.,!?') in ("bad","sad","terrible","hate","awful","horrible","negative","angry","upset","worse","dislike","never"))
        # crude sentiment score in [-1,1]
        denom = max(1, len(tokens))
        sentiment = (pos - neg) / denom
        # map sentiment to 1-5 (center at 3)
        score = 3 + sentiment * 4
        # clamp
        score = max(1, min(5, round(score)))
        return str(int(score))

