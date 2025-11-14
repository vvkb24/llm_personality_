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

    def __init__(self, model_name: str = "meta-llama/Llama-3-8B-Instruct"):
        self.model_name = model_name
        self.tokenizer: Optional[object] = None
        self.model: Optional[object] = None

        if not _HAS_TRANSFORMERS:
            logging.warning("transformers not available; using dummy LLM backend")
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # model loading may fail on machines without GPUs or auth
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=getattr(torch, "float16", None),
                device_map="auto"
            )
        except Exception as e:
            logging.warning("Failed to load model '%s': %s. Falling back to dummy.", self.model_name, e)
            self.tokenizer = None
            self.model = None

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        if self.model is None or self.tokenizer is None:
            # deterministic, short dummy generation
            return f"[DUMMY OUTPUT] {prompt[:200]}"

        tokens = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**tokens, max_length=min(200, len(tokens['input_ids'][0]) + max_tokens))
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def score_prompt(self, prompt: str) -> str:
        """Return a Likert choice as a string '1'..'5'.

        If a real model is available, a simple heuristic is used; otherwise
        return a neutral '3' so downstream code can run during tests.
        """
        if self.model is None or self.tokenizer is None:
            return "3"

        options = ["1", "2", "3", "4", "5"]
        scores = {}

        try:
            for opt in options:
                scored_text = prompt + f"\nAnswer: {opt}"
                tokens = self.tokenizer(scored_text, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    out = self.model(**tokens)

                # try to pick a token logit for the option; if anything fails, fallback
                try:
                    logit = out.logits[0, -1]
                    token_id = self.tokenizer.encode(opt, add_special_tokens=False)[0]
                    scores[opt] = float(logit[token_id])
                except Exception:
                    scores[opt] = random.random()

            return max(scores, key=scores.get)
        except Exception:
            return "3"

