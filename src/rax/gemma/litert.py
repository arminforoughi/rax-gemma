"""Gemma on LiteRT-LM (Google AI Edge): the same questions, with no server at all.

    gemma = LiteRtGemma("~/models/gemma-4-E2B-it.litertlm")
    gemma.ask("What is on the table?", images=[bgr], schema={...})   # -> dict

The drop-in for :class:`rax.gemma.client.Gemma` on a Raspberry Pi 5: the model runs in
this process on the Pi's CPU (Google's figures for Gemma 4 E2B there: ~99 tokens/s in,
~9 tokens/s out, ~1.4 GB), so the arm, the camera and the model share one board. Each
question is its own conversation, so nothing from one leaks into the next.

    pip install litert-lm-api        # Linux aarch64 (Pi OS 64-bit), macOS arm64, x86_64
    hf download litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm \\
        --local-dir ~/models

``schema`` is enforced by LiteRT's constrained decoding where the build supports it;
otherwise the schema is put in the prompt and the first JSON object in the answer is
taken, which Gemma follows well at temperature 0.
"""

from __future__ import annotations

import json
import os
import re
import time

import cv2
import numpy as np

from .client import GemmaError, is_vision_model

E2B_REPO, E2B_FILE = "litert-community/gemma-4-E2B-it-litert-lm", "gemma-4-E2B-it.litertlm"
#: Gemma 3 1B: text only, for a fast chat while RAX's detector does the seeing.
SMALL_REPO, SMALL_FILE = "litert-community/Gemma3-1B-IT", "gemma3-1b-it-int4.litertlm"


def jpeg_bytes(bgr: np.ndarray, max_side: int = 512, quality: int = 85) -> bytes:
    """The image as JPEG, long side at most ``max_side`` (the Pi pays for every pixel)."""
    h, w = bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise GemmaError("could not encode the image")
    return buf.tobytes()


def first_json(text: str) -> dict:
    """The first JSON object in ``text`` (models like to wrap it in ```json fences)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        raise GemmaError(f"gemma did not answer in JSON: {text[:200]!r}")
    depth, quote, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if quote:
            esc, quote = (not esc and ch == "\\"), (quote if esc or ch != '"' else False)
            continue
        if ch == '"':
            quote = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError as e:
                    raise GemmaError(f"gemma's JSON did not parse: {text[:200]!r}") from e
    raise GemmaError(f"gemma's JSON was cut off: {text[:200]!r}")


class LiteRtGemma:
    """One Gemma model loaded in this process by LiteRT-LM."""

    def __init__(self, model_path: str = f"~/models/{E2B_FILE}", backend: str = "cpu",
                 temperature: float = 0.0, max_output_tokens: int = 384,
                 max_side: int = 512, constrain: bool = True, vision: bool | None = None,
                 log=None):
        self.model_path = os.path.expanduser(model_path)
        self.model = os.path.basename(self.model_path)
        #: False for the text-only Gemma 3 270M and 1B: then RAX's detector does the
        #: seeing and Gemma only talks and plans. None: tell from the file name.
        self.vision = is_vision_model(self.model) if vision is None else vision
        self.backend, self.temperature = backend, temperature
        self.max_output_tokens, self.max_side, self.constrain = max_output_tokens, max_side, constrain
        self.log = log or (lambda msg: None)
        self.last_s = 0.0
        self._engine = None

    def ready(self) -> bool:
        """True if the model file is there and LiteRT-LM is installed."""
        try:
            import litert_lm  # noqa: F401
        except ImportError:
            return False
        return os.path.isfile(self.model_path)

    def why_not_ready(self) -> str:
        try:
            import litert_lm  # noqa: F401
        except ImportError:
            return "pip install litert-lm-api"
        return (f"no model at {self.model_path}: hf download {E2B_REPO} {E2B_FILE} "
                f"--local-dir {os.path.dirname(self.model_path)}")

    def _load(self):
        if self._engine is None:
            import litert_lm as lm
            lm.set_min_log_severity(lm.LogSeverity.ERROR)
            be = {"cpu": lm.Backend.CPU, "gpu": lm.Backend.GPU, "npu": lm.Backend.NPU}[self.backend]()
            cache = os.path.expanduser("~/.cache/litert-lm")
            os.makedirs(cache, exist_ok=True)      # a fresh Pi has none, and the engine won't make it
            img_kw = dict(vision_backend=lm.Backend.CPU(), max_num_images=1) if self.vision else {}
            self._engine = lm.Engine(self.model_path, backend=be, cache_dir=cache, **img_kw)
        return self._engine

    def warm(self) -> float:
        """Load the model now (seconds on a Pi), so the first question is not the slow one."""
        t0 = time.time()
        self._load()
        return time.time() - t0

    def close(self):
        if self._engine is not None:
            self._engine.__exit__(None, None, None)
            self._engine = None

    def ask(self, prompt: str, images=(), schema: dict | None = None,
            system: str | None = None):
        """Gemma's answer: a dict when ``schema`` is given, else the text."""
        if images and not self.vision:
            raise GemmaError(f"{self.model} is text-only: it cannot look at a picture")
        import litert_lm as lm
        engine = self._load()
        if schema is not None and not self.constrain:
            prompt += ("\n\nAnswer with only a JSON object that matches this JSON schema, "
                       f"and nothing else:\n{json.dumps(schema)}")
        parts = [lm.Content.ImageBytes(jpeg_bytes(i, self.max_side)) for i in images]
        parts.append(lm.Content.Text(prompt))
        conv_kw = dict(sampler_config=lm.SamplerConfig(temperature=self.temperature or None,
                                                       top_k=1 if not self.temperature else None),
                       thinking_config=lm.ThinkingConfig(enable_thinking=False),
                       max_output_tokens=self.max_output_tokens)
        if system:
            conv_kw["system_message"] = system
        send_kw = {}
        if schema is not None and self.constrain:
            conv_kw["constrained_decoding_config"] = lm.ConstrainedDecodingConfig(
                enable=True, provider=lm.LiteRtLmConstraintProviderType.LL_GUIDANCE)
            send_kw["response_format"] = lm.ResponseFormat.json(schema)
        t0 = time.time()
        try:
            with engine.create_conversation(**conv_kw) as conv:
                reply = conv.send_message(lm.Message.user(lm.Contents(parts)), **send_kw)
        except Exception as e:
            if schema is not None and self.constrain:
                # a build without constrained decoding: fall back to asking for the JSON
                self.log(f"gemma: constrained decoding unavailable ({e}); asking for JSON")
                self.constrain = False
                return self.ask(prompt, images, schema, system)
            raise GemmaError(f"litert-lm: {type(e).__name__}: {e}") from e
        self.last_s = time.time() - t0
        text = str(reply.contents) if hasattr(reply, "contents") else reply["content"][0]["text"]
        self.log(f"gemma {self.model} ({self.backend}): {self.last_s:.1f}s")
        if schema is None:
            return text.strip()
        return first_json(text)


def open_gemma(backend: str = "ollama", model: str | None = None, **kw):
    """The Gemma for ``backend``: ``"ollama"`` (a laptop) or ``"litert"`` (a Pi)."""
    if backend == "litert":
        return LiteRtGemma(model or f"~/models/{E2B_FILE}", **kw)
    from .client import Gemma
    return Gemma(model or "gemma4:e4b", **kw)
