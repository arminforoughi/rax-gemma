"""Gemma, served on this machine by Ollama. Nothing leaves the device.

    gemma = Gemma("gemma4:e4b")
    gemma.ask("What is on the table?", images=[bgr], schema={...})   # -> dict

``schema`` is a JSON schema; Ollama constrains the output to it, so the answer always
parses. The host must be this machine or the local network: a public address, or one
of Ollama's ``:cloud`` models, is refused, because the whole point is that the robot
thinks for itself.
"""

from __future__ import annotations

import base64
import ipaddress
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

import cv2
import numpy as np


class GemmaError(RuntimeError):
    pass


def is_local(host_url: str) -> bool:
    """True if ``host_url`` resolves only to loopback or private addresses."""
    host = urllib.parse.urlparse(host_url).hostname or ""
    try:
        addrs = {a[4][0] for a in socket.getaddrinfo(host, None)}
    except socket.gaierror:
        return False
    return bool(addrs) and all(ipaddress.ip_address(a.split("%")[0]).is_private or
                               ipaddress.ip_address(a.split("%")[0]).is_loopback
                               for a in addrs)


def jpeg_b64(bgr: np.ndarray, max_side: int = 640, quality: int = 85) -> str:
    """The image as base64 JPEG, shrunk so its long side is at most ``max_side``."""
    h, w = bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise GemmaError("could not encode the image")
    return base64.b64encode(buf.tobytes()).decode()


class Gemma:
    """One Gemma model behind a local Ollama server."""

    def __init__(self, model: str = "gemma4:e4b", host: str = "http://127.0.0.1:11434",
                 timeout: float = 180.0, temperature: float = 0.0, log=None):
        if "cloud" in model.split(":")[-1]:
            raise GemmaError(f"{model} runs in the cloud; pick a local tag")
        if not is_local(host):
            raise GemmaError(f"{host} is not on this machine or the local network")
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout
        self.temperature = temperature
        self.log = log or (lambda msg: None)
        self.last_s = 0.0                 # how long the last answer took

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.host + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise GemmaError(f"ollama {e.code}: {e.read().decode(errors='replace')}") from e
        except urllib.error.URLError as e:
            raise GemmaError(f"ollama is not running at {self.host} ({e.reason}); "
                             "start it with `ollama serve`") from e
        except TimeoutError as e:
            raise GemmaError(f"{self.model} took over {self.timeout:.0f}s "
                             "(the first call loads the model; warm it up)") from e

    def warm(self) -> float:
        """Load the model now, so the first real question is not the slow one."""
        t0 = time.time()
        self._post("/api/chat", {"model": self.model, "messages": [], "keep_alive": "30m"})
        return time.time() - t0

    def ready(self) -> bool:
        """True if the model is pulled. ``ollama pull <model>`` otherwise."""
        try:
            tags = self._post("/api/show", {"model": self.model})
        except GemmaError:
            return False
        return "capabilities" in tags or "modelfile" in tags

    def ask(self, prompt: str, images=(), schema: dict | None = None,
            system: str | None = None):
        """Gemma's answer: a dict when ``schema`` is given, else the text."""
        msgs = [{"role": "system", "content": system}] if system else []
        msgs.append({"role": "user", "content": prompt,
                     **({"images": [jpeg_b64(i) for i in images]} if images else {})})
        body = {"model": self.model, "messages": msgs, "stream": False, "keep_alive": "30m",
                "options": {"temperature": self.temperature}}
        if schema is not None:
            body["format"] = schema
        t0 = time.time()
        out = self._post("/api/chat", body)
        self.last_s = time.time() - t0
        text = out.get("message", {}).get("content", "")
        self.log(f"gemma {self.model}: {self.last_s:.1f}s")
        if schema is None:
            return text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise GemmaError(f"gemma did not answer in JSON: {text[:200]!r}") from e
