"""vLLM engine adapter (Step 5): server launcher and streaming chat client.

Talks to a vLLM OpenAI-compatible server over plain HTTP using only the standard
library, so the client adds no dependencies.

Timing definitions (all from one monotonic clock, ``time.monotonic`` by default):

* ``t_start``          taken immediately before the HTTP request is written. The TCP
                       connect happens inside the request, so connect time, server
                       queueing and transport are all inside TTFT.
* ``t_first_content``  arrival of the first SSE event whose delta carries NON-EMPTY
                       text. Role-only, empty, and usage-only events are ignored.
* ``t_last_content``   arrival of the last such event.
* ``t_complete``       the stream finished (``[DONE]`` received).

"Arrival" means the moment the client finished reading that SSE line. A content
event can hold more than one token, so event timing is a chunk-level measurement.
Token counts are never derived from chunk counts: they come from the engine's own
``usage`` block, or are recorded as unavailable.
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class StreamResult:
    status: str = "error"  # "ok" or "error"
    error_category: Optional[str] = None
    error_message: Optional[str] = None
    http_status: Optional[int] = None
    t_start: float = 0.0
    t_first_content: Optional[float] = None
    t_last_content: Optional[float] = None
    t_complete: Optional[float] = None
    content_chunks: int = 0  # NOT a token count
    text: str = ""
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None  # engine usage
    completion_tokens: Optional[int] = None  # engine usage
    response_model: Optional[str] = None
    raw_event_count: int = field(default=0)


def build_payload(model: str, user_text: str, max_tokens: int, ignore_eos: bool = False) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_text}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True  # vLLM extension; recorded in the run config
    return payload


def stream_chat(host: str, port: int, payload: dict, *, timeout_s: float,
                clock: Callable[[], float] = time.monotonic) -> StreamResult:
    """Send one streaming chat request and return its timing and engine usage.

    Never raises for request-level failures: they are returned with an
    ``error_category`` so the caller can persist a record for every request.
    """
    res = StreamResult()
    res.t_start = clock()
    conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    done = False
    try:
        body = json.dumps(payload).encode()
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        resp = conn.getresponse()
        res.http_status = resp.status
        if resp.status != 200:
            res.error_category = "http_error"
            res.error_message = f"HTTP {resp.status}: {resp.read(500).decode('utf-8', 'replace')}"
            return res

        while True:
            line = resp.readline()
            now = clock()
            if now - res.t_start > timeout_s:
                res.error_category = "timeout"
                res.error_message = f"exceeded {timeout_s}s overall deadline"
                return res
            if not line:
                break  # EOF
            text = line.decode("utf-8", "replace").strip()
            if not text or text.startswith(":") or not text.startswith("data:"):
                continue
            data = text[5:].strip()
            if data == "[DONE]":
                done = True
                break
            res.raw_event_count += 1
            try:
                event = json.loads(data)
            except json.JSONDecodeError as exc:
                res.error_category = "malformed_stream"
                res.error_message = f"invalid JSON in stream: {exc}"
                return res
            if "error" in event:
                res.error_category = "server_error"
                res.error_message = str(event["error"])[:500]
                return res

            if event.get("model"):
                res.response_model = event["model"]
            choices = event.get("choices") or []
            if choices:
                content = (choices[0].get("delta") or {}).get("content")
                if isinstance(content, str) and content != "":
                    if res.t_first_content is None:
                        res.t_first_content = now
                    res.t_last_content = now
                    res.content_chunks += 1
                    res.text += content
                if choices[0].get("finish_reason"):
                    res.finish_reason = choices[0]["finish_reason"]
            usage = event.get("usage")
            if usage:
                res.prompt_tokens = usage.get("prompt_tokens")
                res.completion_tokens = usage.get("completion_tokens")

        res.t_complete = clock()
        if not done:
            res.error_category = "incomplete_stream"
            res.error_message = "stream ended without [DONE]"
            return res
        if res.content_chunks == 0:
            res.error_category = "empty_output"
            res.error_message = "stream completed with no generated content"
            return res
        res.status = "ok"
        return res
    except TimeoutError as exc:
        res.error_category = "timeout"
        res.error_message = f"socket timeout after {timeout_s}s: {exc}"
        return res
    except (ConnectionError, OSError, http.client.HTTPException) as exc:
        res.error_category = "connection_error"
        res.error_message = f"{type(exc).__name__}: {exc}"
        return res
    finally:
        conn.close()


class ServerStartError(RuntimeError):
    pass


class VLLMServer:
    """Launch and stop ``vllm serve`` as a child process (one GPU, pinned revision)."""

    def __init__(self, model: str, revision: str, *, port: int = 8000, host: str = "127.0.0.1",
                 dtype: str = "float16", max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.85, enable_prefix_caching: bool = False,
                 enforce_eager: bool = True, log_path: str = "server.log",
                 startup_timeout_s: float = 900.0, extra_args: tuple = ()):
        self.model, self.revision, self.port, self.host = model, revision, port, host
        self.dtype, self.max_model_len = dtype, max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_prefix_caching, self.enforce_eager = enable_prefix_caching, enforce_eager
        self.log_path, self.startup_timeout_s = log_path, startup_timeout_s
        self.extra_args = tuple(extra_args)
        self.proc: Optional[subprocess.Popen] = None

    @property
    def settings(self) -> dict:
        """Run-relevant server settings (excludes host/port so the config hash is stable)."""
        return {
            "engine": "vllm serve",
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enforce_eager": self.enforce_eager,
            # "vllm" ignores the model's generation_config.json, whose defaults
            # (repetition_penalty 1.1, top_k 20, top_p 0.8 for Qwen) would otherwise
            # silently apply to every request that does not override them.
            "generation_config": "vllm",
            "cuda_visible_devices": "0",
            "extra_args": list(self.extra_args),
        }

    def argv(self) -> list:
        exe = shutil.which("vllm")
        if exe is None:
            raise ServerStartError("the 'vllm' command was not found on PATH; is vLLM installed?")
        args = [exe, "serve", self.model, "--revision", self.revision,
                "--tokenizer-revision", self.revision, "--dtype", self.dtype,
                "--max-model-len", str(self.max_model_len),
                "--gpu-memory-utilization", str(self.gpu_memory_utilization),
                "--generation-config", "vllm",
                "--host", self.host, "--port", str(self.port)]
        args.append("--enable-prefix-caching" if self.enable_prefix_caching
                    else "--no-enable-prefix-caching")
        if self.enforce_eager:
            args.append("--enforce-eager")
        return args + list(self.extra_args)

    @property
    def pid(self) -> Optional[int]:
        return self.proc.pid if self.proc is not None else None

    def _log_tail(self, lines: int = 40) -> str:
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as fh:
                return "".join(fh.readlines()[-lines:])
        except OSError:
            return "(no server log)"

    def _ready(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://{self.host}:{self.port}/v1/models", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def start(self) -> "VLLMServer":
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}  # one GPU, set before CUDA loads
        self._log = open(self.log_path, "w", encoding="utf-8")
        kwargs = {"start_new_session": True} if os.name == "posix" else {}
        self.proc = subprocess.Popen(self.argv(), stdout=self._log, stderr=subprocess.STDOUT,
                                     env=env, **kwargs)
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise ServerStartError(
                    f"vllm exited with code {self.proc.returncode} during startup.\n"
                    f"--- last server log lines ({self.log_path}) ---\n{self._log_tail()}")
            if self._ready():
                return self
            time.sleep(2)
        self.stop()
        raise ServerStartError(f"server not ready within {self.startup_timeout_s}s.\n{self._log_tail()}")

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            else:
                self.proc.terminate()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()
        finally:
            try:
                self._log.close()
            except Exception:
                pass

    def __enter__(self) -> "VLLMServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
