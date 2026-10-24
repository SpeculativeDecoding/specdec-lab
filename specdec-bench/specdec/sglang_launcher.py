"""Path A: launch one SGLang server per configuration.

Flags verified against sglang v0.5.22 (tag commit baa9bef, 13 Oct 2026):
  --speculative-algorithm EAGLE3 | EAGLE
  --speculative-draft-model-path, --speculative-num-steps, --speculative-eagle-topk,
  --speculative-num-draft-tokens, --speculative-adaptive, --speculative-adaptive-config
  --cuda-graph-max-bs-decode   (NOT --cuda-graph-max-bs in this version)
  --max-running-requests  (SGLang resets it to 48 for spec decoding unless set)
  --mem-fraction-static, --disable-radix-cache, --dtype, --port
  env SGLANG_RECORD_STEP_TIME=1 -> /server_info internal_states[0].step_time_dict
If all three tree flags are omitted, SGLang auto-picks (5, 4, 8) for LlamaForCausalLM.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .tree_config import TreeConfig

DEFAULT_TARGET = "meta-llama/Llama-3.1-8B-Instruct"
# The target/draft pair SGLang's own CI uses (python/sglang/test/test_utils.py).
DEFAULT_EAGLE3_DRAFT = "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"


@dataclass
class ServerSpec:
    mode: str                          # "spec" | "ar" | "auto" | "adaptive" | "talon"
    tree: Optional[TreeConfig] = None  # required for mode == "spec" / "talon"
    algorithm: str = "EAGLE3"
    model_path: str = DEFAULT_TARGET
    draft_model_path: str = DEFAULT_EAGLE3_DRAFT
    port: int = 30000
    dtype: str = "float16"             # match the EAGLE-repo path for cross-checks
    max_running_requests: int = 32
    cuda_graph_max_bs: int = 32
    mem_fraction_static: float = 0.80
    disable_radix_cache: bool = True   # repeated prompts must not hit the prefix cache
    adaptive_config: Optional[str] = None
    adaptive_start: TreeConfig = field(default_factory=lambda: TreeConfig(3, 1, 4, "chain"))
    extra_args: List[str] = field(default_factory=list)
    talon_mu: float = 0.03             # TALON-S confidence threshold (mode == "talon")

    @property
    def name(self) -> str:
        if self.mode == "spec":
            return f"{self.algorithm.lower()}_{self.tree.name}"
        if self.mode == "talon":
            return f"talon_{self.tree.name}_mu{self.talon_mu:g}"
        return f"{self.algorithm.lower()}_{self.mode}" if self.mode != "ar" else "ar"

    def command(self, python: str = sys.executable) -> List[str]:
        cmd = [python, "-m", "sglang.launch_server",
               "--model-path", self.model_path,
               "--port", str(self.port),
               "--dtype", self.dtype,
               "--max-running-requests", str(self.max_running_requests),
               "--cuda-graph-max-bs-decode", str(self.cuda_graph_max_bs),
               "--mem-fraction-static", str(self.mem_fraction_static)]
        if self.disable_radix_cache:
            cmd.append("--disable-radix-cache")
        if self.mode in ("spec", "talon", "auto", "adaptive"):
            cmd += ["--speculative-algorithm", self.algorithm,
                    "--speculative-draft-model-path", self.draft_model_path]
        if self.mode in ("spec", "talon"):
            if self.tree is None:
                raise ValueError(f"mode={self.mode} needs a TreeConfig")
            cmd += self.tree.validate().sglang_args()
        elif self.mode == "adaptive":
            t = self.adaptive_start.validate()
            if t.topk != 1:
                raise ValueError("SGLang adaptive mode only works with topk=1")
            cmd += t.sglang_args() + ["--speculative-adaptive"]
            if self.adaptive_config:
                cmd += ["--speculative-adaptive-config", self.adaptive_config]
        elif self.mode not in ("ar", "auto"):
            raise ValueError(f"unknown mode {self.mode}")
        return cmd + list(self.extra_args)

    def to_dict(self) -> Dict:
        return {"mode": self.mode, "name": self.name, "algorithm": self.algorithm,
                "tree": self.tree.to_dict() if self.tree else None,
                "model_path": self.model_path, "draft_model_path": self.draft_model_path,
                "dtype": self.dtype, "max_running_requests": self.max_running_requests,
                "cuda_graph_max_bs": self.cuda_graph_max_bs,
                "mem_fraction_static": self.mem_fraction_static,
                "disable_radix_cache": self.disable_radix_cache,
                "talon": ({"mu": self.talon_mu, "num_steps": self.tree.steps, "root_k": self.tree.topk}
                          if self.mode == "talon" else None)}

    def talon_env(self, out_dir: str) -> Dict[str, str]:
        """Environment that switches on specdec/sglang_talon_plugin.py in the server."""
        if self.mode != "talon":
            return {}
        return {"SPECDEC_TALON": "1", "SPECDEC_TALON_MU": str(self.talon_mu),
                "SPECDEC_TALON_NUM_STEPS": str(self.tree.steps),
                "SPECDEC_TALON_ROOT_K": str(self.tree.topk), "SPECDEC_TALON_OUT_DIR": out_dir}


class Server:
    """Context manager: start the server, wait for /health, kill the whole group."""

    def __init__(self, spec: ServerSpec, log_path: str, timeout_s: float = 1200,
                 host: str = "127.0.0.1"):
        self.spec, self.log_path, self.timeout_s = spec, log_path, timeout_s
        self.base_url = f"http://{host}:{spec.port}"
        self.proc: Optional[subprocess.Popen] = None

    def __enter__(self):
        import requests
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        self.out_dir = os.path.dirname(os.path.abspath(self.log_path))
        self.marker = os.path.join(self.out_dir, "talon_marker.jsonl")
        if os.path.exists(self.marker):
            os.remove(self.marker)  # a stale marker must not vouch for this launch
        env = dict(os.environ, SGLANG_RECORD_STEP_TIME="1", **self.spec.talon_env(self.out_dir))
        self._log = open(self.log_path, "w")
        self._log.write(" ".join(self.spec.command()) + "\n\n")
        self._log.flush()
        self.proc = subprocess.Popen(self.spec.command(), stdout=self._log,
                                     stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with code {self.proc.returncode}; see {self.log_path}")
            try:
                if requests.get(self.base_url + "/health", timeout=5).status_code == 200:
                    return self
            except requests.RequestException:
                pass
            time.sleep(5)
        self.__exit__(None, None, None)
        raise TimeoutError(f"server not healthy after {self.timeout_s}s; see {self.log_path}")

    def verify_talon(self):
        """Raise unless the TALON-S hook was registered and called in this server's processes
        with this spec's settings. Call after warm-up: with CUDA graphs the first call happens
        during capture, without them on the first decode."""
        if self.spec.mode != "talon":
            return
        try:
            events = [json.loads(line) for line in open(self.marker) if line.strip()]
        except OSError:
            events = []
        want = {"mu": self.spec.talon_mu, "num_steps": self.spec.tree.steps, "root_k": self.spec.tree.topk}
        ok = [e for e in events if e.get("event") == "called"
              and all(abs(float(e.get(k, -1)) - float(v)) < 1e-9 for k, v in want.items())]
        if not ok:
            raise RuntimeError(
                f"TALON-S hook did not run with {want} (marker {self.marker}: {events}). Is the "
                "'talon' entry point in group sglang.srt.plugins visible to the server (see chtc/job_sglang.sh)?")

    def __exit__(self, *exc):
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=60)
            except Exception:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if getattr(self, "_log", None):
            self._log.close()
        time.sleep(5)  # let the GPU memory drain before the next launch
        return False
