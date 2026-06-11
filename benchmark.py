#!/usr/bin/env python3
"""
benchmark.py – LLM Energy Benchmark (WSL2 / NVIDIA edition)
============================================================
Measures GPU power draw (W) via nvidia-smi while sending a fixed prompt
set to Ollama-hosted models in Docker.  Produces per-model CSV timeseries
and plots.

Usage
-----
    python3 benchmark.py                          # uses defaults
    python3 benchmark.py --models gemma3:4b llama3.2:3b
    python3 benchmark.py --runs 5 --idle-secs 20
    python3 benchmark.py --no-docker              # Ollama already running

Requirements (WSL2)
-------------------
    pip install requests psutil matplotlib pandas numpy
    docker pull ollama/ollama
    nvidia-smi must be reachable (WSL2 with GPU passthrough)
"""

import argparse
import csv
import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import platform

import requests

# ── Optional heavy deps (imported lazily) ────────────────────────────────────
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── Constants ─────────────────────────────────────────────────────────────────
OLLAMA_URL      = "http://localhost:11434"
CONTAINER_NAME  = "llm-bench-ollama"
OLLAMA_IMAGE    = "ollama/ollama"
POLL_INTERVAL_S = 0.5   # nvidia-smi polling cadence
Path("results").mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)
RESULTS_DIR     = Path("results/"+datetime.now().strftime("%Y%m%d_%H%M%S"))
PLOTS_DIR       = Path("plots/"+datetime.now().strftime("%Y%m%d_%H%M%S"))
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


def collect_system_info() -> dict:
    """Collects hardware/OS metadata for reproducibility documentation."""
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "os": {
            "system":   platform.system(),
            "release":  platform.release(),
            "version":  platform.version(),
            "machine":  platform.machine(),
            "node":     platform.node(),
        },
        "cpu": {
            "brand":        platform.processor(),
            "physical_cores": None,
            "logical_cores":  None,
            "max_freq_mhz":   None,
        },
        "ram_total_gb": None,
        "gpu": [],
    }

    # CPU details via psutil
    if HAS_PSUTIL:
        info["cpu"]["physical_cores"] = psutil.cpu_count(logical=False)
        info["cpu"]["logical_cores"]  = psutil.cpu_count(logical=True)
        freq = psutil.cpu_freq()
        if freq:
            info["cpu"]["max_freq_mhz"] = freq.max
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / 1024**3, 2)

    # GPU details via nvidia-smi
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader"
        ], stderr=subprocess.DEVNULL, text=True)
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            info["gpu"].append({
                "name":          parts[0] if len(parts) > 0 else "",
                "driver_version": parts[1] if len(parts) > 1 else "",
                "vram_mb":       parts[2] if len(parts) > 2 else "",
                "compute_cap":   parts[3] if len(parts) > 3 else "",
            })
    except Exception:
        pass

    return info


def save_system_info(info: dict, out_dir: Path, stamp: str) -> None:
    path = out_dir / f"system_info_{stamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"System info saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# GPU power polling
# ═══════════════════════════════════════════════════════════════════════════════

class GpuPoller:
    """Polls nvidia-smi in a background thread; collects (timestamp, watts) pairs."""

    def __init__(self, interval: float = POLL_INTERVAL_S):
        self.interval = interval
        self._samples: list[tuple[float, float, float, float]] = [] # (t_mono, gpu_w, gpu_temp_c, cpu_temp_c)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.available = self._check_available()

    @staticmethod
    def _check_available() -> bool:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, text=True
            )
            float(out.strip().split("\n")[0])
            return True
        except Exception:
            return False

    @staticmethod
    def _read_gpu() -> tuple[float, float] | tuple[None, None]:
        """Returns (watts, temp_celsius) summed/averaged across all GPUs."""
        try:
            out = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=power.draw,temperature.gpu",
                "--format=csv,noheader,nounits"
            ], stderr=subprocess.DEVNULL, text=True)
            watts, temps = [], []
            for line in out.strip().splitlines():
                parts = line.split(",")
                if len(parts) == 2:
                    watts.append(float(parts[0].strip()))
                    temps.append(float(parts[1].strip()))
            if watts:
                return sum(watts), sum(temps) / len(temps)
        except Exception:
            pass
        return None, None

    @staticmethod
    def _read_cpu_temp() -> float | None:
        """Best-effort CPU temperature: tries psutil.sensors_temperatures first,
        falls back to /sys/class/thermal (common in WSL2)."""
        # psutil sensors (needs lm-sensors on Linux)
        if HAS_PSUTIL and hasattr(psutil, "sensors_temperatures"):
            try:
                sensors = psutil.sensors_temperatures()
                # Priority: coretemp → k10temp → acpitz
                for key in ("coretemp", "k10temp", "zenpower", "acpitz"):
                    if key in sensors:
                        entries = sensors[key]
                        # Take 'Package id 0' or first entry
                        pkg = next((e for e in entries if "package" in e.label.lower()), entries[0])
                        return pkg.current
            except Exception:
                pass

        # WSL2 fallback: /sys/class/thermal/thermal_zone*
        try:
            import glob
            zones = sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp"))
            if zones:
                with open(zones[0]) as f:
                    return float(f.read().strip()) / 1000.0
        except Exception:
            pass

        return None

    def _run(self):
        while not self._stop.is_set():
            w, gt = self._read_gpu()
            ct = self._read_cpu_temp()
            if w is not None:
                self._samples.append((
                    time.monotonic(),
                    w,
                    gt if gt is not None else float("nan"),
                    ct if ct is not None else float("nan"),
                ))
            time.sleep(self.interval)

    def start(self):
        self._samples.clear()
        self._stop.clear()
        if not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[tuple[float, float, float, float]]:
        self._stop.set()
        self._thread.join(timeout=5)
        return list(self._samples)

    def snapshot(self) -> list[tuple[float, float, float, float]]:
        return list(self._samples)


# ═══════════════════════════════════════════════════════════════════════════════
# Docker / Ollama helpers
# ═══════════════════════════════════════════════════════════════════════════════

def docker_run_ollama(gpu: bool = True) -> None:
    """Start the Ollama Docker container."""
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME],
                   capture_output=True)
    cmd = [
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "-p", "11434:11434",
        "-v", "ollama_models:/root/.ollama",
    ]
    if gpu:
        cmd += ["--gpus", "all"]
    cmd.append(OLLAMA_IMAGE)
    subprocess.run(cmd, check=True)


def docker_stop_ollama() -> None:
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)


def wait_ollama_ready(timeout: int = 60) -> None:
    print("  Waiting for Ollama API...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            if r.status_code == 200:
                print(" ready.")
                return
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(2)
    raise RuntimeError("Ollama did not become ready in time.")


def ollama_pull(model: str) -> None:
    print(f"  Pulling model {model} (skips if cached)...")
    subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "ollama", "pull", model],
        check=True
    )


def ollama_warmup(model: str) -> None:
    """Send a tiny request to load model weights into GPU VRAM."""
    print(f"  Warming up {model}...", end="", flush=True)
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": "hello", "stream": False},
            timeout=120
        )
    except Exception:
        pass
    print(" done.")


def ollama_unload(model: str) -> None:
    """Unloads the model from gpu-vram via Ollama API."""
    print(f"  Unloading {model} from VRAM...", end="", flush=True)
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=30
        )
        print(" done.")
    except Exception as e:
        print(f" WARN: could not unload ({e})")


def ollama_unload_all() -> None:
    """Unloads all models from gpu-vram via Ollama API."""
    print("  Unloading all loaded models from VRAM...", end="", flush=True)
    try:
        r = requests.get(f"{OLLAMA_URL}/api/ps", timeout=10)
        r.raise_for_status()
        running = r.json().get("models", [])
        if not running:
            print(" nothing loaded.")
            return
        for entry in running:
            model_name = entry.get("name", "")
            if model_name:
                requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": model_name, "keep_alive": 0},
                    timeout=30
                )
                print(f"\n    Unloaded: {model_name}", end="", flush=True)
        print(" done.")
    except Exception as e:
        print(f" WARN: could not unload all models ({e})")


def ollama_generate(model: str, prompt: str, timeout: int = 180) -> dict:
    """Send a single prompt, return parsed JSON response."""
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout
    )
    r.raise_for_status()
    return r.json()


# ═══════════════════════════════════════════════════════════════════════════════
# CSV writers
# ═══════════════════════════════════════════════════════════════════════════════

PROMPT_CSV_HEADER = [
    "run_id", "model", "phase", "prompt_idx", "prompt_short",
    "t_start_mono", "t_end_mono", "duration_ms",
    "tokens_per_sec", "prompt_tokens", "prompt_tokens_per_sec", "gen_tokens",
    "gpu_mean_w", "gpu_peak_w", "gpu_energy_j",
    "cpu_mean_pct",
    "total_tokens", "j_per_token", "j_per_out_token"
]

TIMESERIES_CSV_HEADER = [
    "t_mono", "wall_ts", "model", "phase",
    "gpu_w", "gpu_temp_c", "cpu_temp_c", "cpu_pct"
]


def _cpu_pct() -> float:
    if HAS_PSUTIL:
        return psutil.cpu_percent(interval=None)
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Main benchmark logic
# ═══════════════════════════════════════════════════════════════════════════════

def run_model(
    model: str,
    prompts: list[str],
    runs: int,
    idle_secs: int,
    warmup_secs: int,
    poller: GpuPoller,
    prompt_csv: csv.DictWriter,
    ts_csv: csv.DictWriter,
    ts_file,
) -> None:

    t0_wall = datetime.now(timezone.utc).isoformat()
    print(f"\n{'═'*60}")
    print(f"  Model: {model}")
    print(f"{'═'*60}")

    ollama_pull(model)
    ollama_warmup(model)

    if warmup_secs > 0:
        print(f"  Extra warmup sleep {warmup_secs}s...")
        time.sleep(warmup_secs)

    # ── Idle baseline ──────────────────────────────────────────────────────
    print(f"  Measuring idle ({idle_secs}s)...", end="", flush=True)
    poller.start()
    t_idle_start = time.monotonic()
    for _ in range(idle_secs * 2):   # 0.5s steps so we capture CPU too
        if HAS_PSUTIL:
            _cpu_pct()               # prime psutil
        time.sleep(0.5)
    idle_samples = poller.stop()
    t_idle_end = time.monotonic()

    idle_gpu_w = [w for _, w, _gt, _ct in idle_samples]
    idle_mean_w = sum(idle_gpu_w) / len(idle_gpu_w) if idle_gpu_w else 0.0

    # Write idle timeseries
    for ts, w, gt, ct in idle_samples:
        ts_csv.writerow({
            "t_mono": f"{ts:.3f}",
            "wall_ts": "",
            "model": model,
            "phase": "idle",
            "gpu_w": f"{w:.2f}",
            "gpu_temp_c": f"{gt:.1f}" if not isinstance(gt, float) or not __import__('math').isnan(gt) else "",
            "cpu_temp_c": f"{ct:.1f}" if not isinstance(ct, float) or not __import__('math').isnan(ct) else "",
            "cpu_pct": "",
        })
    ts_file.flush()

    # Write idle summary row
    prompt_csv.writerow({
        "run_id": 0,
        "model": model,
        "phase": "idle",
        "prompt_idx": -1,
        "prompt_short": "IDLE_BASELINE",
        "t_start_mono": f"{t_idle_start:.3f}",
        "t_end_mono": f"{t_idle_end:.3f}",
        "duration_ms": int((t_idle_end - t_idle_start) * 1000),
        "tokens_per_sec": 0,
        "prompt_tokens": 0,
        "prompt_tokens_per_sec": 0,
        "gen_tokens": 0,
        "gpu_mean_w": f"{idle_mean_w:.3f}",
        "gpu_peak_w": f"{max(idle_gpu_w, default=0):.3f}",
        "gpu_energy_j": f"{idle_mean_w * (t_idle_end - t_idle_start):.3f}",
        "cpu_mean_pct": "",
        "total_tokens": 0,
        "j_per_token": 0,
        "j_per_out_token": 0,
    })
    print(f" idle GPU: {idle_mean_w:.1f} W")

    # ── Load runs ──────────────────────────────────────────────────────────
    for run in range(1, runs + 1):
        print(f"\n  Run {run}/{runs}")
        for pidx, prompt in enumerate(prompts):
            short = prompt[:60].replace("\n", " ")
            print(f"    [{pidx+1:02d}] {short[:50]}...", end="", flush=True)

            cpu_samples_pct: list[float] = []

            # Start polling
            poller.start()
            t_start = time.monotonic()

            try:
                resp = ollama_generate(model, prompt)
            except Exception as e:
                poller.stop()
                print(f" FAILED ({e})")
                continue

            t_end = time.monotonic()
            samples = poller.stop()

            duration_ms = int((t_end - t_start) * 1000)
            gpu_watts = [w for _, w, _gt, _ct in samples]
            mean_w = sum(gpu_watts) / len(gpu_watts) if gpu_watts else 0.0
            peak_w = max(gpu_watts, default=0.0)
            energy_j = mean_w * (t_end - t_start)

            eval_count = int(resp.get("eval_count") or 0)
            eval_duration = int(resp.get("eval_duration") or 0)  # in Nanosekunden
            tps = (eval_count / (eval_duration / 1e9)) if eval_duration > 0 else 0.0
            g_tok   = int(resp.get("eval_count") or 0)
            prompt_tps = 0.0
            p_dur = int(resp.get("prompt_eval_duration") or 0)
            p_tok = int(resp.get("prompt_eval_count") or 0)
            if p_dur > 0:
                prompt_tps = p_tok / (p_dur / 1e9)

            # Write timeseries
            for ts, w, gt, ct in samples:
                ts_csv.writerow({
                    "t_mono": f"{ts:.3f}",
                    "wall_ts": "",
                    "model": model,
                    "phase": f"run{run}_p{pidx}",
                    "gpu_w": f"{w:.2f}",
                    "gpu_temp_c": f"{gt:.1f}" if not isinstance(gt, float) or not __import__('math').isnan(gt) else "",
                    "cpu_temp_c": f"{ct:.1f}" if not isinstance(ct, float) or not __import__('math').isnan(ct) else "",
                    "cpu_pct": "",
                })
            ts_file.flush()

            total_tokens = p_tok + g_tok
            j_per_token = energy_j / total_tokens if total_tokens > 0 else 0.0
            j_per_out_token = energy_j / g_tok if g_tok > 0 else 0.0

            prompt_csv.writerow({
                "run_id": run,
                "model": model,
                "phase": "load",
                "prompt_idx": pidx,
                "prompt_short": short,
                "t_start_mono": f"{t_start:.3f}",
                "t_end_mono": f"{t_end:.3f}",
                "duration_ms": duration_ms,
                "tokens_per_sec": f"{tps:.1f}",
                "prompt_tokens": p_tok,
                "prompt_tokens_per_sec": f"{prompt_tps:.1f}",
                "gen_tokens": g_tok,
                "gpu_mean_w": f"{mean_w:.3f}",
                "gpu_peak_w": f"{peak_w:.3f}",
                "gpu_energy_j": f"{energy_j:.3f}",
                "cpu_mean_pct": "",
                "total_tokens" : total_tokens,
                "j_per_token" : j_per_token,
                "j_per_out_token" : j_per_out_token,
            })

            print(f" {duration_ms}ms | {tps:.0f} tok/s | {mean_w:.1f} W (peak {peak_w:.1f} W) | {energy_j:.1f} J")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="LLM Energy Benchmark (WSL2/NVIDIA)")
    p.add_argument("--models", nargs="+",
                   default=["gemma3:4b", "llama3.2:3b", "phi4-mini:3.8b"],
                   help="Ollama model tags to benchmark")
    p.add_argument("--prompts-file", default="prompts.txt",
                   help="Path to prompts file (one prompt per line, # = comment)")
    p.add_argument("--runs", type=int, default=3,
                   help="Number of times to run the full prompt set per model")
    p.add_argument("--idle-secs", type=int, default=20,
                   help="Seconds to measure idle baseline before each model")
    p.add_argument("--warmup-secs", type=int, default=10,
                   help="Extra sleep after model load before measuring")
    p.add_argument("--no-docker", action="store_true",
                   help="Skip Docker management (Ollama already running)")
    p.add_argument("--no-gpu", action="store_true",
                   help="Don't pass --gpus all to Docker (CPU-only mode)")
    p.add_argument("--out-dir", default=f"{RESULTS_DIR}",
                   help="Directory for CSV output files")
    return p.parse_args()


def load_prompts(path: str) -> list[str]:
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # System-Info dokumentieren
    sys_info = collect_system_info()
    save_system_info(sys_info, out_dir, stamp)

    prompts = load_prompts(args.prompts_file)
    print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    print(f"Models: {args.models}")
    print(f"Runs per model: {args.runs}")

    # Check nvidia-smi
    poller = GpuPoller()
    if not poller.available:
        print("\nWARN: nvidia-smi not available or returned no power data.")
        print("  Check: nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits")
        print("  Continuing – gpu_w columns will be 0.\n")

    prompt_path = out_dir / f"prompts_{stamp}.csv"
    ts_path     = out_dir / f"timeseries_{stamp}.csv"

    # Start Docker if needed
    if not args.no_docker:
        print("Starting Ollama container...")
        docker_run_ollama(gpu=not args.no_gpu)
        wait_ollama_ready()

    ollama_unload_all()  # <- Clean-State vor dem ersten Modell

    try:
        with open(prompt_path, "w", newline="", encoding="utf-8") as pf, \
             open(ts_path,     "w", newline="", encoding="utf-8") as tf:

            prompt_csv = csv.DictWriter(pf, fieldnames=PROMPT_CSV_HEADER)
            prompt_csv.writeheader()

            ts_csv = csv.DictWriter(tf, fieldnames=TIMESERIES_CSV_HEADER)
            ts_csv.writeheader()

            # Wir holen uns die Gesamtzahl der Modelle
            total_models = len(args.models)

            for index, model in enumerate(args.models):
                try:
                    run_model(
                        model=model,
                        prompts=prompts,
                        runs=args.runs,
                        idle_secs=args.idle_secs,
                        warmup_secs=args.warmup_secs,
                        poller=poller,
                        prompt_csv=prompt_csv,
                        ts_csv=ts_csv,
                        ts_file=tf,
                    )

                    # ── Nach allen Runs: Modell entladen
                    ollama_unload(model)

                    # Cooldown nur, wenn es NICHT das letzte Modell ist
                    if index < total_models - 1:
                        print("  Cooling down 15s...")
                        time.sleep(15)

                except Exception as e:
                    print(f"  ERROR: {e}")
                    print(f"  Model {model} skipped!")

    finally:
        if not args.no_docker:
            print("\nStopping Ollama container...")
            docker_stop_ollama()

    print(f"\n{'═'*60}")
    print(f"✓ Done!")
    print(f"  Prompt summary : {prompt_path}")
    print(f"  Timeseries     : {ts_path}")
    print(f"\nPlot with:")
    print(f"  python3 plot_results.py {stamp} \n")


if __name__ == "__main__":
    main()
