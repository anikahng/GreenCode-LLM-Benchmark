#!/usr/bin/env bash
# =============================================================================
# setup.sh – One-time setup for LLM Energy Benchmark on WSL2 + NVIDIA
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
log() { echo "[setup] $*"; }

# ── Python venv ───────────────────────────────────────────────────────────────
VENV="$SCRIPT_DIR/.venv"
log "Creating Python venv..."
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q requests psutil matplotlib pandas numpy datetime requests argparse

# ── nvidia-smi check ─────────────────────────────────────────────────────────
log "Checking nvidia-smi..."
if nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits &>/dev/null; then
  GPU_W=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1 | tr -d ' ')
  log "nvidia-smi OK – current GPU power: ${GPU_W} W ✓"
else
  log "WARN: nvidia-smi not found or returned no data."
  log "  Make sure NVIDIA drivers are installed on the Windows host (>=530)"
  log "  and nvidia-container-toolkit is set up inside WSL2."
fi

# ── Docker check ──────────────────────────────────────────────────────────────
log "Checking Docker..."
if docker info &>/dev/null; then
  log "Docker OK ✓"
else
  log "WARN: Docker not running. Install Docker Desktop or 'sudo apt install docker.io && sudo service docker start'"
fi

# ── GPU Docker runtime check ──────────────────────────────────────────────────
log "Checking NVIDIA container runtime..."
if docker info 2>/dev/null | grep -q nvidia; then
  log "NVIDIA container runtime found ✓"
else
  log "WARN: NVIDIA container runtime not found."
  log "  Install with:"
  log "    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
  log "    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
  log "    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
  log "    sudo nvidia-ctk runtime configure --runtime=docker"
  log "    sudo service docker restart"
fi

chmod +x "$SCRIPT_DIR/benchmark.py" "$SCRIPT_DIR/plot_results.py"

log ""
log "Setup complete!"
log ""
log "Activate venv and run:"
log "  source $VENV/bin/activate"
log "  python3 benchmark.py --models gemma3:4b llama3.2:3b --runs 3"
log ""
log "Then plot:"
log "  python3 plot_results.py"
