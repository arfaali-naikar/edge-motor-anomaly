#!/usr/bin/env bash
# One-shot setup inside WSL2 Ubuntu. Idempotent -- safe to re-run.
set -euo pipefail

echo "==> checking we're actually in WSL"
grep -qi microsoft /proc/version || { echo "this expects WSL2 Ubuntu"; exit 1; }

# Keep the repo on the ext4 side. Working out of /mnt/c goes through the 9p
# bridge and makes every file operation several times slower -- painfully
# obvious once pytest is walking the tree.
case "$PWD" in
  /mnt/*) echo "WARNING: you're under /mnt/. Move this repo to ~/projects/ for a big speedup." ;;
esac

echo "==> apt packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip git make

echo "==> venv"
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "==> smoke test"
PYTHONPATH=src python -m pytest tests -q

cat <<'EOF'

done. next:
  source .venv/bin/activate
  make all

If you want the local LLM in the loop (see docs/ARCHITECTURE.md):
  export OLLAMA_API_BASE=http://127.0.0.1:11434
  tmux new -s motor && aider
EOF
