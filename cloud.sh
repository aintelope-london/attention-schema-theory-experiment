#!/usr/bin/env bash
# cloud.sh — Cloud instance bootstrap for aintelope experiments.
# Two-command setup from a fresh instance:
#   wget https://raw.githubusercontent.com/YOUR_ORG/YOUR_REPO/main/cloud.sh
#   bash cloud.sh

set -euo pipefail

# ── Constants ──────────────────────────────────────────────────────────────────
REPO_URL="https://github.com/aintelope-london/attention-schema-theory-experiment.git"
REPO_DIR="repo"
INSTALL_SCRIPT="install.py"

# ── Clone repo ─────────────────────────────────────────────────────────────────
if [ ! -d "$REPO_DIR" ]; then
  echo "--- Cloning repo ---"
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# ── Install dependencies ───────────────────────────────────────────────────────
echo "--- Running $INSTALL_SCRIPT ---"
python3 "$INSTALL_SCRIPT"

# ── Done ───────────────────────────────────────────────────────────────────────
INSTANCE_IP=$(curl -s ifconfig.me)

echo ""
echo "=== Setup complete ==="
echo ""
echo "Activate with:"
echo "  source venv_aintelope/bin/activate"
echo ""
echo "Then run:"
echo "  aintelope-gui"
echo ""
echo "When done with experiments, pull results from your local machine with:"
echo "  scp -r ubuntu@$INSTANCE_IP:~/repo/outputs ./outputs"