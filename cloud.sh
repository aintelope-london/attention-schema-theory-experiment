#!/usr/bin/env bash
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# cloud.sh — Cloud instance bootstrap for aintelope experiments.
# Two-command setup from a fresh instance:
#   wget https://raw.githubusercontent.com/aintelope-london/attention-schema-theory-experiment/main/cloud.sh
#   bash cloud.sh

set -euo pipefail

# ── Constants ──────────────────────────────────────────────────────────────────
REPO_URL="https://github.com/aintelope-london/attention-schema-theory-experiment.git"
REPO_DIR="repo"
INSTALL_SCRIPT="install.py"
VENV="venv_aintelope"

# ── System dependencies ────────────────────────────────────────────────────────
echo "--- Installing system dependencies ---"
sudo apt update -qq
sudo apt install -y build-essential python3-venv python3.10-dev tmux

# ── Clone repo ─────────────────────────────────────────────────────────────────
if [ ! -d "$REPO_DIR" ]; then
  echo "--- Cloning repo ---"
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# ── Install dependencies ───────────────────────────────────────────────────────
echo "--- Running $INSTALL_SCRIPT ---"
python3 "$INSTALL_SCRIPT"

echo "--- Installing dev dependencies ---"
source "$VENV/bin/activate"
make install-dev
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    pip install torch  # PyPI carries CUDA-enabled aarch64 builds
else
    pip install torch --index-url https://download.pytorch.org/whl/cu121
fi

# ── Done ───────────────────────────────────────────────────────────────────────
INSTANCE_IP=$(curl -s ifconfig.me)

echo ""
echo "=== Setup complete ==="
echo ""
echo "Create a tmux:"
echo "  tmux new -s main"
echo ""
echo "Activate with:"
echo "  source $REPO_DIR/$VENV/bin/activate"
echo ""
echo "Then run:"
echo "  make tests-validation 2>&1 | tee ~/repo/outputs/test_run.log"
echo ""
echo "When done with experiments, pull results from your local machine with:"
echo "  scp -r ubuntu@$INSTANCE_IP:~/$REPO_DIR/outputs ./outputs"
