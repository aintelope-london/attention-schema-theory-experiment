# AST experiments (title pending)

Description pending.

## Installation

### 1. Install Python

Install Python 3.10.3 - 3.10.13. We recommend CPython from python.org. Do not use Conda.

- **Linux (Ubuntu/Debian):**
  ```bash
  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt update
  sudo apt install python3.10 python3.10-dev python3.10-venv build-essential git
  ```

- **Windows:** Download from [python.org](https://www.python.org/downloads/release/python-31010/) and install Git from [gitforwindows.org](https://gitforwindows.org/)

- **Mac:** `brew install python@3.10 git`

### 2. Clone and install

```bash
git clone https://github.com/aintelope-london/attention-schema-theory-experiment.git
cd attention-schema-theory-experiment
python install.py
```

On Linux/Mac, you may need to use `python3` or `python3.10` instead of `python`.

### 3. Activate

After installation, activate the virtual environment:

- **Linux/Mac:** `source venv_aintelope/bin/activate`
- **Windows:** `venv_aintelope\Scripts\activate`

VSCode users: Press `Ctrl+Shift+P` → "Python: Select Interpreter" → select `venv_aintelope`.

## Running the GUI

Via module:

```bash
python -m aintelope --gui
```

## Development

Install dev dependencies:

```bash
pip install -r requirements/dev.txt
```

Available make commands (Linux/Mac):

```bash
make tests-local    # Run tests
make tests-learning # Run learning validation
make format         # Apply black formatter
make isort          # Sort imports
make flake8         # Lint
make typecheck-local # Type check
```


### VSCode launch configurations

Copy the template: 

- **Linux/Mac:** `cp .vscode/launch.json.template .vscode/launch.json`
- **Windows:** `copy .vscode\launch.json.template .vscode\launch.json`

Edit the `PYTHONPATH` in `launch.json` to point to your local repo path.

## Cloud setup

### Choosing an instance
Use a plain Ubuntu 22.04 base image. Avoid provider-managed ML stacks (e.g. Lambda Stack) — `install.py` manages all dependencies explicitly, and pre-installed frameworks risk version conflicts.

A single A100 instance is sufficient for the current workload. Multi-node clusters (e.g. Lambda 1-Click Clusters) are out of scope.

### One-time configuration
Set `REPO_URL` at the top of `cloud.sh` to the HTTPS URL of the repository and commit the file. This is the only configuration required.

### SSH key setup (one-time per machine)
Generate a key and register it with the provider before launching an instance:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/lambda
```
Paste the contents of `~/.ssh/lambda.pub` into the provider's SSH key dashboard. The full string including the `ssh-ed25519` prefix and email suffix is required.

### Per-instance workflow
Find the instance IP in the provider dashboard once the instance is running. SSH in:
```bash
ssh -i ~/.ssh/lambda ubuntu@INSTANCE_IP
```

Run the bootstrap script:
```bash
wget https://raw.githubusercontent.com/aintelope-london/attention-schema-theory-experiment/main/cloud.sh
bash cloud.sh
```

To check the instance specs:
```bash
{ echo "=== CPU ==="; lscpu | grep -E "Model name|CPU\(s\)|Thread|Socket"; echo "=== RAM ==="; free -h; echo "=== GPU ==="; nvidia-smi; echo "=== DISK ==="; df -h /; } > specs.txt 2>&1
```

Start a persistent tmux session so the run survives disconnects and can be reattached from any future SSH:
```bash
tmux new -s main
```

Activate the environment and run experiments:
```bash
source repo/venv_aintelope/bin/activate
make tests-validation 2>&1 | tee ~/repo/outputs/test_run.log
```

To detach from the session (leaves it running): `Ctrl+B, D`  
To reattach after reconnecting via SSH: `tmux attach -t main`

### Retrieving results
Instance storage is ephemeral — results must be pulled before terminating the instance. Use the `scp` command printed at the end of `cloud.sh` output, run from your local machine:
```bash
scp -r ubuntu@INSTANCE_IP:~/repo/outputs ./outputs
```

---

## Attribution & License

Original upstream repository: https://github.com/aintelope-london/attention-schema-theory-experiment

License: see [`LICENSE.txt`](LICENSE.txt).  
Code contribution details: see [`AUTHORS.md`](AUTHORS.md).
