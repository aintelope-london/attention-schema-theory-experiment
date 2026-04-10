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

Experiments can be run on cloud GPU instances using `cloud.sh`, a bootstrap script located at the repo root. It is provider-agnostic and has been tested on Lambda Labs (Ubuntu 22.04 base image, no Lambda Stack).

### Choosing an instance

Use a plain Ubuntu 22.04 base image. Avoid provider-managed ML stacks (e.g. Lambda Stack) — `install.py` manages all dependencies explicitly, and pre-installed frameworks risk version conflicts.

A single A100 instance is sufficient for the current workload. Multi-node clusters (e.g. Lambda 1-Click Clusters) are out of scope.

### One-time configuration

Set `REPO_URL` at the top of `cloud.sh` to the HTTPS URL of the repository and commit the file. This is the only configuration required.

### Per-instance workflow

On a fresh instance, run:

```bash
wget https://raw.githubusercontent.com/YOUR_ORG/YOUR_REPO/main/cloud.sh
bash cloud.sh
```

This clones the repo, runs `install.py`, and prints a ready-to-use `scp` command for pulling results. After setup, activate the environment and launch:

```bash
source repo/venv_aintelope/bin/activate
aintelope-gui
```

### Retrieving results

Instance storage is ephemeral — results must be pulled before terminating the instance. At the end of `cloud.sh` output, a copy-pasteable command is printed with the instance's current IP:

```bash
scp -r ubuntu@INSTANCE_IP:~/repo/outputs ./outputs
```

Run this from your local machine before terminating the instance.

---

## Attribution & License

Original upstream repository: https://github.com/aintelope-london/attention-schema-theory-experiment

License: see [`LICENSE.txt`](LICENSE.txt).  
Code contribution details: see [`AUTHORS.md`](AUTHORS.md).
