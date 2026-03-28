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
make format         # Apply black formatter
make isort          # Sort imports
make flake8         # Lint
make typecheck-local # Type check
```

### Setting up the LLM API access

Set environment variable `OPENAI_API_KEY`. Ensure you have loaded credits on your OpenAI API account.

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

## Actions map

The actions the agents can take have the following mapping:
```
  NOOP = 0
  LEFT = 1
  RIGHT = 2
  UP = 3
  DOWN = 4
```

Eating and drinking are not individual actions. Eating and drinking occurs always when an action ends with the agent being on top of a food or water tile, correspondingly. If the agent continues to stay on that tile then eating and drinking continues until the agent leaves. Likewise with collecting gold and silver. The agent is harmed by danger tile or predator, when the agent action ends up on a danger tile or predator tile. Cooperation reward is provided to the **OTHER** agent each time an agent is eating or drinking.

Additionally, when `observation_direction_mode` = 2 or `action_direction_mode` = 2 then the following actions become available:
```
  TURN_LEFT_90 = 5
  TURN_RIGHT_90 = 6
  TURN_LEFT_180 = 7
  TURN_RIGHT_180 = 8
```
By default, the separate turning actions are turned off.

## Human-playable demos

In the folder `aintelope/environments/demos/gridworlds/` are located the human-playable demo environments, which have same configuration as the benchmarks in our orchestrator. Playing these human-playable demos manually can give you a better intuition of the rules and how the benchmarks work.

You can launch these Python files without additional arguments. 

You can move the agents around using arrow keys (left, right, up, down). For no-op action you can use space key. 

In food sharing environment there are two agents. In a human-playable demo these agents take turns. In an RL setting they agents take actions concurrently and the environment implements their actions in a random order (randomising the order for each turn).

The human-playable benchmark environments are in the following files:
```
food_unbounded.py
danger_tiles.py
predators.py
food_homeostasis.py
food_sustainability.py
food_drink_homeostasis.py
food_drink_homeostasis_gold.py
food_drink_homeostasis_gold_silver.py
food_sharing.py
```

---

## Attribution & License

This repository is a fork and derivative of "Roland Pihlakas. From homeostasis to resource sharing: Biologically and economically aligned multi-objective multi-agent gridworld-based AI safety benchmarks". Please cite the original work (DOI: [`10.48550/arXiv.2410.00081`](https://doi.org/10.48550/arXiv.2410.00081)) — see [`CITATION.cff`](CITATION.cff) for citation metadata.  
Original upstream repository: https://github.com/aintelope-london/attention-schema-theory-experiment

License: see [`LICENSE.txt`](LICENSE.txt).  
Code contribution details: see [`AUTHORS.md`](AUTHORS.md).
