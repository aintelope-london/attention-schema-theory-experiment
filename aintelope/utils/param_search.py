# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/aintelope-london/attention-schema-theory-experiment

"""Dynamic hyperparameter search via Optuna TPE.

One search run = N optimize trials; each trial = one full run() call with
Optuna-suggested overrides merged into the base experiment config. TPE
proposes new param sets from the history of (params, score) pairs, with
pure random warmup for the first `n_startup_trials` trials.

The search produces:
    outputs/search.db     — SQLite study, resumable via load_if_exists
    outputs/search.log    — per-trial log lines
    outputs/trials.csv    — flat index: trial_id, score, run_outputs_dir, params...
    outputs/<timestamp>/  — one standard run directory per trial
"""

import csv
import logging
from pathlib import Path

import optuna
from omegaconf import OmegaConf


_SUGGEST = {
    "float":      lambda trial, name, spec: trial.suggest_float(name, spec["low"], spec["high"]),
    "loguniform": lambda trial, name, spec: trial.suggest_float(name, spec["low"], spec["high"], log=True),
    "int":        lambda trial, name, spec: trial.suggest_int(name, spec["low"], spec["high"]),
    "categorical": lambda trial, name, spec: trial.suggest_categorical(name, spec["choices"]),
}


def run_search(search_filename):
    """Run an Optuna hyperparameter search driven by a search config yaml."""
    from aintelope.__main__ import run
    from aintelope.config.config_utils import CONFIG_DIR

    search_cfg = OmegaConf.load(Path(CONFIG_DIR) / search_filename)
    base_cfg = OmegaConf.load(Path(CONFIG_DIR) / search_cfg.base)
    overrides = OmegaConf.create(
        {k: v for k, v in search_cfg.items() if k not in ("base", "run")}
    )

    outputs_root = Path("outputs")
    outputs_root.mkdir(parents=True, exist_ok=True)

    _configure_logging(outputs_root / "search.log")
    logger = logging.getLogger("param_search")

    s = search_cfg.run.search
    study = optuna.create_study(
        study_name="param_search",
        storage=f"sqlite:///{outputs_root / 'search.db'}",
        direction=s.objective.direction,
        sampler=optuna.samplers.TPESampler(n_startup_trials=s.n_startup_trials),
        load_if_exists=True,
    )

    initials, missing = _resolve_initials(s.params, base_cfg)
    if missing:
        raise KeyError(f"param paths not found in base config: {missing}")
    study.enqueue_trial(initials, skip_if_exists=True)

    trials_csv = outputs_root / "trials.csv"
    param_paths = [p.path for p in s.params]
    _ensure_csv_header(trials_csv, ["trial_id", "score", "outputs_dir", *param_paths])

    def objective(trial):
        cfg = OmegaConf.merge(base_cfg, overrides)
        suggested = {p.path: _SUGGEST[p.type](trial, p.path, p) for p in s.params}
        for block in cfg:
            for path, value in suggested.items():
                OmegaConf.update(cfg[block], path, value, force_add=True)
            OmegaConf.update(cfg[block], "run.trials", s.inner_trials, force_add=True)

        result = run(cfg)
        score = result["analytics"][s.objective.analytic][s.objective.block][s.objective.field]

        with trials_csv.open("a", newline="") as f:
            csv.writer(f).writerow(
                [trial.number, score, result["outputs_dir"], *[suggested[p] for p in param_paths]]
            )
        logger.info("trial %d | score=%.4f | %s", trial.number, score, suggested)
        return score

    study.optimize(objective, n_trials=s.n_trials)
    logger.info("search complete | best=%.4f | params=%s", study.best_value, study.best_params)


def _resolve_initials(params, base_cfg):
    from aintelope.config.config_utils import init_config

    resolved_cfg = init_config(base_cfg)
    resolved, missing = {}, []
    for p in params:
        value = OmegaConf.select(resolved_cfg, p.path)
        if value is None:
            missing.append(p.path)
        else:
            resolved[p.path] = value
    return resolved, missing


def _configure_logging(logfile):
    handler = logging.FileHandler(logfile)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger = logging.getLogger("param_search")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)


def _ensure_csv_header(path, columns):
    if not path.exists():
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(columns)