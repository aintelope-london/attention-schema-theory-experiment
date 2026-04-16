# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import re
from pathlib import Path

OUTPUTS_DIR = Path(__file__).parents[2] / "outputs"


def read_reports(outputs_dir=OUTPUTS_DIR):
    return {
        p.parent.name: p.read_text()
        for p in sorted(Path(outputs_dir).glob("*/report.txt"))
    }


def parse_reports(reports):
    return {name: _parse(text) for name, text in reports.items()}


def _parse(text):
    arch_map = {
        m.group(1): m.group(2).strip()
        for m in re.finditer(r"Block:\s+(\S+)\n\s*Agent:.*\n\s*Arch:\s+(.+)", text)
    }
    eff_map = {
        m.group(1): float(m.group(2))
        for m in re.finditer(
            r"Block:\s+(\S+)(?:(?!Block:).)*?Efficiency:\s*([\d.]+)%",
            text,
            re.DOTALL,
        )
    }
    return {
        block: {"arch": arch_map.get(block, ""), "efficiency": eff_map.get(block)}
        for block in arch_map.keys() | eff_map.keys()
    }


_MODEL_RE = re.compile(r"agent_0:\s+model:\s+(\S+)")


def read_models(outputs_dir=OUTPUTS_DIR):
    models = {}
    for folder in sorted(Path(outputs_dir).glob("*")):
        config = folder / "config.yaml"
        source = config if config.exists() else folder / "report.txt"
        m = _MODEL_RE.search(source.read_text()) if source.exists() else None
        models[folder.name] = m.group(1) if m else None
    return models


def show_model(models):
    print("=== Models Used ===")
    for run, model in sorted(models.items()):
        print(f"  {run}: {model}")


def show_algorithms(parsed):
    print("=== Algorithms Used ===")
    for run, blocks in sorted(parsed.items()):
        print(f"\n{run}:")
        for block, data in blocks.items():
            print(f"  [{block}] {data['arch']}")


def show_final_efficiency(parsed, block):
    print(f"=== Final Efficiency: block='{block}' ===")
    for run, blocks in sorted(parsed.items()):
        eff = blocks.get(block, {}).get("efficiency")
        print(f"  {run}: {eff}%")


if __name__ == "__main__":
    parsed = parse_reports(read_reports())
    models = read_models()

    tasks = [
        (show_model, [models]),
        (show_algorithms, [parsed]),
        (show_final_efficiency, [parsed, "test"]),
    ]

    for fn, args in tasks:
        fn(*args)
        print()
