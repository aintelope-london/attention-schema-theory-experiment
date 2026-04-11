#!/usr/bin/env python3
"""monitor.py — snapshot report for a single run folder.

Usage: python monitor.py outputs/20260411123456
"""

import sys
from pathlib import Path
from aintelope.utils.report_aggregator import read_reports, parse_reports
sys.path.insert(0, str(Path(__file__).parent))

def main():
    if len(sys.argv) != 2:
        print("Usage: python monitor.py <run_folder>")
        sys.exit(1)

    folder = Path(sys.argv[1])
    reports = read_reports(folder)

    if not reports:
        print(f"{folder.name}/  no report.txt found (still running or crashed)")
        return

    parsed = parse_reports(reports)
    for run, blocks in parsed.items():
        for block, data in blocks.items():
            eff = (
                f"{data['efficiency']:.1f}%"
                if data["efficiency"] is not None
                else "no efficiency data"
            )
            arch = f"  [{data['arch']}]" if data["arch"] else ""
            print(f"{block}: {eff}{arch}")


if __name__ == "__main__":
    main()
