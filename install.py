#!/usr/bin/env python3
"""Cross-platform installation script for aintelope."""

import subprocess
import sys
from pathlib import Path

VENV_NAME = "venv_aintelope"
MIN_VERSION = (3, 10, 3)


def main():
    # Check Python version
    if sys.version_info < MIN_VERSION:
        print(f"Error: Python {'.'.join(map(str, MIN_VERSION))}+ required.")
        print(f"Current: {sys.version}")
        sys.exit(1)

    venv_path = Path(VENV_NAME)
    is_windows = sys.platform == "win32"
    pip_path = venv_path / ("Scripts" if is_windows else "bin") / "pip"

    # Create venv
    if not venv_path.exists():
        print(f"Creating virtual environment: {VENV_NAME}")
        subprocess.run([sys.executable, "-m", "venv", VENV_NAME], check=True)
    else:
        print(f"Virtual environment already exists: {VENV_NAME}")

    # Install package
    print("Installing dependencies...")
    subprocess.run([str(pip_path), "install", "-e", "."], check=True)

    # Print activation instructions
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("=" * 50)
    print("\nActivate with:")
    if is_windows:
        print(f"  {VENV_NAME}\\Scripts\\activate")
    else:
        print(f"  source {VENV_NAME}/bin/activate")
    print("\nVSCode users: Ctrl+Shift+P → 'Python: Select Interpreter'")
    print("\nThen run:")
    print("  aintelope-gui")


if __name__ == "__main__":
    main()