"""
Project entrypoint for running the Streamlit UI.

Why this file exists:
- Some VS Code setups expect a Python entrypoint like `main.py`.
- The actual app lives in `app/app.py` and is started by Streamlit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    app_path = root / "app" / "app.py"

    if not app_path.is_file():
        raise SystemExit(f"Could not find Streamlit app at: {app_path}")

    # Use the current Python interpreter to ensure VS Code uses the venv.
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.run(cmd, check=True).returncode)


if __name__ == "__main__":
    main()

