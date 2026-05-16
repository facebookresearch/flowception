#!/usr/bin/env python3
"""Build a native macOS .app bundle for the desktop GUI using PyInstaller."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Flowception native .app (non-web) with PyInstaller.")
    parser.add_argument(
        "--name",
        type=str,
        default="FlowceptionInstaller",
        help="Application bundle name.",
    )
    parser.add_argument(
        "--icon",
        type=str,
        default="",
        help="Optional .icns icon path.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    entry = root / "app_metadata" / "desktop_app.py"

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--windowed",
        "--name",
        args.name,
        str(entry),
    ]

    if args.icon:
        command.extend(["--icon", args.icon])

    command.extend(
        [
            "--add-data",
            f"{root / 'app_metadata' / 'compatibility-policy.json'}:app_metadata",
            "--add-data",
            f"{root / 'app_metadata' / 'dependency-decision-matrix.json'}:app_metadata",
            "--add-data",
            f"{root / 'app_metadata' / 'source-monitoring.json'}:app_metadata",
        ]
    )

    try:
        subprocess.run(command, check=True, cwd=str(root))
    except FileNotFoundError:
        print("PyInstaller is not installed. Install it with: pip install pyinstaller")
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"Build failed with exit code {exc.returncode}")
        return exc.returncode

    app_path = root / "dist" / f"{args.name}.app"
    print(f"Built native app: {app_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
