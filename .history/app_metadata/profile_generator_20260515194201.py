#!/usr/bin/env python3
"""Auto-detect macOS system specs and generate a compatible profile.

Usage: flowception-profile [--output profile.json] [--interactive]
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


def detect_macos_specs() -> dict[str, Any]:
    """Detect system specs on macOS."""
    profile: dict[str, Any] = {
        "os": "macos",
        "arch": platform.machine(),
        "hardwareProfile": "unknown",
        "workflow": ["toy-i2v"],
        "selectedDeps": [],
        "modelVariant": "distilled",
        "freeDiskTb": 0.0,
        "licenseAcceptanceRequired": False,
        "licenseAccepted": False,
        "userPreference": {},
    }

    # Detect architecture
    machine = platform.machine()
    if machine == "arm64":
        profile["hardwareProfile"] = "macbook-apple-silicon-laptop"
    elif machine == "x86_64":
        profile["hardwareProfile"] = "macbook-intel-laptop"

    # Detect free disk space
    try:
        result = subprocess.run(
            ["df", "-h", "/"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) > 3:
                    free_str = parts[3]
                    free_tb = _parse_disk_size(free_str)
                    profile["freeDiskTb"] = round(free_tb, 2)
    except Exception:
        profile["freeDiskTb"] = 0.0

    # Detect GPU/accelerator capability
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and "Apple" in result.stdout:
            profile["selectedDeps"] = ["python", "pytorch"]
        else:
            profile["selectedDeps"] = ["python", "pytorch"]
    except Exception:
        profile["selectedDeps"] = ["python", "pytorch"]

    return profile


def _parse_disk_size(size_str: str) -> float:
    """Parse disk size like '500Gi' or '1.2Ti' to TB."""
    size_str = size_str.strip().upper()
    multipliers = {
        "B": 1e-12,
        "K": 1e-9,
        "M": 1e-6,
        "G": 1e-3,
        "T": 1.0,
        "P": 1e3,
    }

    for unit, mult in multipliers.items():
        if unit in size_str:
            num_str = size_str.replace(unit, "").replace("I", "")
            try:
                return float(num_str) * mult
            except ValueError:
                return 0.0

    try:
        return float(size_str) * 1e-12
    except ValueError:
        return 0.0


def interactive_prompt(profile: dict[str, Any]) -> dict[str, Any]:
    """Allow user to override detected values interactively."""
    print("\n=== Flowception Profile Generator ===\n")
    print(f"Detected: {profile['hardwareProfile']} ({profile['arch']})")
    print(f"Free disk: {profile['freeDiskTb']} TB")

    override = input("\nOverride workflow? (leave blank for 'toy-i2v'): ").strip()
    if override:
        profile["workflow"] = [override]

    override = input("Override model variant? (leave blank for 'distilled'): ").strip()
    if override:
        profile["modelVariant"] = override

    accept = input("Accept license terms? (y/n, default n): ").strip().lower()
    if accept == "y":
        profile["licenseAcceptanceRequired"] = True
        profile["licenseAccepted"] = True

    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Flowception installer profile.")
    parser.add_argument("--output", type=str, default="", help="Output JSON file path.")
    parser.add_argument("--interactive", action="store_true", help="Prompt for overrides.")
    args = parser.parse_args()

    profile = detect_macos_specs()

    if args.interactive:
        profile = interactive_prompt(profile)

    payload = json.dumps(profile, indent=2) + "\n"

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload)
        print(f"Profile written to {out_path}")
    else:
        print(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
