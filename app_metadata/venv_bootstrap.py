#!/usr/bin/env python3
"""Bootstrap helper for running Flowception app in a venv.

This script ensures the app_metadata package is on the Python path
and provides entry points for venv-installed console scripts.

Usage: flowception-app-venv [command] [args...]
Commands: rule-engine, api-server, profile-generator, test
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """Ensure app_metadata is importable and delegate to subcommands."""
    app_metadata_dir = Path(__file__).resolve().parent
    flowception_root = app_metadata_dir.parent

    if str(flowception_root) not in sys.path:
        sys.path.insert(0, str(flowception_root))

    if len(sys.argv) < 2:
        print("Usage: flowception-app-venv [rule-engine|api-server|profile-generator|test] [args...]")
        return 1

    command = sys.argv[1]
    remaining_args = sys.argv[2:]

    if command == "rule-engine":
        from app_metadata.rule_engine import main as rule_engine_main

        sys.argv = ["flowception-rule-engine"] + remaining_args
        return rule_engine_main()

    elif command == "api-server":
        from app_metadata.api_server import main as api_main

        sys.argv = ["flowception-rule-api"] + remaining_args
        return api_main()

    elif command == "profile-generator":
        from app_metadata.profile_generator import main as profile_main

        sys.argv = ["flowception-profile"] + remaining_args
        return profile_main()

    elif command == "test":
        try:
            import pytest
        except ImportError:
            print("pytest not installed. Install with: pip install pytest")
            return 1

        test_file = app_metadata_dir / "test_rule_engine.py"
        return pytest.main([str(test_file)] + remaining_args)

    else:
        print(f"Unknown command: {command}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
