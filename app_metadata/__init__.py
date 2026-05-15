"""Flowception installer rules and evaluation engine.

Modules:
- rule_engine: Evaluate installer decisions from JSON policy/matrix
- api_server: HTTP API wrapper (POST /evaluate)
- profile_generator: Auto-detect macOS specs and generate profile
- venv_bootstrap: Venv entry point router
- test_rule_engine: pytest unit tests
"""

__version__ = "0.0.1"
