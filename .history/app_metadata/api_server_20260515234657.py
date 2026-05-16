#!/usr/bin/env python3
"""Lightweight HTTP API wrapper for the rule engine.

Serves JSON evaluation results over HTTP/1.1 without external deps.
Run: flowception-rule-api [--port 8000] [--bind 127.0.0.1]
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from rule_engine import evaluate_profile, result_to_dict


class RuleEngineHandler(BaseHTTPRequestHandler):
    """HTTP handler for rule engine evaluation."""

    def _set_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/":
            html_path = Path(__file__).parent / "index.html"
            try:
                with html_path.open("r", encoding="utf-8") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("content-type", "text/html")
                self.end_headers()
                self.wfile.write(content.encode("utf-8"))
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"UI not found")
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/evaluate":
            content_length = int(self.headers.get("content-length", 0))
            try:
                body = self.rfile.read(content_length).decode("utf-8")
                profile: dict[str, Any] = json.loads(body)
            except (ValueError, json.JSONDecodeError) as e:
                self.send_response(400)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"error": f"Invalid JSON: {e}"}).encode("utf-8")
                )
                return

            try:
                result = evaluate_profile(profile)
                payload = result_to_dict(result)
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload, indent=2).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"error": f"Evaluation failed: {e}"}).encode("utf-8")
                )
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": "Not found"}).encode("utf-8")
            )

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default logging."""
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Flowception rule engine as HTTP API."
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to listen on."
    )
    parser.add_argument(
        "--bind", type=str, default="127.0.0.1", help="Address to bind to."
    )
    args = parser.parse_args()

    server = HTTPServer((args.bind, args.port), RuleEngineHandler)
    print(
        f"Flowception Rule Engine API listening at "
        f"http://{args.bind}:{args.port}"
    )
    print("POST /evaluate with JSON profile to evaluate.")
    print("GET /health for health check.")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
