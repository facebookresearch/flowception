#!/usr/bin/env python3
"""Native desktop GUI for Flowception installer evaluation.

This app is local-only and uses Tkinter (no web UI).
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from app_metadata.profile_generator import detect_macos_specs
from app_metadata.rule_engine import evaluate_profile, result_to_dict


class FlowceptionDesktopApp:
    """Simple native macOS-friendly desktop UI for profile evaluation."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Flowception Installer")
        self.root.geometry("980x720")

        self.profile_text = tk.Text(root, wrap="word", height=18)
        self.result_text = tk.Text(root, wrap="word", height=18)

        self._build_layout()
        self.load_detected_profile()

    def _build_layout(self) -> None:
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")

        ttk.Label(
            top,
            text="Flowception Native Installer Assistant",
            font=("SF Pro Text", 15, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            top,
            text="Generate profile, evaluate compatibility, and review actions.",
        ).pack(anchor="w", pady=(2, 8))

        button_row = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        button_row.pack(fill="x")

        ttk.Button(
            button_row,
            text="Detect Profile",
            command=self.load_detected_profile,
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            button_row,
            text="Open Profile JSON",
            command=self.open_profile,
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            button_row,
            text="Save Profile JSON",
            command=self.save_profile,
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            button_row,
            text="Evaluate",
            command=self.evaluate,
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            button_row,
            text="Save Result JSON",
            command=self.save_result,
        ).pack(side="left")

        body = ttk.Panedwindow(self.root, orient=tk.VERTICAL)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        profile_frame = ttk.Labelframe(body, text="Profile JSON", padding=8)
        profile_frame.pack(fill="both", expand=True)
        self.profile_text.pack(in_=profile_frame, fill="both", expand=True)

        result_frame = ttk.Labelframe(body, text="Evaluation Output", padding=8)
        result_frame.pack(fill="both", expand=True)
        self.result_text.pack(in_=result_frame, fill="both", expand=True)

        body.add(profile_frame, weight=1)
        body.add(result_frame, weight=1)

    def _read_profile_editor(self) -> dict[str, Any]:
        raw = self.profile_text.get("1.0", tk.END).strip()
        if not raw:
            raise ValueError("Profile JSON is empty.")
        loaded = json.loads(raw)
        if not isinstance(loaded, dict):
            raise ValueError("Profile JSON must be an object.")
        return loaded

    def load_detected_profile(self) -> None:
        profile = detect_macos_specs()
        self.profile_text.delete("1.0", tk.END)
        self.profile_text.insert("1.0", json.dumps(profile, indent=2) + "\n")
        self.result_text.delete("1.0", tk.END)

    def open_profile(self) -> None:
        path = filedialog.askopenfilename(
            title="Open profile JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            data = Path(path).read_text(encoding="utf-8")
            loaded = json.loads(data)
            if not isinstance(loaded, dict):
                raise ValueError("Profile file must contain a JSON object.")
            self.profile_text.delete("1.0", tk.END)
            self.profile_text.insert("1.0", json.dumps(loaded, indent=2) + "\n")
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc))

    def save_profile(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save profile JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            profile = self._read_profile_editor()
            Path(path).write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def evaluate(self) -> None:
        try:
            profile = self._read_profile_editor()
            result = evaluate_profile(profile)
            payload = result_to_dict(result)
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert("1.0", json.dumps(payload, indent=2) + "\n")

            blocked_count = len(payload.get("blocked_rules", []))
            warnings_count = len(payload.get("warnings", []))
            if blocked_count > 0:
                messagebox.showwarning(
                    "Blocked",
                    f"{blocked_count} hard-block rules triggered.\nWarnings: {warnings_count}",
                )
        except Exception as exc:
            messagebox.showerror("Evaluation failed", str(exc))

    def save_result(self) -> None:
        raw = self.result_text.get("1.0", tk.END).strip()
        if not raw:
            messagebox.showinfo("Nothing to save", "Evaluate a profile first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save evaluation result JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            loaded = json.loads(raw)
            Path(path).write_text(json.dumps(loaded, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))


def main() -> int:
    root = tk.Tk()
    app = FlowceptionDesktopApp(root)
    _ = app
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
