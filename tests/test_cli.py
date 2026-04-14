"""Smoke tests for CLI JSON and human output."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestCli(unittest.TestCase):
    """Run subprocess against run.py (may train models on first run)."""

    def test_json_output_single_line(self) -> None:
        """Stdout is a single JSON line with inputs and predictions."""
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "run.py"),
                "--config",
                str(ROOT / "input.yaml"),
                "--output",
                "json",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        self.assertEqual(len(lines), 1)
        data = json.loads(lines[0])
        self.assertIn("inputs", data)
        self.assertIn("predictions", data)

    def test_human_output_contains_sections(self) -> None:
        """Human mode prints labeled INPUTS and RESULTS sections."""
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "run.py"),
                "--config",
                str(ROOT / "input.yaml"),
                "--output",
                "human",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("INPUTS", result.stdout)
        self.assertIn("RESULTS", result.stdout)


if __name__ == "__main__":
    unittest.main()
