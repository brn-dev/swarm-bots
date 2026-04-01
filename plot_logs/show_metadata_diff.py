from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from pathlib import Path
from typing import Any

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_CYAN = "\033[36m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show a git-style unified diff between two metadata json files. "
            "If paths are omitted, file dialogs are opened."
        )
    )
    parser.add_argument("left", nargs="?", type=Path, help="First metadata json file.")
    parser.add_argument("right", nargs="?", type=Path, help="Second metadata json file.")
    parser.add_argument(
        "--context",
        type=int,
        default=3,
        help="Number of unchanged context lines around each change block.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in output.",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        help=(
            "Field name to ignore during diff. "
            "Repeat this flag to ignore multiple fields. "
            "Default: script"
        ),
    )
    parser.add_argument(
        "--no-default-ignore",
        action="store_true",
        help="Disable default ignored fields (default ignored: script).",
    )
    return parser.parse_args()


def choose_two_files(initial_dir: Path | None = None) -> tuple[Path, Path]:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    start_dir = str(initial_dir) if initial_dir is not None else os.getcwd()
    first = filedialog.askopenfilename(
        title="Select first metadata JSON file",
        initialdir=start_dir,
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
    )
    if not first:
        raise ValueError("No first file selected.")

    second = filedialog.askopenfilename(
        title="Select second metadata JSON file",
        initialdir=str(Path(first).parent),
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
    )
    if not second:
        raise ValueError("No second file selected.")

    root.destroy()
    return Path(first), Path(second)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.left is not None and args.right is not None:
        return args.left, args.right
    if args.left is None and args.right is None:
        return choose_two_files()
    raise ValueError("Either provide both file paths or none.")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def canonical_json_lines(payload: Any) -> list[str]:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    return text.splitlines()


def filter_ignored_fields(payload: Any, ignored_fields: set[str]) -> Any:
    if isinstance(payload, dict):
        return {
            key: filter_ignored_fields(value, ignored_fields)
            for key, value in payload.items()
            if key not in ignored_fields
        }
    if isinstance(payload, list):
        return [filter_ignored_fields(item, ignored_fields) for item in payload]
    return payload


def build_diff_lines(
    left_path: Path,
    right_path: Path,
    left_payload: Any,
    right_payload: Any,
    context: int,
    show_full_file: bool = False,
) -> list[str]:
    left_lines = canonical_json_lines(left_payload)
    right_lines = canonical_json_lines(right_payload)
    effective_context = (
        max(len(left_lines), len(right_lines)) if show_full_file else context
    )
    return list(
        difflib.unified_diff(
            left_lines,
            right_lines,
            fromfile=str(left_path),
            tofile=str(right_path),
            n=effective_context,
            lineterm="",
        )
    )


def colorize_diff_line(line: str) -> str:
    if line.startswith("+++ ") or line.startswith("--- "):
        return f"{ANSI_BOLD}{line}{ANSI_RESET}"
    if line.startswith("@@"):
        return f"{ANSI_CYAN}{line}{ANSI_RESET}"
    if line.startswith("+"):
        return f"{ANSI_GREEN}{line}{ANSI_RESET}"
    if line.startswith("-"):
        return f"{ANSI_RED}{line}{ANSI_RESET}"
    return line


def output_diff(diff_lines: list[str], use_color: bool) -> int:
    if len(diff_lines) == 0:
        print("No differences found.")
        return 0
    for line in diff_lines:
        print(colorize_diff_line(line) if use_color else line)
    return 1


def main() -> None:
    args = parse_args()
    try:
        left_path, right_path = resolve_paths(args)
        left_payload = load_json(left_path)
        right_payload = load_json(right_path)
        ignored_fields = {field for field in args.ignore if field.strip()}
        if not args.no_default_ignore:
            ignored_fields.add("script")
        left_payload = filter_ignored_fields(left_payload, ignored_fields)
        right_payload = filter_ignored_fields(right_payload, ignored_fields)
        diff_lines = build_diff_lines(
            left_path=left_path,
            right_path=right_path,
            left_payload=left_payload,
            right_payload=right_payload,
            context=args.context,
        )
        has_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        use_color = has_tty and not args.no_color
        status = output_diff(diff_lines=diff_lines, use_color=use_color)
        raise SystemExit(status)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
