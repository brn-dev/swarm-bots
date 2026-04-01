from __future__ import annotations

import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Sequence

from show_metadata_diff import (
    build_diff_lines,
    filter_ignored_fields,
    load_json,
)

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    HAS_DND = True
except ImportError:
    DND_FILES = None
    TkinterDnD = None
    HAS_DND = False


class MetadataDiffInteractiveApp:
    def __init__(
        self,
        root: tk.Tk,
        default_left: Path | None = None,
        default_right: Path | None = None,
        context_lines: int = 3,
        ignored_fields: Sequence[str] = (),
        no_default_ignore: bool = False,
    ) -> None:
        self.root = root
        self.root.title("Metadata Diff")
        self.root.geometry("1100x760")
        self.root.minsize(820, 540)

        self.context_lines = context_lines
        self.no_default_ignore = no_default_ignore
        self.manual_ignored_fields = tuple(
            field.strip() for field in ignored_fields if field.strip()
        )

        self.left_path_var = tk.StringVar(value=str(default_left) if default_left else "")
        self.right_path_var = tk.StringVar(value=str(default_right) if default_right else "")
        self.show_full_file_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(
            value="Pick two metadata JSON files, then press Compare."
        )

        self.diff_text: tk.Text | None = None
        self._build_ui()
        self._update_compare_state()

        if default_left and default_right:
            self.compare_files()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        controls = ttk.Frame(self.root, padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        left_slot = self._create_file_slot(
            parent=controls,
            title="Left File",
            column=0,
            path_var=self.left_path_var,
            browse_command=self.choose_left_file,
            assign_func=self.assign_left_path,
        )
        right_slot = self._create_file_slot(
            parent=controls,
            title="Right File",
            column=1,
            path_var=self.right_path_var,
            browse_command=self.choose_right_file,
            assign_func=self.assign_right_path,
        )

        action_row = ttk.Frame(controls)
        action_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        action_row.columnconfigure(5, weight=1)

        compare_button = ttk.Button(action_row, text="Compare", command=self.compare_files)
        compare_button.grid(row=0, column=0, padx=(0, 8))
        self.compare_button = compare_button

        swap_button = ttk.Button(action_row, text="Swap", command=self.swap_files)
        swap_button.grid(row=0, column=1, padx=(0, 8))

        clear_button = ttk.Button(action_row, text="Clear", command=self.clear_diff)
        clear_button.grid(row=0, column=2, padx=(0, 8))

        full_file_checkbox = ttk.Checkbutton(
            action_row,
            text="Show full file",
            variable=self.show_full_file_var,
            command=self._on_toggle_show_full_file,
        )
        full_file_checkbox.grid(row=0, column=3, padx=(0, 12))

        if not HAS_DND:
            dnd_note = ttk.Label(
                action_row,
                text="Drag-and-drop disabled: install tkinterdnd2 for file dropping.",
            )
            dnd_note.grid(row=0, column=4, sticky="w")

        status_label = ttk.Label(controls, textvariable=self.status_var, wraplength=1080)
        status_label.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        output_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        output_frame.grid(row=1, column=0, sticky="nsew")
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        text_widget = tk.Text(output_frame, wrap="none", font=("Consolas", 10))
        text_widget.grid(row=0, column=0, sticky="nsew")
        self.diff_text = text_widget

        y_scroll = ttk.Scrollbar(output_frame, orient="vertical", command=text_widget.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        text_widget.configure(yscrollcommand=y_scroll.set)

        x_scroll = ttk.Scrollbar(output_frame, orient="horizontal", command=text_widget.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        text_widget.configure(xscrollcommand=x_scroll.set)

        text_widget.tag_configure("header", foreground="#0b5fff")
        text_widget.tag_configure("hunk", foreground="#0d7a8a")
        text_widget.tag_configure("add", foreground="#007d3c")
        text_widget.tag_configure("remove", foreground="#b00020")
        text_widget.tag_configure("error", foreground="#b00020")

        self._configure_drop_target(left_slot, self.assign_left_path)
        self._configure_drop_target(right_slot, self.assign_right_path)

    def _create_file_slot(
        self,
        parent: ttk.Frame,
        title: str,
        column: int,
        path_var: tk.StringVar,
        browse_command: Callable[[], None],
        assign_func: Callable[[Path], None],
    ) -> ttk.Frame:
        slot = ttk.LabelFrame(parent, text=title, padding=8)
        slot.grid(row=0, column=column, padx=(0, 8) if column == 0 else (8, 0), sticky="nsew")
        slot.columnconfigure(0, weight=1)

        path_entry = ttk.Entry(slot, textvariable=path_var)
        path_entry.grid(row=0, column=0, sticky="ew")
        path_entry.bind("<FocusOut>", lambda _event: self._path_entry_to_slot(path_var, assign_func))
        path_entry.bind("<Return>", lambda _event: self._path_entry_to_slot(path_var, assign_func))

        browse_button = ttk.Button(slot, text="Browse...", command=browse_command)
        browse_button.grid(row=0, column=1, padx=(8, 0))

        drop_hint = "Drop JSON file here" if HAS_DND else "Paste path or use Browse"
        drop_label = ttk.Label(slot, text=drop_hint, relief="solid", anchor="center")
        drop_label.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        return drop_label

    def _path_entry_to_slot(
        self,
        path_var: tk.StringVar,
        assign_func: Callable[[Path], None],
    ) -> None:
        raw_value = path_var.get().strip()
        if not raw_value:
            self._update_compare_state()
            return
        assign_func(Path(raw_value))

    def _configure_drop_target(
        self,
        widget: ttk.Widget,
        assign_func: Callable[[Path], None],
    ) -> None:
        if not HAS_DND or DND_FILES is None:
            return
        widget.drop_target_register(DND_FILES)
        widget.dnd_bind(
            "<<Drop>>",
            lambda event: self._on_drop(event_data=event.data, assign_func=assign_func),
        )

    def _on_drop(self, event_data: str, assign_func: Callable[[Path], None]) -> None:
        files = self._parse_drop_files(event_data)
        if not files:
            self.status_var.set("Drop did not contain a valid file path.")
            return
        assign_func(files[0])

    def _parse_drop_files(self, event_data: str) -> list[Path]:
        if not event_data.strip():
            return []
        split_values = self.root.tk.splitlist(event_data)
        paths: list[Path] = []
        for raw in split_values:
            candidate = Path(raw)
            if candidate.exists():
                paths.append(candidate)
        return paths

    def choose_left_file(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Select left metadata JSON file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if chosen:
            self.assign_left_path(Path(chosen))

    def choose_right_file(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Select right metadata JSON file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if chosen:
            self.assign_right_path(Path(chosen))

    def assign_left_path(self, path: Path) -> None:
        self.left_path_var.set(str(path))
        self._update_compare_state()

    def assign_right_path(self, path: Path) -> None:
        self.right_path_var.set(str(path))
        self._update_compare_state()

    def swap_files(self) -> None:
        left = self.left_path_var.get()
        right = self.right_path_var.get()
        self.left_path_var.set(right)
        self.right_path_var.set(left)
        self._update_compare_state()

    def _update_compare_state(self) -> None:
        ready = bool(self.left_path_var.get().strip()) and bool(
            self.right_path_var.get().strip()
        )
        self.compare_button.configure(state="normal" if ready else "disabled")

    def _ignored_fields(self) -> set[str]:
        ignored = set(self.manual_ignored_fields)
        if not self.no_default_ignore:
            ignored.add("script")
        return ignored

    def _on_toggle_show_full_file(self) -> None:
        if not self.left_path_var.get().strip() or not self.right_path_var.get().strip():
            return
        self.compare_files()

    def clear_diff(self) -> None:
        if self.diff_text is None:
            return
        self.diff_text.delete("1.0", "end")
        self.status_var.set("Cleared.")

    def compare_files(self) -> None:
        left_raw = self.left_path_var.get().strip()
        right_raw = self.right_path_var.get().strip()
        if not left_raw or not right_raw:
            self.status_var.set("Select both files first.")
            return

        left_path = Path(left_raw)
        right_path = Path(right_raw)
        try:
            left_payload = load_json(left_path)
            right_payload = load_json(right_path)
            ignored = self._ignored_fields()
            left_filtered = filter_ignored_fields(left_payload, ignored)
            right_filtered = filter_ignored_fields(right_payload, ignored)
            diff_lines = build_diff_lines(
                left_path=left_path,
                right_path=right_path,
                left_payload=left_filtered,
                right_payload=right_filtered,
                context=self.context_lines,
                show_full_file=self.show_full_file_var.get(),
            )
        except ValueError as exc:
            self._render_error(str(exc))
            return
        except OSError as exc:
            self._render_error(f"File error: {exc}")
            return
        except json.JSONDecodeError as exc:
            self._render_error(f"JSON error: {exc}")
            return

        self._render_diff(diff_lines)
        if diff_lines:
            self.status_var.set(f"Differences found: {len(diff_lines)} diff lines.")
        else:
            self.status_var.set("No differences found.")

    def _render_error(self, message: str) -> None:
        if self.diff_text is not None:
            self.diff_text.delete("1.0", "end")
            self.diff_text.insert("end", f"Error: {message}\n", "error")
        self.status_var.set(f"Error: {message}")
        messagebox.showerror("Metadata Diff", message)

    def _render_diff(self, diff_lines: Sequence[str]) -> None:
        if self.diff_text is None:
            return
        text_widget = self.diff_text
        text_widget.delete("1.0", "end")
        if not diff_lines:
            text_widget.insert("end", "No differences found.\n")
            return
        for line in diff_lines:
            tag = self._line_tag(line)
            text_widget.insert("end", line + "\n", tag)

    @staticmethod
    def _line_tag(line: str) -> str | None:
        if line.startswith("--- ") or line.startswith("+++ "):
            return "header"
        if line.startswith("@@"):
            return "hunk"
        if line.startswith("+"):
            return "add"
        if line.startswith("-"):
            return "remove"
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open interactive metadata JSON diff UI."
    )
    parser.add_argument("left", nargs="?", type=Path, help="Optional left JSON file.")
    parser.add_argument("right", nargs="?", type=Path, help="Optional right JSON file.")
    parser.add_argument(
        "--context",
        type=int,
        default=3,
        help="Number of unchanged context lines around each change block.",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Field name to ignore during diff. Repeat to ignore more fields.",
    )
    parser.add_argument(
        "--no-default-ignore",
        action="store_true",
        help="Disable default ignored fields (default ignored: script).",
    )
    return parser.parse_args()


def create_root() -> tk.Tk:
    if HAS_DND and TkinterDnD is not None:
        return TkinterDnD.Tk()
    return tk.Tk()


def main() -> int:
    args = parse_args()
    if (args.left is None) != (args.right is None):
        raise SystemExit("Either provide both file paths or none.")

    root = create_root()
    MetadataDiffInteractiveApp(
        root=root,
        default_left=args.left,
        default_right=args.right,
        context_lines=args.context,
        ignored_fields=args.ignore,
        no_default_ignore=args.no_default_ignore,
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
