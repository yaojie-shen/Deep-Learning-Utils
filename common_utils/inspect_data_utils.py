# -*- coding: utf-8 -*-
# @Time    : 8/1/25
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : inspect_data_utils.py

__all__ = ["inspect_data"]

import argparse
import json
import os
import pickle
from functools import partial
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.text import Text
from rich.tree import Tree


def is_structurally_equal(a: Any, b: Any) -> bool:
    """Check if two Python objects have the same structure."""
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        return set(a.keys()) == set(b.keys())
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(is_structurally_equal(x, y) for x, y in zip(a, b))
    return True


def inspect_data(
    data: Any,
    max_items: int = 10,
    max_dict_items: Optional[int] = None,
    max_list_items: Optional[int] = None,
    max_depth: int = 2,
    name: Optional[str] = None,
) -> None:
    """Recursively inspects and prints the structure of a data object using a rich Tree.

    Args:
        data: The data object to inspect.
        max_items: Maximum number of items to display for each container (dict, list, tuple).
        max_dict_items: Maximum number of items to display for dictionaries.
        max_list_items: Maximum number of items to display for lists and tuples.
        max_depth: Maximum recursion depth.
        name: Optional name to label the variable.
    """
    console = Console()
    tree = _inspect_node(
        data,
        label=str(name) if name is not None else "root",
        max_items=max_items,
        max_dict_items=max_dict_items,
        max_list_items=max_list_items,
        max_depth=max_depth,
        _depth=0,
    )
    console.print(tree)


def _inspect_node(
    data: Any,
    label: str,
    max_items: int = 20,
    max_dict_items: Optional[int] = None,
    max_list_items: Optional[int] = None,
    max_depth: int = 3,
    _depth: int = 0,
) -> Tree:
    """Internal recursive helper to create a rich Tree node with colors."""
    dict_limit = max_dict_items if max_dict_items is not None else max_items
    list_limit = max_list_items if max_list_items is not None else max_items
    kwargs = dict(
        max_items=max_items,
        max_dict_items=max_dict_items,
        max_list_items=max_list_items,
        max_depth=max_depth,
    )

    def preview_text(val: Any) -> str:
        string = str(val)
        return string[:100] + "..." if len(string) > 100 else string

    def colorize_label(name: str, type_str: str) -> Text:
        return Text.assemble(
            (f"{name}", "cyan"), (": ", "dim"), (f"{type_str}", "bold yellow")
        )

    if _depth > max_depth:
        return Tree(
            Text.assemble((f"{label}: ", "cyan"), ("<Max depth reached>", "red"))
        )

    if isinstance(data, dict):
        branch = Tree(colorize_label(label, f"dict (len={len(data)})"))
        for i, (k, v) in enumerate(data.items()):
            if i >= dict_limit:
                branch.add(
                    Text(f"... ({len(data) - dict_limit} more keys)", style="red")
                )
                break
            child = _inspect_node(v, label=repr(k), _depth=_depth + 1, **kwargs)
            branch.add(child)
        return branch

    elif isinstance(data, (list, tuple)):
        label_type = "list" if isinstance(data, list) else "tuple"

        # Check for simple scalar items
        simple_types = (int, float, bool, str)
        if (
            all(isinstance(x, simple_types) for x in data)
            and all(not isinstance(x, str) or len(x) < 30 for x in data)
            and len(data) <= max(10, list_limit)  # only summarize short sequences
        ):
            preview = [repr(x) for x in data]
            return Tree(
                Text.assemble(
                    (f"{label}: ", "cyan"),
                    (f"{label_type} ", "bold magenta"),
                    ("[" + ", ".join(preview) + "]", "green"),
                )
            )

        # Regular recursive display
        branch = Tree(colorize_label(label, f"{label_type} (len={len(data)})"))
        if len(data) == 0:
            return branch

        first = data[0]
        repeated = all(is_structurally_equal(first, item) for item in data[1:10])
        if repeated:
            child = _inspect_node(
                first, label="[0] (representative)", _depth=_depth + 1, **kwargs
            )
            branch.add(child)
            branch.add(
                Text(
                    f"... (remaining {len(data) - 1} items have same structure)",
                    style="dim",
                )
            )
        else:
            for i, item in enumerate(data[:list_limit]):
                child = _inspect_node(item, label=f"[{i}]", _depth=_depth + 1, **kwargs)
                branch.add(child)
            if len(data) > list_limit:
                branch.add(
                    Text(f"... ({len(data) - list_limit} more items)", style="red")
                )
        return branch

    elif isinstance(data, torch.Tensor):
        preview = data.flatten()[:5].tolist()
        text = Text.assemble(
            (f"{label}: ", "cyan"),
            ("torch.Tensor ", "magenta"),
            (
                f"(shape={tuple(data.shape)}, dtype={data.dtype}, device={data.device})",
                "dim",
            ),
            ("\n  values: ", "dim"),
            (str(preview), "green" if data.numel() <= 5 else "green bold"),
            (" ..." if data.numel() > 5 else "", "dim"),
            (
                f"\n  std: {data.float().std():5.2e} mean: {data.float().mean():5.2e}",
                "dim",
            ),
            (f" min: {data.min():5.2e} max: {data.max():5.2e}", "dim"),
        )
        return Tree(text)

    elif isinstance(data, np.ndarray):
        preview = data.flatten()[:5].tolist()
        text = Text.assemble(
            (f"{label}: ", "cyan"),
            ("np.ndarray ", "magenta"),
            (f"(shape={data.shape}, dtype={data.dtype})", "dim"),
            ("\n  values: ", "dim"),
            (str(preview), "green" if data.size <= 5 else "green bold"),
            (" ..." if data.size > 5 else "", "dim"),
            (f"\n  std: {data.std():5.2e} mean: {data.mean():5.2e}", "dim"),
            (f" min: {data.min():5.2e} max: {data.max():5.2e}", "dim"),
        )
        return Tree(text)

    elif isinstance(data, pd.DataFrame):
        head = data.head(3).to_string().replace("\n", "\n  ")
        text = Text.assemble(
            (f"{label}: ", "cyan"),
            ("pandas.DataFrame", "magenta"),
            (f" (shape={data.shape})", "dim"),
            ("\n  columns: ", "dim"),
            (str(list(data.columns)), "green"),
            ("\n  preview:\n  ", "dim"),
            (head, "white"),
        )
        return Tree(text)

    elif isinstance(data, str):
        return Tree(
            Text.assemble((f"{label}: ", "cyan"), (f"'{preview_text(data)}'", "green"))
        )

    elif isinstance(data, (int, float, bool)):
        return Tree(Text.assemble((f"{label}: ", "cyan"), (str(data), "yellow")))

    else:
        return Tree(
            Text.assemble((f"{label}: ", "cyan"), (preview_text(data), "white"))
        )


def main() -> None:
    """Load a data file and inspect its contents."""
    parser = argparse.ArgumentParser(description="Inspect contents of data files.")
    parser.add_argument("file", type=str, help="Path to the data file")
    parser.add_argument(
        "--format",
        choices=["auto", "torch", "csv", "json", "pkl"],
        default="auto",
        help="Specify file format (default: auto-detect)",
    )
    parser.add_argument("--depth", type=int, default=2, help="Recursion depth")
    parser.add_argument(
        "--items", type=int, default=10, help="Default max items to display"
    )
    parser.add_argument(
        "--dict",
        dest="dict_items",
        type=int,
        default=None,
        help="Override max dict items",
    )
    parser.add_argument(
        "--list",
        dest="list_items",
        type=int,
        default=None,
        help="Override max list items",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Drop into IPython shell after loading",
    )
    args = parser.parse_args()

    file_path = args.file
    file_ext = os.path.splitext(file_path)[-1].lower()

    try:
        if args.format == "torch" or (
            args.format == "auto" and file_ext in [".pt", ".pth"]
        ):
            data = torch.load(file_path, map_location="cpu")
        elif args.format == "csv" or (args.format == "auto" and file_ext == ".csv"):
            data = pd.read_csv(file_path)
        elif args.format == "json" or (args.format == "auto" and file_ext == ".json"):
            with open(file_path, "r") as f:
                data = json.load(f)
        elif args.format == "pkl" or (
            args.format == "auto" and file_ext in [".pkl", ".pickle"]
        ):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
        else:
            print(f"❌ Unsupported file format: {args.format} or extension: {file_ext}")
            return
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        exit(1)

    if not args.interactive:
        print(f"\n📦 Inspecting file: {file_path}\n")
        inspect_data(
            data,
            max_items=args.items,
            max_dict_items=args.dict_items,
            max_list_items=args.list_items,
            max_depth=args.depth,
        )
    else:
        try:
            from IPython import embed

            # This function is for interactive mode
            def save_as(new_file_path: str):
                if args.format == "torch" or (
                    args.format == "auto" and file_ext in [".pt", ".pth"]
                ):
                    torch.save(data, new_file_path)
                elif args.format == "csv" or (
                    args.format == "auto" and file_ext == ".csv"
                ):
                    data.to_csv(new_file_path, index=False)
                elif args.format == "json" or (
                    args.format == "auto" and file_ext == ".json"
                ):
                    with open(new_file_path, "w") as f:
                        json.dump(data, f, indent=4)
                elif args.format == "pkl" or (
                    args.format == "auto" and file_ext in [".pkl", ".pickle"]
                ):
                    with open(new_file_path, "wb") as f:
                        pickle.dump(data, f)
                else:
                    raise RuntimeError()

            # This function is for interactive mode
            def save():
                save_as(file_path)

            inspect = partial(inspect_data, data)

            embed(
                header="🔍 Entering IPython shell. You can explore the variable `data`.\n"
                "\n"
                "Basic Usage:\n"
                "  - `data` to access the loaded data\n"
                "  - `inspect()` to inspect the data structure\n"
                "  - `exit()` to exit the shell\n"
                "\n"
                "Modifying Data:\n"
                "  - Edit the `data` variable in the shell to modify the data\n"
                "  - `save()` to save the modified data back to the file\n"
                "  - `save_as('new_file_path')` to save the modified data to a new file\n"
            )
        except ImportError:
            print(
                "❌ IPython not installed. Run `pip install ipython` to use interactive mode."
            )


if __name__ == "__main__":
    main()
