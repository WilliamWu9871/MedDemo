#!/usr/bin/env python3
"""Patch an ipykernel kernel.json so it ignores host /etc/ipython Colab configs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ISOLATED_LAUNCHER_CODE = (
    "import sys, IPython.core.application as _app; "
    "_app.SYSTEM_CONFIG_DIRS[:] = []; "
    "sys.path = [p for p in sys.path if p not in ('', '.')]; "
    "from ipykernel import kernelapp; "
    "kernelapp.launch_new_instance()"
)


def patch_kernel_json(kernel_json_path: Path) -> bool:
    """Rewrite ipykernel_launcher argv in kernel.json to bypass host SYSTEM_CONFIG_DIRS."""
    data = json.loads(kernel_json_path.read_text(encoding="utf-8"))
    argv = data.get("argv", [])
    if argv and "-m" in argv and "ipykernel_launcher" in argv:
        data["argv"] = [argv[0], "-c", ISOLATED_LAUNCHER_CODE, "-f", "{connection_file}"]
        kernel_json_path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        return True
    return False


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <kernel_name>")
    kernel_name = sys.argv[1]
    from jupyter_client.kernelspec import KernelSpecManager

    spec = KernelSpecManager().get_kernel_spec(kernel_name)
    kj_path = Path(spec.resource_dir) / "kernel.json"
    if patch_kernel_json(kj_path):
        print(f"[patch_kernelspec] Patched '{kernel_name}' at {kj_path} to bypass host /etc/ipython config.")


if __name__ == "__main__":
    main()
