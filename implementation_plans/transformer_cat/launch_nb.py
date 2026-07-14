"""
Launcher for pipeline_validation.ipynb, pinned to the repo's uv-managed .venv.

Two problems this solves:
1. Ensures JupyterLab always runs with the project's .venv interpreter,
   never the system Python — regardless of which `python` happens to be
   first on PATH when this script is invoked.
2. Works around a Python 3.14 / Windows bug: subprocess.Popen raises
   ValueError for env variable names starting with '=' (Windows-internal
   per-drive vars like =C:, =D:). jupyter_client does its own
   os.environ.copy() when spawning the kernel, so the fix must happen in
   THIS process — before any JupyterLab code is imported — and this
   process must itself already be running under .venv's interpreter.

Usage (from any directory, with any Python on PATH):
    python implementation_plans/transformer_cat/launch_nb.py
"""
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent  # implementation_plans/transformer_cat -> repo root
_VENV_PYTHON = _REPO_ROOT / ".venv" / "Scripts" / "python.exe"
_NOTEBOOK = _HERE / "pipeline_validation.ipynb"


def _relaunch_under_venv() -> None:
    """Re-exec this script using .venv's python if we aren't already."""
    if not _VENV_PYTHON.exists():
        raise SystemExit(
            f"Expected venv interpreter not found at {_VENV_PYTHON}. "
            "Run 'uv sync --extra transformer-cat' from the repo root first."
        )
    if Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
        os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON), str(Path(__file__).resolve())])


_relaunch_under_venv()

# --- From here on, sys.executable is guaranteed to be .venv's python.exe ---

# Strip Windows-internal =C: / =D: style vars from THIS process's environ.
# jupyter_client will later call os.environ.copy() to build the kernel env;
# this ensures that copy is already clean.
for _k in list(os.environ.keys()):
    if not _k or _k.startswith("="):
        del os.environ[_k]

sys.argv = ["jupyter-lab", f"--notebook-dir={_HERE}"]
os.chdir(_HERE)

from jupyterlab.labapp import LabApp  # noqa: E402

LabApp.launch_instance()
