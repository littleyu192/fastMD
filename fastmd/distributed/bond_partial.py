"""Moved to ``fastmd.models.matris.distributed.bond_partial``.

This old path is the same module object (kept for compatibility)."""
import sys

_NEW = "fastmd.models.matris.distributed.bond_partial"

if __name__ == "__main__":
    import runpy

    runpy.run_module(_NEW, run_name="__main__", alter_sys=True)
else:
    import importlib

    sys.modules[__name__] = importlib.import_module(_NEW)
