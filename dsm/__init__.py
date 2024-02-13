import logging
import os
import runpy

from absl import app

# ruff: noqa: E402
from rich import console as rc
from rich import logging as rl
from rich import traceback

TRACEBACKS_EXCLUDES = [runpy, __file__, app]

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()

console: rc.Console = rc.Console()

traceback.install(console=console, suppress=TRACEBACKS_EXCLUDES, show_locals=False)
logging.basicConfig(
    level=LOGLEVEL,
    format="%(message)s",
    datefmt="[%X]",
    force=True,
    handlers=[
        rl.RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            tracebacks_suppress=TRACEBACKS_EXCLUDES,
        )
    ],
)

__all__: list[str] = ["console", "LOGLEVEL"]
