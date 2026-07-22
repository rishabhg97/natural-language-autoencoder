"""Offline NLA Observatory evidence-generation and bundle tooling."""

from pathlib import Path
import sys


# The Observatory is packaged independently, while the model/eval primitives it
# composes still live in the repository's established flat scripts directory.
_CORE_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_CORE_SCRIPTS) not in sys.path:
    sys.path.append(str(_CORE_SCRIPTS))

__all__ = ["__version__"]

__version__ = "0.1.0"
