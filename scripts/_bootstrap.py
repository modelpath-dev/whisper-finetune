"""Make ``src/`` importable when running scripts directly (no install needed).

Importing this module as the first line of a script adds the project's ``src``
directory to ``sys.path`` so ``import whisper_hi`` works whether or not the
package has been ``pip install -e .``'d. This is handy on Colab.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
