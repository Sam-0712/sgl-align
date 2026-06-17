"""viz -- text-similarity visualisation tool (HTML frontend + HTTP server).

Run standalone::

    python -m viz.app
    # opens http://127.0.0.1:7421 in the browser
"""

from viz.app import main, compute_alignment

__all__ = ["main", "compute_alignment"]
