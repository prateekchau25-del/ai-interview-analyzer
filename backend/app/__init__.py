import sys

if not ((3, 11) <= sys.version_info[:2] <= (3, 12)):
    raise RuntimeError(
        f"AI Interview Analyzer needs Python 3.11 or 3.12 (you are running {sys.version.split()[0]}). "
        "Create the virtual environment with:  py -3.12 -m venv .venv  and activate it."
    )
