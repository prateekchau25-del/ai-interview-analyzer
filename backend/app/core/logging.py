import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )
    for noisy in ("httpx", "urllib3", "numba", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
