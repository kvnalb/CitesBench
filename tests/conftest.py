import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_REVIEW_DIR = REPO_ROOT / "paper_review"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

if str(PAPER_REVIEW_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_REVIEW_DIR))


def _load_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_aggregate_module():
    return _load_module(
        "aggregate_persona_reviews",
        PAPER_REVIEW_DIR / "08_aggregate_persona_reviews.py",
    )


def load_editorial_ab_module():
    return _load_module(
        "exp_editorial_dedup_ab",
        EXPERIMENTS_DIR / "exp_editorial_dedup_ab.py",
    )
