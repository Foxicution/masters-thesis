from pathlib import Path

PACKAGE_DIR = Path(__file__).parent
ROOT_DIR = PACKAGE_DIR.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
REPO_SEARCH_DIR = DATA_DIR / "repo_search"
REPO_SEARCH_DIR.mkdir(exist_ok=True)
REPO_DIR = DATA_DIR / "repos"
REPO_DIR.mkdir(exist_ok=True)
DATASET_DIR = DATA_DIR / "datasets"
DATASET_DIR.mkdir(exist_ok=True)
