from pathlib import Path

PACKAGE_DIR = Path(__file__).parent
ROOT_DIR = PACKAGE_DIR.parent
DATA_DIR = ROOT_DIR / "data"
REPO_SEARCH_DIR = DATA_DIR / "repo_search"
REPO_DIR = DATA_DIR / "repos"
