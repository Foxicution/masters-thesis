import json
import logging as log
import os
import subprocess
from pathlib import Path

from mt.definitions import REPO_DIR, ROOT_DIR
from mt.featurize_repo import repo_to_file_features

REQUIRED_FILES = ["commits.json", "issues.json", "repo.json", "stars.json"]


def delete_empty_files(repo_dir: Path):
    for file in REQUIRED_FILES:
        file_path = repo_dir / file
        if file_path.exists():
            with open(file_path) as f:
                data = json.load(f)
                size = len(data)
            # If the file is empty (size is 0), delete it
            if not size:
                log.info(f"Deleting empty file: {file_path}")
                file_path.unlink()


def clone_repo(repo_dir: Path, repo_url: str):
    if not (repo_dir / "repo").exists():
        log.info(f"Cloning {repo_url}")
        os.chdir(repo_dir)
        subprocess.run(["git", "clone", repo_url, "repo"], check=True)
        os.chdir(ROOT_DIR)


def process_repo_dir(repo_dir: Path) -> None:
    with open(repo_dir / "repo.json") as f:
        repo_data = json.load(f)
    log.info(f"Processing {repo_data['html_url']}")
    clone_repo(repo_dir, repo_data["html_url"])
    if not (feature_path := repo_dir / "features.json").exists():
        repo = repo_dir / "repo"
        with open(feature_path, "w") as f:
            json.dump(repo_to_file_features(repo), f)


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    for org_dir in REPO_DIR.iterdir():
        for repo_dir in org_dir.iterdir():
            if (feat := repo_dir / "features.json").exists():
                feat.unlink()
            delete_empty_files(repo_dir)
            if all((repo_dir / file).exists() for file in REQUIRED_FILES):
                process_repo_dir(repo_dir)
