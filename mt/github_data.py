import json
import logging as log
from pathlib import Path
from typing import Any

from mt.definitions import REPO_DIR, REPO_SEARCH_DIR
from mt.helper import api_get, flatten


def process_repo(repo: dict[str, Any]):
    log.info(f"Processing {repo['full_name']}")
    directory: Path = REPO_DIR / repo["full_name"]
    directory.mkdir(parents=True, exist_ok=True)

    repo_path = directory / "repo.json"
    commit_path = directory / "commits.json"
    issue_path = directory / "issues.json"
    star_path = directory / "stars.json"

    # Dumping repository data to separate repository folders
    if not repo_path.exists():
        with open(repo_path, "w") as f:
            json.dump(repo, f)
    log.info("Retrieved top level info")

    if commit_path.exists():
        with open(commit_path) as f:
            commits = json.load(f)
    if (not commit_path.exists()) or (not commits):
        commits = list(api_get(repo["commits_url"].format(**{"/sha": "?per_page=100"})))
        all_commits = flatten([response["items"] for response in commits])
        with open(commit_path, "w") as f:
            json.dump(commits, f)
        log.info(f"Retrieved {len(all_commits)} commits")

    if issue_path.exists():
        with open(issue_path) as f:
            issues = json.load(f)
    if (not issue_path.exists()) or (not issues):
        issues = list(
            api_get(repo["issues_url"].format(**{"/number": "?per_page=100&state=all"}))
        )
        with open(issue_path, "w") as f:
            json.dump(issues, f)
        log.info("Retrieved issues")

    if star_path.exists():
        with open(star_path) as f:
            stars = json.load(f)
    if (not star_path.exists()) or (not stars):
        stars = list(api_get(f"{repo['stargazers_url']}"))
        with open(star_path, "w") as f:
            json.dump(stars, f)
        log.info("Retrieved star info")


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    search_file = REPO_SEARCH_DIR / "python.json"
    if not search_file.exists():
        repos = list(
            api_get(
                "https://api.github.com/search/repositories?q=language:python+stars:<20000"
                "&sort=stars&order=desc&per_page=100&page=1"
            )
        )
        with open(search_file, "w") as f:
            json.dump(repos, f)
        log.info(f"Finished getting {len(repos)} repo pages")
    else:
        with open(search_file) as f:
            repos = json.load(f)
        log.info(f"Loaded {len(repos)} pages from file")

    for repo_list in repos:
        for repo in repo_list["items"]:
            directory = REPO_DIR / repo["full_name"]
            if directory.exists():
                continue
            try:
                process_repo(repo)
            except Exception:
                log.excpetion(f"Failed for repo {repo['full_name']}")
