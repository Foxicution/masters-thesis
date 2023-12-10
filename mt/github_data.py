import json
import logging as log
from typing import Any

from mt.definitions import REPO_DIR, REPO_SEARCH_DIR
from mt.helper import api_get, flatten


def process_repo(repo: dict[str, Any]):
    log.info(f"Processing {repo['full_name']}")
    # Dumping repository data to separate repository folders
    directory = REPO_DIR / repo["full_name"]
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "repo.json", "w") as f:
        json.dump(repo, f)
    log.info("Retrieved top level info")
    # Dumping commit history to json
    commits = list(api_get(repo["commits_url"].format(**{"/sha": "?per_page=100"})))
    all_commits = flatten([response["items"] for response in commits])
    with open(directory / "commits.json", "w") as f:
        json.dump(commits, f)
    log.info(f"Retrieved {len(all_commits)} commits")
    issues = list(
        api_get(repo["issues_url"].format(**{"/number": "?per_page=100&state=all"}))
    )
    with open(directory / "issues.json", "w") as f:
        json.dump(issues, f)
    log.info("Retrieved issues")
    stars = list(api_get(f"{repo['stargazers_url']}"))
    with open(directory / "stars.json", "w") as f:
        json.dump(stars, f)
    log.info("Retrieved star info")


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    repos = list(
        api_get(
            "https://api.github.com/search/repositories?q=language:python+stars:<20000"
            "&sort=stars&order=desc&per_page=100&page=1"
        )
    )
    with open(REPO_SEARCH_DIR / "python.json", "w") as f:
        json.dump(repos, f)
    log.info(f"Finished getting {len(repos)} repos")
    for repo_list in repos:
        for repo in repo_list["items"]:
            directory = REPO_DIR / repo["full_name"]
            if directory.exists():
                continue
            try:
                process_repo(repo)
            except Exception:
                log.error(f"Failed for repo {repo['full_name']}")
