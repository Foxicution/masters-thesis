from mt.definitions import REPO_DIR, DATA_DIR
import json
from datetime import datetime
from dateutil.parser import parse
from typing import Any
from mt.helper import flatten
from mt.pull_repos import delete_empty_files


def issue_open_at(issue: dict[str, Any], date: datetime) -> bool:
    created_at = parse(issue["created_at"])
    closed_at = parse(issue["closed_at"]) if issue["closed_at"] else None
    return created_at.replace(tzinfo=None) < date and (
        not closed_at or closed_at.replace(tzinfo=None) > date
    )


def number_of_issues_open(issues: dict[str, Any], date: datetime) -> int:
    return sum([1 for issue in issues if issue_open_at(issue, date)])


def number_of_stars_at(stars: list[dict[str, Any]], date: datetime) -> int:
    count = 0
    for star in stars:
        if parse(star["starred_at"]).replace(tzinfo=None) <= date:
            count += 1
    return count


Y = []
Y_scale = []
feature_set = []

for org_dir in REPO_DIR.iterdir():
    for repo_dir in org_dir.iterdir():
        print(repo_dir)
        # delete_empty_files(repo_dir)
        if (feat := repo_dir / "features.json").exists():
            with open(feat) as f:
                features = json.load(f)
            with open(repo_dir / "issues.json") as f:
                issues = json.load(f)
            with open(repo_dir / "stars.json") as f:
                stars = json.load(f)
            with open(repo_dir / "commits.json") as f:
                commits = json.load(f)
            with open(repo_dir / "repo.json") as f:
                repo = json.load(f)

            commits = flatten([commit_page["items"] for commit_page in commits])
            issues = flatten([issue_page["items"] for issue_page in issues])
            stars = flatten([star_page["items"] for star_page in stars])
            print(len(stars), repo["stargazers_count"])
            if abs(len(stars) - repo["stargazers_count"]) < 5:
                for commit in commits:
                    if feats := features.get(commit["sha"]):
                        commit_date = parse(commit["commit"]["author"]["date"]).replace(
                            tzinfo=None
                        )

                        Y.append(number_of_issues_open(issues, commit_date))
                        Y_scale.append(number_of_stars_at(stars, commit_date))
                        feature_set.append(feats)


# Save feature_set as JSON
with open(DATA_DIR / "dirty_features" / "feature_set.json", "w") as f:
    json.dump(feature_set, f)

# Save Y and Y_scale as JSON
with open(DATA_DIR / "dirty_features" / "Y.json", "w") as f:
    json.dump(Y, f)

with open(DATA_DIR / "dirty_features" / "Y_scale.json", "w") as f:
    json.dump(Y_scale, f)
