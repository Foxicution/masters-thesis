import concurrent.futures
import json
import pickle
import re
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import requests
import torch
from dateutil.parser import parse
from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from mt.definitions import DATA_DIR, REPO_DIR
from mt.helper import flatten

# EDGE_TYPE_PATH = DATA_DIR / "type_to_int.json"


# def get_node_type_to_int() -> dict[str, int]:
#     if (path := EDGE_TYPE_PATH).exists():
#         with open(path) as f:
#             node_type_to_int = json.load(f)
#     else:
#         response = requests.get(
#             "https://raw.githubusercontent.com/tree-sitter/tree-sitter-python/master/src/node-types.json"
#         )
#         types = re.findall(r'"type": "(.+)"', response.text)
#         node_type_to_int = {t: i + 1 for i, t in enumerate(list(set(types)))}
#         with open(EDGE_TYPE_PATH, "w") as f:
#             json.dump(node_type_to_int, f)
#     return node_type_to_int


# node_type_to_int = get_node_type_to_int()
# edge_type_to_int = {
#     "child": 0,
#     "occurance_of": 1,
#     "may_next_use": 2,
# }


def get_commit_paths(repo: Path) -> dict[str, Path]:
    commit_data_dir = repo / "commit_data"
    commit_paths = list(commit_data_dir.glob("*.json"))
    commit_paths.sort(key=lambda path: int(path.name.split("_")[0]))
    commit_paths = {
        path.name.split("_")[1].removesuffix(".json"): path for path in commit_paths
    }
    return commit_paths


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


def fit_reg_calc_res(
    no_issues: dict[str, int], no_stars: dict[str, int]
) -> dict[str, int]:
    residuals = {}
    stars = np.array(list(no_stars.values())).reshape(-1, 1)
    issues = np.array(list(no_issues.values()))

    model = LinearRegression()
    model.fit(stars, issues)

    predicted_issues = model.predict(stars)
    residuals_list = issues - predicted_issues

    for sha, residual in zip(no_stars.keys(), residuals_list):
        residuals[sha] = residual

    return residuals


def process_commit(
    commit: dict[str, Any],
    path: Path,
    issues: dict[str, Any],
    stars: dict[str, Any],
) -> tuple[str, int, int]:
    "returns msg, no_issues, no_stars"
    commit_date = parse(commit["commit"]["author"]["date"]).replace(tzinfo=None)
    with open(path) as f:
        raw_data = json.load(f)

    return (
        f"{commit['sha']} Processed",
        number_of_issues_open(issues, commit_date),
        number_of_stars_at(stars, commit_date),
    )


def process_repo(repo: Path) -> None:
    with open(repo / "stars.json") as f:
        stars = flatten([page["items"] for page in json.load(f)])

    with open(repo / "commits.json") as f:
        commits = flatten([page["items"] for page in json.load(f)])

    with open(repo / "issues.json") as f:
        issues = flatten([page["items"] for page in json.load(f)])

    pt_dir = repo / "pts"
    pt_dir.mkdir(exist_ok=True)
    commit_paths = get_commit_paths(repo)

    all_issues, all_stars = {}, {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as pool:
        futures, counter = {}, 0
        for commit in commits:
            if path := commit_paths.get(commit["sha"]):
                futures[
                    pool.submit(process_commit, commit, path, issues, stars)
                ] = commit["sha"]
                counter += 1

        for future in concurrent.futures.as_completed(futures):
            sha = futures[future]
            msg, no_issues, no_stars = future.result()
            all_issues[sha] = no_issues
            all_stars[sha] = no_stars
            print(msg)

    # residuals = fit_reg_calc_res(all_issues, all_stars)

    with open(repo / "no_stars.pkl", "wb") as f:
        pickle.dump(all_stars, f)

    with open(repo / "no_issues.pkl", "wb") as f:
        pickle.dump(all_issues, f)

    # with open(repo / "residuals.pkl", "wb") as f:
    #     pickle.dump(residuals, f)


if __name__ == "__main__":
    repo = REPO_DIR / "pytorch/vision"
    process_repo(repo)
