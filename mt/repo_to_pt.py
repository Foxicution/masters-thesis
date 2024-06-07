import concurrent.futures
import json
import pickle
import re
from datetime import datetime
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
from functools import cache

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


g_types = {"ast": 1, "ch22": 2, "cfg": 3, "dg": 4, "cfg_ast": 5}


def nx_to_pyg_graph(nx_graph: nx.DiGraph, maps: dict) -> Data:
    node_mapping = {node: i for i, node in enumerate(nx_graph.nodes())}
    # original_ids = {i: node for node, i in node_mapping.items()}

    # Combine node type and node_type attributes
    x = [
        [
            maps["nodes"].get(nx_graph.nodes[node].get("type", "unk"), 0),
            g_types.get(nx_graph.nodes[node].get("node_type", "unk"), 0),
        ]
        for node in nx_graph.nodes
    ]
    x = torch.tensor(x, dtype=torch.long)  # Shape (num_nodes, 2)

    edge_list = []
    edge_attr = []
    for u, v, data in nx_graph.edges(data=True):
        edge_list.append([node_mapping[u], node_mapping[v]])
        # Combine edge type and edge_type attributes
        edge_attr.append(
            [maps["edges"].get(data["type"], 0), g_types.get(data["edge_type"], 0)]
        )

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)  # Shape (num_edges, 2)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def path_to_graph(path: Path) -> nx.DiGraph:
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G


def path_to_dict(path: Path) -> dict[str, str]:
    with open(path) as f:
        d = json.load(f)
    return d


def file_features_to_graph(
    features: dict[str, str | bool | dict[str, int] | list[str]]
) -> tuple[Data, dict[int, str]]:
    with open(features["feature_file"]) as f:
        nx_graph = nx.node_link_graph(json.load(f)["ast"])
        return nx_to_pyg_graph(nx_graph)


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


def construct_graphs(
    graph_list: list[nx.DiGraph], maps
) -> tuple[nx.DiGraph, nx.DiGraph, nx.DiGraph, nx.DiGraph, nx.DiGraph]:
    combined_graph = nx.DiGraph()
    ast_graph = nx.DiGraph()
    dg_graph = nx.DiGraph()
    cfg_graph = nx.DiGraph()
    ch22_graph = nx.DiGraph()

    for G in graph_list:
        for n, d in G.nodes(data=True):
            combined_graph.add_node(n, **d)
            node_type = d.get("node_type")
            if node_type == "ast":
                ast_graph.add_node(n, **d)
                ch22_graph.add_node(n, **d)
            elif node_type == "dg":
                dg_graph.add_node(n, **d)
            elif node_type == "cfg":
                cfg_graph.add_node(n, **d)
            elif node_type == "ch22":
                ch22_graph.add_node(n, **d)

        for u, v, d in G.edges(data=True):
            combined_graph.add_edge(u, v, **d)
            edge_type = d.get("edge_type")
            if edge_type == "ast":
                ast_graph.add_edge(u, v, **d)
            elif edge_type == "dg":
                dg_graph.add_edge(u, v, **d)
            elif edge_type == "cfg":
                cfg_graph.add_edge(u, v, **d)
            elif edge_type == "ch22":
                ch22_graph.add_edge(u, v, **d)

    combined_graph = nx_to_pyg_graph(combined_graph, maps["all"])
    ast_graph = nx_to_pyg_graph(ast_graph, maps["ast"])
    dg_graph = nx_to_pyg_graph(dg_graph, maps["dg"])
    cfg_graph = nx_to_pyg_graph(cfg_graph, maps["cfg"])
    ch22_graph = nx_to_pyg_graph(ch22_graph, maps["ch22"])

    return combined_graph, ast_graph, dg_graph, cfg_graph, ch22_graph


def process_commit(
    commit: dict[str, Any],
    path: Path,
    idx: int,
    pt_dir: Path,
    # issues: dict[str, Any],
    # stars: dict[str, Any],
    maps: dict[str, dict[str, dict[str, int]]],
) -> tuple[str, int, int]:
    "returns msg, no_issues, no_stars"
    # commit_date = parse(commit["commit"]["author"]["date"]).replace(tzinfo=None)
    with open(path) as f:
        raw_data = json.load(f)

    feature_paths = (features["feature_file"] for features in raw_data.values())
    feature_files = (path_to_dict(path) for path in feature_paths)
    graphs, static = [], []
    for file in feature_files:
        graphs.append(file.pop("ast"))
        static.append(file)
    # G = nx.compose_all((path_to_graph(path) for path in graphs))
    ALL, AST, DG, CFG, CH22 = construct_graphs(
        (path_to_graph(path) for path in graphs), maps
    )

    torch.save((commit["sha"], ALL), pt_dir / f"all_{idx}.pt")
    # with open(pt_dir / f"map_all_{idx}.pkl", "wb") as f:
    #     pickle.dump(ALL_map, f)

    torch.save((commit["sha"], AST), pt_dir / f"ast_{idx}.pt")
    # with open(pt_dir / f"map_ast_{idx}.pkl", "wb") as f:
    #     pickle.dump(AST_map, f)

    torch.save((commit["sha"], DG), pt_dir / f"dg_{idx}.pt")
    # with open(pt_dir / f"map_dg_{idx}.pkl", "wb") as f:
    #     pickle.dump(DG_map, f)

    torch.save((commit["sha"], CFG), pt_dir / f"cfg_{idx}.pt")
    # with open(pt_dir / f"map_cfg_{idx}.pkl", "wb") as f:
    #     pickle.dump(CFG_map, f)

    torch.save((commit["sha"], CH22), pt_dir / f"ch22_{idx}.pt")
    # with open(pt_dir / f"map_ch22_{idx}.pkl", "wb") as f:
    #     pickle.dump(CH22_map, f)

    all_out = []
    for feature in static:
        out_feat = []
        out_feat.extend(feature["loc"].values())
        out_feat.append(feature["complexity"])
        out_feat.append(len(feature["dependencies"]))
        match = re.search("Your code has been rated at (\d\.\d\d)", feature["quality"])
        if not match:
            out_feat.append(-1.0)
        else:
            out_feat.append(float(match.group(1)))
        all_out.append(out_feat)

    static = np.mean(np.array(all_out), axis=0)
    np.save(pt_dir / f"f_{idx}.npy", static)

    # number_of_issues_open(issues, commit_date),
    # number_of_stars_at(stars, commit_date),
    return f"{commit['sha']} Processed"


def process_repo(repo: Path) -> None:
    # with open(repo / "stars.json") as f:
    #     stars = flatten([page["items"] for page in json.load(f)])

    with open(repo / "commits.json") as f:
        commits = flatten([page["items"] for page in json.load(f)])

    # with open(repo / "issues.json") as f:
    #     issues = flatten([page["items"] for page in json.load(f)])

    with open(repo / "maps.json") as f:
        maps = json.load(f)

    pt_dir = repo / "pts"
    pt_dir.mkdir(exist_ok=True)
    commit_paths = get_commit_paths(repo)

    # all_issues, all_stars = {}, {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=9) as pool:
        futures, counter = {}, 0
        for commit in commits:
            if path := commit_paths.get(commit["sha"]):
                futures[
                    pool.submit(process_commit, commit, path, counter, pt_dir, maps)
                ] = commit["sha"]
                counter += 1

        for future in concurrent.futures.as_completed(futures):
            # sha = futures[future]
            msg = future.result()
            # all_issues[sha] = no_issues
            # all_stars[sha] = no_stars
            print(msg)

    # residuals = fit_reg_calc_res(all_issues, all_stars)

    # with open(pt_dir / "no_stars.pkl", "wb") as f:
    #     pickle.dump(all_stars, f)

    # with open(pt_dir / "no_issues.pkl", "wb") as f:
    #     pickle.dump(all_issues, f)

    # with open(pt_dir / "residuals.pkl", "wb") as f:
    #     pickle.dump(residuals, f)


if __name__ == "__main__":
    repo = REPO_DIR / "pytorch/vision"
    process_repo(repo)
