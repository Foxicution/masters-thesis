import ast
import pickle
import subprocess
from hashlib import sha1
from pathlib import Path
from typing import Any

import networkx as nx
import tree_sitter
from function_pipes import pipe
from nltk.tokenize import word_tokenize
from pydriller import Commit, Repository
from radon.complexity import average_complexity, cc_visit
from tree_sitter_languages import get_parser

from mt.definitions import DATA_DIR

repo_path = DATA_DIR / "test" / "python"
parser = get_parser("python")


def read_file_contents(file_path: Path) -> bytes:
    """Read the content of a file"""
    try:
        return file_path.read_bytes()
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return b""


####################################################################################
# Simple metrics
def count_lines_of_code(file_content: str) -> dict[str, int]:
    lines = file_content.split("\n")
    total_lines = len(lines)
    code_lines = len(
        [line for line in lines if line.strip() and not line.strip().startswith("#")]
    )
    comment_lines = len([line for line in lines if line.strip().startswith("#")])
    blank_lines = total_lines - code_lines - comment_lines
    return {
        "total": total_lines,
        "code": code_lines,
        "comments": comment_lines,
        "blanks": blank_lines,
    }


# TODO: Add typing here
def calculate_cyclomatic_complexity(file_content):
    blocks = cc_visit(file_content)
    if blocks:
        return average_complexity(blocks)
    else:
        return 0  # Return 0 or an appropriate value for files with no blocks


def check_code_quality(file_path: Path) -> str:
    result = subprocess.run(["pylint", file_path], capture_output=True, text=True)
    return result.stdout


# TODO: Compare efficiency of this vs treesitter and consolidate into one
def analyze_dependencies(file_content: str) -> list[str]:
    tree = ast.parse(file_content)
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom)
    ]
    return [node.names[0].name for node in imports]


def tokenize_comments(file_content: str) -> list[list[str]]:
    comments = [
        line for line in file_content.split("\n") if line.strip().startswith("#")
    ]
    tokens = [word_tokenize(comment) for comment in comments]
    return tokens


#################################################################################
# Filtering for anon nodes is necessary to avoid a lot of noise and non important features?
def tree_to_graph(root: tree_sitter.Node, with_anon: bool = False) -> nx.DiGraph:
    G = nx.DiGraph()
    todo = [root]
    while todo:
        node = todo.pop()
        if with_anon or node.is_named:
            G.add_node(node.id, type=node.type)
        for child in node.children:
            if with_anon or child.is_named:
                G.add_edge(node.id, child.id)
            todo.append(child)
    return G


def get_tree_from_code(code: bytes) -> nx.DiGraph:
    return pipe(code, parser.parse, lambda tree: tree.root_node, tree_to_graph)


def save_graph_to_pickle(graph: nx.Graph, path: Path):
    with open(path, "wb") as f:
        pickle.dump(graph, f)


# TODO: https://chat.openai.com/c/f858a84a-0811-4187-b0c5-97765b2eab11
# AST trees
def bytes_to_ast(file_content: bytes) -> dict:
    "Reads the file content and returns the AST as a json nx graph."
    # TODO: if not efficient enough, transform to .pkl files
    return pipe(file_content, get_tree_from_code, nx.node_link_data)


def process_file(file_path: Path, file_bytes: bytes, file_text: str) -> dict[str, Any]:
    features = {
        "loc": count_lines_of_code(file_text),
        "complexity": calculate_cyclomatic_complexity(file_text),
        "dependencies": analyze_dependencies(file_text),
        "quality": check_code_quality(file_path),
        "comment_tokens": tokenize_comments(file_text),
        "ast": bytes_to_ast(file_bytes),
    }
    return features


def process_commit(
    commit: Commit,
    repo_path: Path,
    last_commit: dict[str, Any] | None,
) -> dict:
    # Checkout commit
    try:
        subprocess.run(
            ["git", "-C", str(repo_path), "checkout", commit.hash],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error checking out commit {commit.hash}: {e}")
        return {}

    # Because pydriller doesn't process merge commits we do our own content processing
    file_features = {}
    for file in repo_path.glob("**/*.py"):
        if file.suffix == "py":
            try:
                file_bytes = read_file_contents(file)
                file_text = file_bytes.decode()

                content_hash = sha1(file_bytes, usedforsecurity=False)
                if (last_file_features := last_commit.get(str(file))) is not None:
                    if last_file_features["content_hash"] == content_hash:
                        file_features[str(file)] = {
                            "content_hash": content_hash,
                            "changed": False,
                        }
                        continue
                file_features[str(file)] = process_file(file, file_bytes, file_text) | {
                    "content_hash": content_hash,
                    "changed": True,
                }
            except Exception as e:
                print(f"Error processing file {file.name}: {e}")
    return file_features


def get_default_branch(repo_path: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "branch", "-r"],
            capture_output=True,
            text=True,
            check=True,
        )
        branches = result.stdout.split()
        # Look for the most common default branch names
        for branch in ["origin/main", "origin/master"]:
            if branch in branches:
                return branch.split("/")[1]  # Return the branch name without "origin/"
        # Fallback strategy
        if branches:
            return branches[0].split("/")[1]  # Return the first branch found
    except subprocess.CalledProcessError as e:
        print(f"Error listing branches: {e}")
    return None


def checkout_default_branch(repo_path: str) -> None:
    default_branch = get_default_branch(repo_path)
    if default_branch:
        subprocess.run(["git", "-C", repo_path, "checkout", default_branch], check=True)
    else:
        print("Default branch could not be determined.")


def repo_to_file_features(repo_path: Path) -> dict[str, dict]:
    featurized_commits = {}
    checkout_default_branch(repo_path)
    last_commit = None
    for commit in Repository(str(repo_path)).traverse_commits():
        print(f"Processing commit: {commit.hash}")
        processed_commit = process_commit(commit, repo_path, last_commit)
        featurized_commits[commit.hash] = processed_commit
        last_commit = processed_commit
        print(f"Completed processing for commit: {commit.hash}\n")
    return featurized_commits


if __name__ == "__main__":
    import json

    with open("t.json", "w") as f:
        json.dump(
            repo_to_file_features(repo_path), f, default=lambda self: self.hex_digest()
        )
