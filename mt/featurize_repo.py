import ast
import json
import pickle
import subprocess
import traceback
from collections import defaultdict
from functools import lru_cache
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterator

import chardet
import networkx as nx
import regex as re
import six
import torch
import tree_sitter
import uuid6
from function_pipes import pipe
from nltk.tokenize import word_tokenize
from py2cfg import CFGBuilder
from pydriller import Commit, Repository
from python_graphs import program_graph, program_graph_graphviz
from radon.complexity import average_complexity, cc_visit
from tree_sitter_languages import get_parser

from mt.definitions import DATA_DIR

repo_path = DATA_DIR / "test" / "python"
parser = get_parser("python")


# TODO: future directions use LLM tokens
# from transformers import AutoModelForCausalLM, AutoTokenizer
# # Load the model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained(
#     "microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True
# )
# model = AutoModelForCausalLM.from_pretrained(
#     "microsoft/Phi-3-mini-128k-instruct",
#     output_hidden_states=False,
#     output_attentions=False,
#     trust_remote_code=True,
# )

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)


# @lru_cache(100_000)
# def get_embeddings(texts):
#     encoded_input = tokenizer(
#         texts, return_tensors="pt", padding=True, truncation=True, max_length=512
#     )

#     # Move encoded input to GPU if CUDA is available
#     if torch.cuda.is_available():
#         encoded_input = {
#             key: tensor.to("cuda") for key, tensor in encoded_input.items()
#         }

#     with torch.no_grad():
#         outputs = model(**encoded_input, output_hidden_states=True)
#     # Use the mean of the output token embeddings as the feature vector
#     embeddings = outputs.hidden_states[-1].mean(dim=1)[0]
#     return embeddings


def read_file_contents(file_path: Path) -> bytes:
    """Read the content of a file"""
    try:
        return file_path.read_bytes()
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return b""


def decode_file_contents(file_bytes: bytes) -> str:
    try:
        # Try decoding with UTF-8
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # If UTF-8 fails, use chardet to detect encoding
        encoding_guess = chardet.detect(file_bytes)["encoding"]
        if encoding_guess:
            try:
                return file_bytes.decode(encoding_guess)
            except UnicodeDecodeError:
                # If decoding with the guessed encoding fails, fall back to ISO-8859-1
                return file_bytes.decode("iso-8859-1", errors="replace")
        else:
            # If no encoding could be guessed, fallback to ISO-8859-1 with replacement
            return file_bytes.decode("iso-8859-1", errors="replace")


def camel_to_snake(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


def class_to_type(class_str: str) -> str:
    return re.search(r"ast\.(\w+)", class_str).group(1)


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
# GRAPHS
def source_to_cfg(code: str) -> nx.DiGraph:
    cfg_graph = nx.DiGraph()
    node_to_id = defaultdict(uuid6.uuid7)

    try:
        graph = program_graph.get_program_graph(code)
    except Exception as e:
        print(f"Error while generating CFG: {e}")
        traceback.print_exc()
        return cfg_graph

    for unused_key, node in graph.nodes.items():
        node_id = node_to_id[node.id].hex
        if node.ast_type:
            cfg_graph.add_node(
                node_id,
                type=camel_to_snake(six.ensure_str(node.ast_type, "utf-8")),
                start_line=getattr(node.ast_node, "lineno", -1),
                end_line=getattr(node.ast_node, "end_lineno", -1),
                start_col=getattr(node.ast_node, "col_offset", -1),
                end_col=getattr(node.ast_node, "end_col_offset", -1),
                node_type="cfg",
            )
        else:
            cfg_graph.add_node(
                node_id,
                type="cfg_point",
                start_line=-1,
                end_line=-1,
                start_col=-1,
                end_col=-1,
                node_type="cfg",
            )

    for edge in graph.edges:
        cfg_graph.add_edge(
            node_to_id[edge.id1].hex,
            node_to_id[edge.id2].hex,
            type=camel_to_snake(edge.type.name),
            edge_type="cfg",
        )

    return cfg_graph


# Filtering for anon nodes is necessary to avoid a lot of noise and non important features?
def resolve_local_import(import_path: str, file_path: Path) -> str:
    parts = file_path.parts
    rest = import_path
    while rest.startswith("."):
        parts = parts[:-1]
        _, rest = rest.split(".", maxsplit=1)
    while parts[0] != "repo" or not parts:
        parts = parts[1:]
    return ".".join(parts) + f".{rest}"


def tree_to_graph(
    root: tree_sitter.Node,
    file_path: Path,
    with_anon: bool = True,
    with_extra: bool = True,
) -> nx.DiGraph:
    G = nx.DiGraph()
    todo = [(root, None)]  # Start with the root node and no parent
    symbol_nodes = {}  # Store symbol node IDs by identifier content
    last_usage = {}  # Track last usage of each identifier for MayNextUse edges

    while todo:
        node, parent_id = todo.pop()
        node_id = uuid6.uuid7().hex  # Unique ID for each node

        if with_anon or node.is_named:
            start_line, start_col = node.start_point
            end_line, end_col = node.end_point

            # Add node to graph
            G.add_node(
                node_id,
                type=node.type,
                start_line=start_line + 1,
                end_line=end_line + 1,
                start_col=start_col,
                end_col=end_col,
                node_type="ast",
            )

            # Connect to parent node with a 'child' edge
            if parent_id:
                G.add_edge(parent_id, node_id, type="child", edge_type="ast")

            if with_extra:
                # Create or connect symbol nodes for identifiers
                if node.type == "identifier":
                    identifier = str(node.text)
                    symbol_node_id = symbol_nodes.get(identifier, uuid6.uuid7().hex)

                    # Create a central symbol node if it doesn't exist
                    if identifier not in symbol_nodes:
                        G.add_node(
                            symbol_node_id,
                            type="symbol",
                            content=identifier,
                            node_type="ch22",
                        )
                        symbol_nodes[identifier] = symbol_node_id

                    # Connect this occurrence to the symbol node
                    G.add_edge(
                        node_id, symbol_node_id, type="occurrence_of", edge_type="ch22"
                    )
                    G.add_edge(
                        symbol_node_id, node_id, type="occurrence_of", edge_type="ch22"
                    )

                    # Handling MayNextUse edges
                    if identifier in last_usage:
                        G.add_edge(
                            node_id,
                            last_usage[identifier],
                            type="may_next_use",
                            edge_type="ch22",
                        )
                    last_usage[identifier] = node_id

                # TODO: Finish this import part
                # Handle import statements
                if node.type in ["import_statement", "import_from_statement"]:
                    if node.type == "import_statement":
                        children = [n.text for n in node.children_by_field_name("name")]
                    elif node.type == "import_from_statement":
                        children = [
                            n.text for n in node.children_by_field_name("module_name")
                        ]

                    last_child = node_id

                    for child in children:
                        child = decode_file_contents(child)
                        if child.startswith("."):
                            resolve_local_import(child, file_path)
                        while True:
                            G.add_node(
                                child,
                                depth=child.count("."),
                                type="import",
                                node_type="dg",
                            )
                            G.add_edge(child, last_child, type="import", edge_type="dg")
                            last_child = child
                            splits = child.rsplit(".", maxsplit=1)
                            if len(splits) < 2:
                                break
                            child = splits[0]

        # Process children nodes
        for child in node.children:
            todo.append((child, node_id))

    return G


def find_closest_ast_node(cfg_node: nx.DiGraph, ast_graph: nx.DiGraph) -> str | None:
    cfg_start_line = cfg_node["start_line"]
    cfg_end_line = cfg_node["end_line"]
    cfg_start_col = cfg_node["start_col"]
    cfg_end_col = cfg_node["end_col"]
    if (
        cfg_start_line == -1
        or cfg_start_col == -1
        or cfg_end_col == -1
        or cfg_end_line == -1
    ):
        return None

    # Exact match
    for ast_node_id, ast_node_data in ast_graph.nodes(data=True):
        if (
            ast_node_data.get("node_type") == "ast"
            and cfg_start_line == ast_node_data.get("start_line")
            and cfg_end_line == ast_node_data.get("end_line")
            and cfg_start_col == ast_node_data.get("start_col")
            and cfg_end_col == ast_node_data.get("end_col")
        ):
            return ast_node_id

    # Hierarchical match
    for ast_node_id, ast_node_data in ast_graph.nodes(data=True):
        if (
            ast_node_data.get("node_type") == "ast"
            and cfg_start_line >= ast_node_data.get("start_line")
            and cfg_end_line <= ast_node_data.get("end_line")
            and cfg_start_col >= ast_node_data.get("start_col")
            and cfg_end_col <= ast_node_data.get("end_col")
        ):
            return ast_node_id

    # Proximity match
    closest_node = None
    closest_distance = float("inf")
    for ast_node_id, ast_node_data in ast_graph.nodes(data=True):
        if ast_node_data.get("node_type") == "ast":
            distance = (
                abs(cfg_start_line - ast_node_data.get("start_line"))
                + abs(cfg_end_line - ast_node_data.get("end_line"))
                + abs(cfg_start_col - ast_node_data.get("start_col"))
                + abs(cfg_end_col - ast_node_data.get("end_col"))
            )
            if distance < closest_distance:
                closest_distance = distance
                closest_node = ast_node_id

    return closest_node


# Function to combine CFG and AST graphs
def combine_cfg_ast(cfg_graph: nx.DiGraph, ast_graph: nx.DiGraph) -> nx.DiGraph:
    combined_graph = nx.compose(cfg_graph, ast_graph)

    for cfg_node_id, cfg_node_data in cfg_graph.nodes(data=True):
        closest_ast_node_id = find_closest_ast_node(cfg_node_data, ast_graph)
        if closest_ast_node_id:
            combined_graph.add_edge(
                cfg_node_id, closest_ast_node_id, type="matches", edge_type="cfg_ast"
            )

    return combined_graph


def get_tree_from_code(code: bytes) -> tree_sitter.Node:
    return pipe(code, parser.parse, lambda tree: tree.root_node)


# TODO: https://chat.openai.com/c/f858a84a-0811-4187-b0c5-97765b2eab11
# AST trees
def bytes_to_ast(file_content: bytes, file_path: Path) -> nx.DiGraph:
    "Reads the file content and returns the AST as a json nx graph."
    root = get_tree_from_code(file_content)
    return tree_to_graph(root, file_path)


def process_file(
    file_path: Path, file_bytes: bytes, file_text: str, graph_dir: Path
) -> dict[str, Any]:
    ast, cfg = bytes_to_ast(file_bytes, file_path), source_to_cfg(file_text)
    combined = combine_cfg_ast(cfg, ast)
    graph_path = graph_dir / f"{uuid6.uuid7().hex}.pkl"
    with open(graph_path, "wb") as f:
        pickle.dump(combined, f)

    try:
        loc = count_lines_of_code(file_text)
    except Exception as e:
        print(f"Error calculating LOC: {e}")
        loc = {
            "total": -1,
            "code": -1,
            "comments": -1,
            "blanks": -1,
        }
    
    try:
        cc = calculate_cyclomatic_complexity(file_text)
    except Exception as e:
        print(f"Error calculating CC: {e}")
        cc = -1

    features = {
        "loc": loc,
        "complexity": cc,
        "dependencies": analyze_dependencies(file_text),
        "quality": check_code_quality(file_path),
        "comment_tokens": tokenize_comments(file_text),
        "ast": str(graph_path),
    }
    return features


def process_commit(
    file_feature_dir: Path,
    graph_dir: Path,
    commit: Commit,
    repo_path: Path,
    last_commit: dict[str, Any],
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
        try:
            file_bytes = read_file_contents(file)
            file_text = decode_file_contents(file_bytes)

            content_hash = sha1(file_bytes, usedforsecurity=False).hexdigest()
            if (last_file_features := last_commit.get(str(file))) is not None:
                if last_file_features["content_hash"] == content_hash:
                    file_features[str(file)] = {
                        "content_hash": content_hash,
                        "changed": False,
                        "feature_file": last_file_features["feature_file"],
                    }
                    continue
            features = process_file(file, file_bytes, file_text, graph_dir)
            file_path = file_feature_dir / f"{uuid6.uuid7().hex}.json"
            with open(file_path, "w") as f:
                json.dump(features, f)
            file_features[str(file)] = {
                "content_hash": content_hash,
                "changed": True,
                "feature_file": file_path.as_posix(),
            }
        except Exception as e:
            print(f"Error processing file {file.name}: {e}")
            traceback.print_exc()
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


def repo_to_file_features(
    repo_root: Path, repo_path: Path
) -> Iterator[tuple[int, str, dict]]:
    file_feature_dir = repo_root / "file_data"
    file_feature_dir.mkdir(exist_ok=True)
    graph_dir = repo_root / "graphs"
    graph_dir.mkdir(exist_ok=True)
    checkout_default_branch(repo_path)
    total_commits = len(list(Repository(str(repo_path)).traverse_commits()))
    checkout_default_branch(repo_path)
    last_commit = {}
    for i, commit in enumerate(Repository(str(repo_path)).traverse_commits()):
        print(f"[{i+1}/{total_commits}] Processing commit: {commit.hash}")
        processed_commit = process_commit(
            file_feature_dir, graph_dir, commit, repo_path, last_commit
        )
        # featurized_commits[commit.hash] = processed_commit
        last_commit = processed_commit
        print(
            f"[{i+1}/{total_commits}] Completed processing for commit: {commit.hash}\n"
        )
        yield i, commit.hash, processed_commit


def test_repo():
    import json

    with open("features.json", "w") as f:
        json.dump(
            repo_to_file_features(repo_path), f, default=lambda self: self.hexdigest()
        )


if __name__ == "__main__":
    import cProfile

    cProfile.run("test_repo()", "old_method.profile")
