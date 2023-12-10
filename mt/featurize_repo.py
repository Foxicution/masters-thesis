from pydriller import Repository
from pathlib import Path
import subprocess
from mt.definitions import DATA_DIR
from radon.complexity import average_complexity, cc_visit
from typing import Any, Iterator
import ast
import subprocess
from nltk.tokenize import word_tokenize
from git import Repo


repo_path = DATA_DIR / "test" / "python"


def read_file_contents(file_path: Path) -> str:
    """Read the content of a file"""
    try:
        return file_path.read_text(encoding="utf-8")
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


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


def calculate_cyclomatic_complexity(file_content):
    blocks = cc_visit(file_content)
    if blocks:
        return average_complexity(blocks)
    else:
        return 0  # Return 0 or an appropriate value for files with no blocks


def check_code_quality(file_path):
    result = subprocess.run(["pylint", file_path], capture_output=True, text=True)
    return result.stdout


def analyze_dependencies(file_content):
    tree = ast.parse(file_content)
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom)
    ]
    return [node.names[0].name for node in imports]


def tokenize_comments(file_content):
    comments = [
        line for line in file_content.split("\n") if line.strip().startswith("#")
    ]
    tokens = [word_tokenize(comment) for comment in comments]
    return tokens


def process_file(file_path: Path) -> dict[str, Any]:
    content = read_file_contents(file_path)
    loc = count_lines_of_code(content)
    complexity = calculate_cyclomatic_complexity(content)
    dependencies = analyze_dependencies(content)
    quality = check_code_quality(file_path)
    comment_tokens = tokenize_comments(content)

    # Combine all features into a dictionary or a similar structure
    features = {
        "loc": loc,
        "complexity": complexity,
        "dependencies": dependencies,
        "quality": quality,
        "comment_tokens": comment_tokens,
    }
    return features


def process_commit(commit, repo_path):
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

    file_features = {}
    for file in commit.modified_files:
        if file.filename.endswith(".py"):
            file_path = repo_path / file.filename
            if file_path.exists():
                try:
                    file_features[str(file_path)] = process_file(file_path)
                except Exception as e:
                    print(f"Error processing file {file.filename}: {e}")
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


def checkout_default_branch(repo_path: str):
    default_branch = get_default_branch(repo_path)
    if default_branch:
        subprocess.run(["git", "-C", repo_path, "checkout", default_branch], check=True)
    else:
        print("Default branch could not be determined.")


def repo_to_file_features(repo_path: Path) -> dict[str, dict]:
    featurized_commits = {}
    checkout_default_branch(repo_path)
    for commit in Repository(str(repo_path)).traverse_commits():
        # for commit in Repo(repo_path).iter_commits():
        # print(commit.hexsha)
        print(f"Processing commit: {commit.hash}")
        featurized_commits[commit.hash] = process_commit(commit, repo_path)
        print(f"Completed processing for commit: {commit.hash}\n")
    return featurized_commits


if __name__ == "__main__":
    ...
    # traverse_commits(repo_dir)
