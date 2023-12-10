import asyncio
import base64
import json
import logging as log
import pickle
from typing import Any

import aiofiles
import aiohttp
import networkx as nx
import tree_sitter
from function_pipes import pipe
from tree_sitter_languages import get_parser
from uuid6 import uuid7

from mt.definitions import REPO_DIR, REPO_SEARCH_DIR
from mt.helper import api_get

parser = get_parser("python")


def get_tree_from_code(code: bytes) -> nx.DiGraph:
    return pipe(code, parser.parse, lambda tree: tree.root_node, tree_to_graph)


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


# Assuming handle_git_tree can also be refactored for async
async def handle_git_tree(
    git_tree: dict[str, str], session: aiohttp.ClientSession, path: str = ""
):
    for item in git_tree["tree"]:
        if item["type"] == "tree":
            async for tree in api_get(item["url"], session):
                async for result in handle_git_tree(
                    tree, session, path + item["path"] + "/"
                ):
                    yield result
        elif item["type"] == "blob":
            async for blob in api_get(item["url"], session):
                yield blob, path + item["path"]


async def handle_commit(
    commit: dict[str, Any],
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    save_files: bool = False,
):
    async with semaphore:
        sha = commit["sha"]
        commit_dir = REPO_DIR / sha
        commit_dir.mkdir(exist_ok=True)

        async with aiofiles.open(commit_dir / "commit.json", "w") as f:
            await f.write(json.dumps(commit))

        # Initialize git_tree to None or a default value
        git_tree = None

        async for git_tree_data in api_get(commit["commit"]["tree"]["url"], session):
            git_tree = git_tree_data
            break  # Exit the loop after getting the first item

        # Check if git_tree was set
        if git_tree is not None:
            async with aiofiles.open(commit_dir / "tree.json", "w") as f:
                await f.write(json.dumps(git_tree))
        file_dir = commit_dir / "files"
        file_dir.mkdir(exist_ok=True)
        featurized_dir = commit_dir / "featurized"
        featurized_dir.mkdir(exist_ok=True)

        async for item, path in handle_git_tree(git_tree, session):
            if save_files:
                item_path = file_dir / (path + ".json")
                item_path.parent.mkdir(exist_ok=True, parents=True)
                async with aiofiles.open(item_path, "w") as f:
                    await f.write(json.dumps(item))

            if path.endswith(".py"):
                uuid = uuid7().hex
                code = base64.b64decode(item["content"])
                tree = get_tree_from_code(
                    code
                )  # Assuming this is a CPU-bound operation
                tree_path = featurized_dir / (uuid + ".pickle")
                async with aiofiles.open(tree_path, "wb") as f:
                    await f.write(pickle.dumps(tree))
                async with aiofiles.open(featurized_dir / (uuid + ".json"), "w") as f:
                    await f.write(json.dumps({"path": path, "file_size": item["size"]}))


async def process_repository(
    repo: dict[str, Any], session: aiohttp.ClientSession, semaphore: asyncio.Semaphore
) -> None:
    log.info(f"Starting processing for repository: {repo['full_name']}")

    directory = REPO_DIR / repo["full_name"]
    directory.mkdir(parents=True, exist_ok=True)

    try:
        # Dump repository data
        async with aiofiles.open(directory / "repo.json", "w") as f:
            await f.write(json.dumps(repo))
        log.info(f"Repository data saved for {repo['full_name']}")

        # Process commits
        commits_url = repo["commits_url"].replace("{/sha}", "?per_page=100")
        commits = []
        async for commit_data in api_get(commits_url, session):
            commits.extend(commit_data.get("items", []))

        commit_tasks = [
            asyncio.create_task(
                handle_commit(commit, session, semaphore, save_files=True)
            )
            for commit in commits
        ]
        await asyncio.gather(*commit_tasks)
        log.info(f"Processed {len(commits)} commits for {repo['full_name']}")

        # Process issues
        issues_url = repo["issues_url"].replace("{/number}", "?per_page=100&state=all")
        issues = []
        async for issue_data in api_get(issues_url, session):
            issues.extend(issue_data.get("items", []))

        async with aiofiles.open(directory / "issues.json", "w") as f:
            await f.write(json.dumps(issues))
        log.info(f"Issues data saved for {repo['full_name']}")

        # Process stars
        stars_url = f"{repo['stargazers_url']}?per_page=100"
        stars = []
        async for star_data in api_get(stars_url, session):
            stars.extend(star_data.get("items", []))

        async with aiofiles.open(directory / "stars.json", "w") as f:
            await f.write(json.dumps(stars))
        log.info(f"Star data saved for {repo['full_name']}")

    except Exception as e:
        log.error(f"Error processing repository {repo['full_name']}: {e}")

    log.info(f"Finished processing repository: {repo['full_name']}")


async def main():
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(5)
        repos = []
        async for repo_data in api_get(
            "https://api.github.com/search/repositories?q=language:python&sort=stars&order=desc&per_page=100&page=1",
            session,
        ):
            repos.extend(repo_data["items"])

        # Process only the first 3 repositories for demonstration
        await asyncio.gather(
            *(process_repository(repo, session, semaphore) for repo in repos[:3])
        )


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    asyncio.run(main())
