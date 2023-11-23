from requests import get
from typing import Any, Iterator, TypeVar
from masters_thesis.credentials import GITHUB_TOKEN

T = TypeVar("T")

def api_get(url: str) -> Iterator[dict[str, Any]]:
    """Get data as a json and put"""
    response = get(
        url,
        headers={
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"Bearer {GITHUB_TOKEN}",
        },
    )
    if response.status_code == 200:
        data = response.json()
        if type(data) != dict:
            data = {"items": data}
        data["request_string"] = url
        data["metadata"] = dict(response.headers)
        yield data
        if "next" in response.links.keys():
            yield from api_get(response.links["next"]["url"])
    else:
        print(f"Error {response.status_code} when getting {url}")


def flatten(list: list[list[T]]) -> list[T]:
    """Flatten a list of lists"""
    return [item for sublist in list for item in sublist]


