from typing import Any, AsyncIterator, TypeVar, Iterator
from requests import get

import aiohttp

from mt.credentials import GITHUB_TOKEN

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


# async def api_get(
#     url: str, session: aiohttp.ClientSession
# ) -> AsyncIterator[dict[str, Any]]:
#     async with session.get(
#         url, headers={"Authorization": f"Bearer {GITHUB_TOKEN}"}
#     ) as response:
#         if response.status == 200:
#             data = await response.json()
#             if type(data) != dict:
#                 data = {"items": data}
#             data["request_string"] = url
#             data["metadata"] = dict(response.headers)
#             yield data
#             if "next" in response.links.keys():
#                 async for item in api_get(response.links["next"]["url"], session):
#                     yield item
#         else:
#             log.exception(f"Error {response.status} when getting {url}")
