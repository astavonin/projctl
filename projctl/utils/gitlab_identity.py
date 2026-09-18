"""Shared resolution of the authenticated GitLab user via GraphQL.

Promoted out of `handlers/timelog.py` so `handlers/loader.py`'s `--json`
path can populate the same `viewer` identity under the same hard-error
contract, without either module re-deriving the null-check.
"""

from typing import Any, Dict

from ..exceptions import PlatformError

CURRENT_USER_QUERY = "query { currentUser { username } }"


def extract_current_username(data: Dict[str, Any]) -> str:
    """Extract and validate the authenticated username from a currentUser response.

    This is the load-bearing safety check for every caller that resolves the
    authenticated identity: `glab` resolves its target host from the current
    directory's git remote and silently falls back to a default host outside
    a GitLab repo, at exit code 0 with `currentUser: null`. Treating that
    null as an absent value would let a caller run against the wrong host
    with no sign of it, so it is a hard error instead, naming the likely
    cause.

    Args:
        data: The `data` object from a `CURRENT_USER_QUERY` GraphQL
            response, as returned by `parse_graphql_data()`.

    Returns:
        The authenticated username.

    Raises:
        PlatformError: If currentUser resolves to null, or carries no username.
    """
    username = (data.get("currentUser") or {}).get("username")
    if not username:
        raise PlatformError(
            "GitLab GraphQL currentUser returned null — glab likely resolved the "
            "wrong host (e.g. this directory has no GitLab remote, so glab fell "
            "back to a default host with no authenticated session). Run from "
            "inside a GitLab-remote repository, or check 'glab auth status'."
        )
    return str(username)
