"""GitHub issue, PR, and milestone loader handler."""

import json
import logging
from typing import Any, Dict, List

from ..config import Config
from ..exceptions import PlatformError
from ..utils.gh_runner import run_gh_command

logger = logging.getLogger(__name__)

# JSON fields requested for each resource type
_ISSUE_FIELDS = "number,title,body,state,labels,assignees,milestone,url"
_PR_FIELDS = "number,title,body,state,labels,assignees,reviewRequests,url,headRefName,baseRefName"


class GithubLoader:
    """Loads and prints GitHub issues, PRs, and milestones using the gh CLI."""

    def __init__(self, config: Config) -> None:
        """Initialize the loader.

        Args:
            config: Configuration object with defaults.
        """
        self.config = config
        self.repo = config.get_github_repo()

    def load_issue(self, ref: str) -> Dict[str, Any]:
        """Load a GitHub issue by reference.

        Accepts ``#123``, ``123``, or a plain integer string.

        Args:
            ref: Issue reference (e.g. "#123" or "123").

        Returns:
            Dictionary containing issue data from the GitHub API.

        Raises:
            PlatformError: If the gh command fails.
            ValueError: If the reference format is unrecognised.
        """
        number = self._extract_number(ref, prefix="#")
        logger.debug("[github] Loading issue #%s", number)
        output = run_gh_command(["issue", "view", number, "--json", _ISSUE_FIELDS])
        data: Dict[str, Any] = json.loads(output)
        self.print_issue_info(data)
        return data

    def load_pr(self, ref: str) -> Dict[str, Any]:
        """Load a GitHub pull request by reference.

        Accepts ``!123`` or a plain integer string.

        Args:
            ref: PR reference (e.g. "!123" or "123").

        Returns:
            Dictionary containing PR data from the GitHub API.

        Raises:
            PlatformError: If the gh command fails.
            ValueError: If the reference format is unrecognised.
        """
        number = self._extract_number(ref, prefix="!")
        logger.debug("[github] Loading PR #%s", number)
        output = run_gh_command(["pr", "view", number, "--json", _PR_FIELDS])
        data: Dict[str, Any] = json.loads(output)
        self.print_pr_info(data)
        return data

    def load_milestone(self, ref: str) -> Dict[str, Any]:
        """Load a GitHub milestone and its issues by reference.

        Accepts ``%5`` or a plain integer string.

        Args:
            ref: Milestone reference (e.g. "%5" or "5").

        Returns:
            Dictionary with ``milestone`` and ``issues`` keys.

        Raises:
            PlatformError: If an API call fails.
            ValueError: If the reference or milestone is not found.
        """
        number = self._extract_number(ref, prefix="%")
        logger.debug("[github] Loading milestone #%s", number)

        ms_output = run_gh_command(["api", f"repos/{self.repo}/milestones/{number}"])
        milestone_data: Dict[str, Any] = json.loads(ms_output)

        milestone_title: str = milestone_data.get("title", "")
        issues = self._load_milestone_issues(milestone_title)

        result: Dict[str, Any] = {"milestone": milestone_data, "issues": issues}
        self.print_milestone_info(result)
        return result

    def _load_milestone_issues(self, milestone_title: str) -> List[Dict[str, Any]]:
        """Fetch issues belonging to a milestone.

        Args:
            milestone_title: Milestone title used as the filter.

        Returns:
            List of issue dicts. Empty list on failure.
        """
        try:
            output = run_gh_command(
                [
                    "issue",
                    "list",
                    "--milestone",
                    milestone_title,
                    "--json",
                    "number,title,state,labels,url",
                    "--limit",
                    "200",
                ]
            )
            issues: List[Dict[str, Any]] = json.loads(output) if output else []
            return issues
        except PlatformError as err:
            logger.warning("[github] Failed to load milestone issues: %s", err)
            return []

    @staticmethod
    def _extract_number(ref: str, prefix: str) -> str:
        """Strip a known prefix and return the numeric portion.

        Args:
            ref: Reference string such as "#123", "!5", "%10", or "42".
            prefix: Expected prefix character to strip.

        Returns:
            Numeric string.

        Raises:
            ValueError: If the remaining string is not a positive integer.
        """
        if ref.startswith(prefix):
            ref = ref[len(prefix):]
        if not ref.isdigit():
            raise ValueError(
                f"Invalid reference {ref!r}: expected a positive integer after '{prefix}'"
            )
        return ref

    def print_issue_info(self, data: Dict[str, Any]) -> None:
        """Print a GitHub issue in a readable format.

        Args:
            data: Issue data dict from ``load_issue``.
        """
        number = data.get("number", "?")
        title = data.get("title", "Untitled")
        state = data.get("state", "unknown")
        url = data.get("url", "")
        assignees = [a.get("login", "") for a in data.get("assignees", [])]
        labels = [lb.get("name", "") for lb in data.get("labels", [])]
        milestone = (data.get("milestone") or {}).get("title", "")
        body = data.get("body", "") or ""

        print(f"\n## Issue #{number}: {title}\n")
        print(f"**State:** {state}")
        if assignees:
            print(f"**Assignees:** {', '.join(assignees)}")
        if labels:
            print(f"**Labels:** {', '.join(labels)}")
        if milestone:
            print(f"**Milestone:** {milestone}")
        print(f"**URL:** {url}\n")
        if body:
            print(body)

    def print_pr_info(self, data: Dict[str, Any]) -> None:
        """Print a GitHub pull request in a readable format.

        Args:
            data: PR data dict from ``load_pr``.
        """
        number = data.get("number", "?")
        title = data.get("title", "Untitled")
        state = data.get("state", "unknown")
        url = data.get("url", "")
        head = data.get("headRefName", "")
        base = data.get("baseRefName", "")
        assignees = [a.get("login", "") for a in data.get("assignees", [])]
        reviewers = [r.get("login", "") for r in data.get("reviewRequests", [])]
        labels = [lb.get("name", "") for lb in data.get("labels", [])]
        body = data.get("body", "") or ""

        print(f"\n## PR #{number}: {title}\n")
        print(f"**State:** {state}")
        print(f"**Branch:** {head} -> {base}")
        if assignees:
            print(f"**Assignees:** {', '.join(assignees)}")
        if reviewers:
            print(f"**Reviewers:** {', '.join(reviewers)}")
        if labels:
            print(f"**Labels:** {', '.join(labels)}")
        print(f"**URL:** {url}\n")
        if body:
            print(body)

    def print_milestone_info(self, data: Dict[str, Any]) -> None:
        """Print a GitHub milestone and its issues in a readable format.

        Args:
            data: Dict with ``milestone`` and ``issues`` keys from ``load_milestone``.
        """
        ms = data.get("milestone", {})
        issues: List[Dict[str, Any]] = data.get("issues", [])

        number = ms.get("number", "?")
        title = ms.get("title", "Untitled")
        state = ms.get("state", "unknown")
        description = ms.get("description", "") or ""
        due_on = ms.get("due_on", "")
        url = ms.get("html_url", "")

        open_count = ms.get("open_issues", 0)
        closed_count = ms.get("closed_issues", 0)

        print(f"\n## Milestone #{number}: {title}\n")
        print(f"**State:** {state}")
        if due_on:
            print(f"**Due:** {due_on}")
        print(f"**Issues:** {open_count} open, {closed_count} closed")
        print(f"**URL:** {url}\n")
        if description:
            print(description)
            print()

        if issues:
            print(f"### Issues ({len(issues)})\n")
            for issue in issues:
                iid = issue.get("number", "?")
                ititle = issue.get("title", "Untitled")
                istate = issue.get("state", "unknown")
                iurl = issue.get("url", "")
                print(f"- #{iid} [{istate}] {ititle}")
                print(f"  {iurl}")
