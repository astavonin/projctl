"""Search handler for issues, epics, and milestones."""

import json
import logging
import subprocess
import urllib.parse
from typing import Any, Dict, List

from ..config import Config
from ..exceptions import PlatformError


logger = logging.getLogger(__name__)


class SearchHandler:
    """Handles search operations for GitLab issues and epics."""

    def __init__(self, config: Config) -> None:
        """Initialize the search handler.

        Args:
            config: Configuration object with defaults.
        """
        self.config = config

    def _run_glab_command(self, cmd: List[str]) -> str:
        """Run a glab command and return its output.

        Args:
            cmd: List of command arguments to pass to glab.

        Returns:
            Command output as a string.

        Raises:
            PlatformError: If the command fails.
        """
        full_cmd = ['glab'] + cmd

        try:
            logger.debug("Executing: %s", ' '.join(full_cmd))
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as err:
            error_msg = f"Command failed: {' '.join(full_cmd)}\n{err.stderr}"
            logger.error(error_msg)
            raise PlatformError(error_msg) from err
        except FileNotFoundError as err:
            error_msg = "glab command not found. Please install glab CLI."
            logger.error(error_msg)
            raise PlatformError(error_msg) from err

    def search_issues(
        self,
        query: str,
        state: str = 'all',
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search for issues matching a query.

        Args:
            query: Search text for title and description.
            state: Filter by state ('opened', 'closed', 'all').
            limit: Maximum number of results to return.

        Returns:
            List of issue dictionaries.

        Raises:
            PlatformError: If search fails.
        """
        # Build API endpoint for current project
        api_endpoint = f"projects/:fullpath/issues?search={urllib.parse.quote(query)}"

        if state != 'all':
            api_endpoint += f"&state={state}"

        api_endpoint += f"&per_page={limit}"

        output = self._run_glab_command(['api', api_endpoint])

        if not output:
            return []

        return json.loads(output)  # type: ignore[no-any-return]

    def search_epics(
        self,
        query: str,
        state: str = 'all',
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search for epics matching a query.

        Args:
            query: Search text for title and description.
            state: Filter by state ('opened', 'closed', 'all').
            limit: Maximum number of results to return.

        Returns:
            List of epic dictionaries.

        Raises:
            PlatformError: If search fails.
            ValueError: If group is not specified.
        """
        group_path = self.config.get_default_group()
        if not group_path:
            raise ValueError(
                "Group path is required for epic search.\n"
                "Please set 'default_group' in your glab_config.yaml file."
            )

        encoded_group = urllib.parse.quote(group_path, safe='')
        api_endpoint = f"groups/{encoded_group}/epics?search={urllib.parse.quote(query)}"

        if state != 'all':
            api_endpoint += f"&state={state}"

        api_endpoint += f"&per_page={limit}"

        output = self._run_glab_command(['api', api_endpoint])

        if not output:
            return []

        return json.loads(output)  # type: ignore[no-any-return]

    def search_milestones(
        self,
        query: str,
        state: str = 'all',
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search for milestones matching a query.

        Args:
            query: Search text for title.
            state: Filter by state ('active', 'closed', 'all').
            limit: Maximum number of results to return.

        Returns:
            List of milestone dictionaries.

        Raises:
            PlatformError: If search fails.
        """
        # Use group API if default_group is configured, otherwise use project API
        group_path = self.config.get_default_group()

        if group_path:
            # Use group milestones API
            encoded_group = urllib.parse.quote(group_path, safe='')
            api_endpoint = f"groups/{encoded_group}/milestones?search={urllib.parse.quote(query)}"
        else:
            # Use project milestones API
            api_endpoint = f"projects/:fullpath/milestones?search={urllib.parse.quote(query)}"

        if state != 'all':
            api_endpoint += f"&state={state}"

        api_endpoint += f"&per_page={limit}"

        output = self._run_glab_command(['api', api_endpoint])

        if not output:
            return []

        return json.loads(output)  # type: ignore[no-any-return]

    def print_issues(
        self,
        issues: List[Dict[str, Any]],
        query: str
    ) -> None:
        """Print search results for issues in text format.

        Args:
            issues: List of issue dictionaries.
            query: The search query used.
        """
        print(f"\n=== ISSUES matching \"{query}\" ===\n")

        if not issues:
            print("No issues found")
            return

        for issue in issues:
            iid = issue.get('iid')
            title = issue.get('title', 'Untitled')
            state = issue.get('state', 'unknown')
            labels = issue.get('labels', [])
            url = issue.get('web_url', '')

            print(f"#{iid} {title}")
            label_str = ', '.join(labels) if labels else 'none'
            print(f"    State: {state} | Labels: {label_str}")
            print(f"    URL: {url}\n")

        print(f"Found {len(issues)} issue{'s' if len(issues) != 1 else ''}")

    def print_epics(
        self,
        epics: List[Dict[str, Any]],
        query: str
    ) -> None:
        """Print search results for epics in text format.

        Args:
            epics: List of epic dictionaries.
            query: The search query used.
        """
        print(f"\n=== EPICS matching \"{query}\" ===\n")

        if not epics:
            print("No epics found")
            return

        for epic in epics:
            iid = epic.get('iid')
            title = epic.get('title', 'Untitled')
            state = epic.get('state', 'unknown')
            labels = epic.get('labels', [])
            url = epic.get('web_url', '')

            print(f"&{iid} {title}")
            label_str = ', '.join(labels) if labels else 'none'
            print(f"    State: {state} | Labels: {label_str}")
            print(f"    URL: {url}\n")

        print(f"Found {len(epics)} epic{'s' if len(epics) != 1 else ''}")

    def print_milestones(
        self,
        milestones: List[Dict[str, Any]],
        query: str
    ) -> None:
        """Print search results for milestones in text format.

        Args:
            milestones: List of milestone dictionaries.
            query: The search query used.
        """
        print(f"\n=== MILESTONES matching \"{query}\" ===\n")

        if not milestones:
            print("No milestones found")
            return

        for milestone in milestones:
            iid = milestone.get('iid')
            title = milestone.get('title', 'Untitled')
            state = milestone.get('state', 'unknown')
            url = milestone.get('web_url', '')
            due_date = milestone.get('due_date', 'N/A')

            print(f"%{iid} {title}")
            print(f"    State: {state} | Due: {due_date}")
            print(f"    URL: {url}\n")

        print(f"Found {len(milestones)} milestone{'s' if len(milestones) != 1 else ''}")
