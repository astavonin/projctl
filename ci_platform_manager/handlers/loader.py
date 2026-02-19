"""Ticket (issue/epic/milestone/MR) loader handler."""

import json
import logging
import subprocess
import urllib.parse
from typing import Any, Dict, List, Optional

from ..config import Config
from ..exceptions import PlatformError


logger = logging.getLogger(__name__)


class TicketLoader:
    """Loads GitLab issue and epic information using the glab CLI."""

    def __init__(self, config: Config) -> None:
        """Initialize the loader.

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

    def _parse_issue_reference(self, issue_ref: str) -> tuple:
        """Parse issue reference to extract project path and iid.

        Args:
            issue_ref: Issue reference (number, URL, or #number format).

        Returns:
            Tuple of (project_path, iid). project_path may be None if not in URL.
        """
        # URL format: https://gitlab.../group/project/-/issues/123
        if '/-/issues/' in issue_ref:
            parts = issue_ref.split('/-/issues/')
            if len(parts) == 2:
                project_url = parts[0]
                iid = parts[1].split('/')[0].split('?')[0]

                # Extract project path from URL
                if '//' in project_url:
                    project_path = '/'.join(project_url.split('//')[1].split('/')[1:])
                else:
                    project_path = project_url

                return (project_path, iid)

        # #123 format
        if issue_ref.startswith('#'):
            return (None, issue_ref[1:])

        # Plain number
        if issue_ref.isdigit():
            return (None, issue_ref)

        raise ValueError(f"Cannot parse issue reference: {issue_ref}")

    def _parse_epic_reference(self, epic_ref: str) -> tuple:
        """Parse epic reference to extract group path and iid.

        Args:
            epic_ref: Epic reference (number, URL, or &number format).

        Returns:
            Tuple of (group_path, iid). group_path may be None if not in URL.

        Raises:
            ValueError: If epic reference cannot be parsed.
        """
        # URL format: https://gitlab.../groups/mygroup/-/epics/21
        if '/-/epics/' in epic_ref:
            parts = epic_ref.split('/-/epics/')
            if len(parts) == 2:
                group_url = parts[0]
                iid = parts[1].split('/')[0].split('?')[0]

                # Extract group path from URL
                # Format: https://gitlab.example.com/groups/mygroup
                if '/groups/' in group_url:
                    group_path = group_url.split('/groups/')[-1]
                elif '//' in group_url:
                    # Fallback: take everything after the domain
                    group_path = '/'.join(group_url.split('//')[1].split('/')[1:])
                else:
                    group_path = group_url

                return (group_path, iid)

        # &21 format (GitLab epic reference)
        if epic_ref.startswith('&'):
            return (None, epic_ref[1:])

        # Plain number
        if epic_ref.isdigit():
            return (None, epic_ref)

        raise ValueError(f"Cannot parse epic reference: {epic_ref}")

    def _parse_milestone_reference(self, milestone_ref: str) -> tuple:
        """Parse milestone reference to extract project/group path, iid, and milestone type.

        Args:
            milestone_ref: Milestone reference (number, URL, or %number format).

        Returns:
            Tuple of (project_or_group_path, iid, is_group_milestone).
            project_or_group_path may be None if not in URL.

        Raises:
            ValueError: If milestone reference cannot be parsed.
        """
        # URL format for group milestone: https://gitlab.../groups/mygroup/-/milestones/123
        if '/groups/' in milestone_ref and '/-/milestones/' in milestone_ref:
            parts = milestone_ref.split('/-/milestones/')
            if len(parts) == 2:
                group_url = parts[0]
                iid = parts[1].split('/')[0].split('?')[0]

                # Extract group path from URL
                # Format: https://gitlab.example.com/groups/mygroup/subgroup
                if '/groups/' in group_url:
                    group_path = group_url.split('/groups/')[-1]
                elif '//' in group_url:
                    # Fallback: take everything after the domain
                    group_path = '/'.join(group_url.split('//')[1].split('/')[1:])
                else:
                    group_path = group_url

                return (group_path, iid, True)

        # URL format for project milestone: https://gitlab.../group/project/-/milestones/123
        if '/-/milestones/' in milestone_ref:
            parts = milestone_ref.split('/-/milestones/')
            if len(parts) == 2:
                project_url = parts[0]
                iid = parts[1].split('/')[0].split('?')[0]

                # Extract project path from URL
                if '//' in project_url:
                    project_path = '/'.join(project_url.split('//')[1].split('/')[1:])
                else:
                    project_path = project_url

                return (project_path, iid, False)

        # %123 format (GitLab milestone reference)
        if milestone_ref.startswith('%'):
            return (None, milestone_ref[1:], None)

        # Plain number
        if milestone_ref.isdigit():
            return (None, milestone_ref, None)

        raise ValueError(f"Cannot parse milestone reference: {milestone_ref}")

    def load_issue(self, issue_ref: str) -> Dict[str, Any]:
        """Load issue information from GitLab.

        Args:
            issue_ref: Issue reference (number, URL, or #number format).

        Returns:
            Dictionary containing issue data.

        Raises:
            PlatformError: If loading fails.
        """
        project_path, iid = self._parse_issue_reference(issue_ref)

        if project_path:
            encoded_project = urllib.parse.quote(project_path, safe='')
        else:
            # Use current repo via glab's :fullpath shorthand
            encoded_project = ":fullpath"

        api_endpoint = f"projects/{encoded_project}/issues/{iid}"

        output = self._run_glab_command(['api', api_endpoint])
        return json.loads(output)  # type: ignore[no-any-return]

    def load_epic(self, group_path: str, epic_iid: int) -> Dict[str, Any]:
        """Load epic information from GitLab.

        Args:
            group_path: GitLab group path.
            epic_iid: Epic iid within the group.

        Returns:
            Dictionary containing epic data.

        Raises:
            PlatformError: If loading fails.
        """
        encoded_group = urllib.parse.quote(group_path, safe='')
        api_endpoint = f"groups/{encoded_group}/epics/{epic_iid}"

        output = self._run_glab_command(['api', api_endpoint])
        return json.loads(output)  # type: ignore[no-any-return]

    def _get_group_milestone_id(self, group_path: str, milestone_iid: str) -> Optional[str]:
        """Convert group milestone iid to id.

        Group milestone API requires id, not iid.
        List all milestones and find the one matching the iid.

        Args:
            group_path: GitLab group path.
            milestone_iid: Milestone iid within the group.

        Returns:
            Milestone id as string, or None if not found.

        Raises:
            PlatformError: If API call fails.
        """
        encoded_group = urllib.parse.quote(group_path, safe='')
        api_endpoint = f"groups/{encoded_group}/milestones?per_page=100"

        try:
            output = self._run_glab_command(['api', api_endpoint])
            milestones = json.loads(output) if output else []

            for ms in milestones:
                if str(ms.get('iid')) == str(milestone_iid):
                    return str(ms.get('id'))

            return None
        except (PlatformError, json.JSONDecodeError) as err:
            logger.warning("Failed to resolve milestone iid to id: %s", err)
            return None

    def load_epic_issues(self, group_path: str, epic_iid: int) -> List[Dict[str, Any]]:
        """Load all issues associated with an epic.

        Args:
            group_path: GitLab group path.
            epic_iid: Epic iid within the group.

        Returns:
            List of issue dictionaries.

        Raises:
            PlatformError: If loading fails.
        """
        encoded_group = urllib.parse.quote(group_path, safe='')
        api_endpoint = f"groups/{encoded_group}/epics/{epic_iid}/issues?per_page=100"

        try:
            output = self._run_glab_command(['api', api_endpoint])
            if not output:
                return []
            return json.loads(output)  # type: ignore[no-any-return]
        except PlatformError as err:
            logger.warning("Failed to load epic issues: %s", err)
            return []

    def load_epic_with_issues(self, epic_ref: str) -> Dict[str, Any]:
        """Load epic and all its associated issues.

        Args:
            epic_ref: Epic reference (number, URL, or &number format).

        Returns:
            Dictionary containing epic and issues data with structure:
            {
                'epic': {epic data},
                'issues': [list of issues]
            }

        Raises:
            PlatformError: If loading fails.
            ValueError: If group path cannot be determined.
        """
        # Parse epic reference
        parsed_group, epic_iid = self._parse_epic_reference(epic_ref)

        # Determine group path
        final_group_path = parsed_group or self.config.get_default_group()

        if not final_group_path:
            raise ValueError(
                "Group path is required to load epic.\n"
                "Either include the group in the URL or set 'default_group' in your glab_config.yaml file."
            )

        # Load epic data
        epic_data = self.load_epic(final_group_path, int(epic_iid))

        # Load epic issues
        issues = self.load_epic_issues(final_group_path, int(epic_iid))

        return {
            'epic': epic_data,
            'issues': issues
        }

    def load_milestone_with_issues(self, milestone_ref: str) -> Dict[str, Any]:
        """Load milestone and all its associated issues.

        Args:
            milestone_ref: Milestone reference (number, URL, or %number format).

        Returns:
            Dictionary containing milestone, issues, and epic mapping with structure:
            {
                'milestone': {milestone data},
                'issues': [list of issues],
                'epic_map': {epic_iid: epic_data}
            }

        Raises:
            PlatformError: If loading fails.
        """
        # Parse milestone reference
        parsed_path, milestone_iid, is_group_milestone = self._parse_milestone_reference(milestone_ref)

        # Determine if this is a group or project milestone
        if is_group_milestone is None:
            # Not determined from URL, use config default
            is_group_milestone = bool(self.config.get_default_group())
            if is_group_milestone:
                parsed_path = self.config.get_default_group()

        if is_group_milestone:
            # Use group milestone API
            if parsed_path:
                encoded_path = urllib.parse.quote(parsed_path, safe='')
            else:
                # Should not happen if is_group_milestone is True
                raise ValueError(
                    "Group path is required for group milestone.\n"
                    "Either include the group in the URL or set 'default_group' in your glab_config.yaml file."
                )

            # Convert iid to id for group milestones (API requires id, not iid)
            milestone_id = self._get_group_milestone_id(parsed_path, milestone_iid)
            if not milestone_id:
                raise PlatformError(f"Milestone iid {milestone_iid} not found in group {parsed_path}")

            api_endpoint = f"groups/{encoded_path}/milestones/{milestone_id}"
            issues_endpoint = f"groups/{encoded_path}/milestones/{milestone_id}/issues?per_page=100"
        else:
            # Use project milestone API
            if parsed_path:
                encoded_path = urllib.parse.quote(parsed_path, safe='')
            else:
                # Use current repo via glab's :fullpath shorthand
                encoded_path = ":fullpath"
            api_endpoint = f"projects/{encoded_path}/milestones/{milestone_iid}"
            issues_endpoint = f"projects/{encoded_path}/milestones/{milestone_iid}/issues?per_page=100"

        # Load milestone data
        output = self._run_glab_command(['api', api_endpoint])
        milestone_data = json.loads(output)

        # Load milestone issues
        try:
            issues_output = self._run_glab_command(['api', issues_endpoint])
            issues = json.loads(issues_output) if issues_output else []
        except PlatformError as err:
            logger.warning("Failed to load milestone issues: %s", err)
            issues = []

        # Load epic information for each issue
        epic_map = {}
        for issue in issues:
            epic_iid = issue.get('epic_iid')
            if epic_iid and epic_iid not in epic_map:
                # Extract group path from issue's project
                project_path = issue.get('references', {}).get('full', '').split('#')[0]
                if project_path:
                    group_path = '/'.join(project_path.split('/')[:-1])
                    if group_path:
                        try:
                            epic_data = self.load_epic(group_path, epic_iid)
                            epic_map[epic_iid] = epic_data
                        except PlatformError as err:
                            logger.warning("Failed to load epic %s: %s", epic_iid, err)
        return {
            'milestone': milestone_data,
            'issues': issues,
            'epic_map': epic_map
        }

    def load_issue_links(self, project_path: str, issue_iid: str) -> Dict[str, List[Dict]]:
        """Load issue dependency links (blocking and blocked relationships).

        Args:
            project_path: GitLab project path.
            issue_iid: Issue iid within the project.

        Returns:
            Dictionary with 'blocking' and 'blocked' lists.

        Raises:
            PlatformError: If loading fails.
        """
        encoded_project = urllib.parse.quote(project_path, safe='')
        api_endpoint = f"projects/{encoded_project}/issues/{issue_iid}/links"

        try:
            output = self._run_glab_command(['api', api_endpoint])
            if not output:
                return {'blocking': [], 'blocked': []}

            links = json.loads(output)

            # Separate into blocking (this issue blocks others) and blocked (this issue is blocked by others)
            blocking = []
            blocked_by = []

            for link in links:
                link_type = link.get('link_type')
                if link_type == 'blocks':
                    # This issue blocks the linked issue
                    blocking.append(link)
                elif link_type == 'is_blocked_by':
                    # This issue is blocked by the linked issue
                    blocked_by.append(link)

            return {'blocking': blocking, 'blocked_by': blocked_by}
        except PlatformError as err:
            logger.warning("Failed to load issue links: %s", err)
            return {'blocking': [], 'blocked_by': []}

    def load_ticket_with_epic(self, issue_ref: str) -> Dict[str, Any]:
        """Load issue and its related epic information.

        Args:
            issue_ref: Issue reference (number, URL, or #number format).

        Returns:
            Dictionary containing issue and epic data.

        Raises:
            PlatformError: If loading fails.
        """
        issue_data = self.load_issue(issue_ref)

        result: Dict[str, Any] = {
            'issue': issue_data,
            'epic': None,
            'links': {'blocking': [], 'blocked_by': []}
        }

        # Check if issue has an associated epic
        epic_iid = issue_data.get('epic_iid')
        if epic_iid:
            # Extract group path from issue's project
            # The epic belongs to the parent group of the project
            project_path = issue_data.get('references', {}).get('full', '')
            if project_path:
                # Remove issue reference part and project name to get group
                # Format: group/subgroup/project#123
                project_full = project_path.split('#')[0]
                group_path = '/'.join(project_full.split('/')[:-1])

                if group_path:
                    try:
                        epic_data = self.load_epic(group_path, epic_iid)
                        result['epic'] = epic_data
                    except PlatformError as err:
                        logger.warning("Failed to load epic %s: %s", epic_iid, err)
        # Load issue dependency links
        project_path = issue_data.get('references', {}).get('full', '').split('#')[0]
        if project_path:
            issue_iid = issue_data.get('iid')
            result['links'] = self.load_issue_links(project_path, str(issue_iid))

        return result

    def print_ticket_info(self, data: Dict[str, Any]) -> None:
        """Print ticket information in markdown format.

        Args:
            data: Dictionary containing issue and epic data.
        """
        issue = data['issue']
        epic = data.get('epic')
        links = data.get('links')
        self._print_markdown(issue, epic, links)

    def _print_markdown(
        self,
        issue: Dict[str, Any],
        epic: Optional[Dict[str, Any]],
        links: Optional[Dict[str, List[Dict]]] = None
    ) -> None:
        """Print ticket info in markdown format."""
        print(f"# Issue #{issue.get('iid')}: {issue.get('title')}\n")

        print(f"**URL:** {issue.get('web_url')}  ")
        print(f"**State:** {issue.get('state')}  ")
        print(f"**Author:** {issue.get('author', {}).get('name', 'Unknown')}  ")

        labels = issue.get('labels', [])
        if labels:
            print(f"**Labels:** {', '.join([f'`{label}`' for label in labels])}  ")

        assignees = issue.get('assignees', [])
        if assignees:
            names = [a.get('name', a.get('username')) for a in assignees]
            print(f"**Assignees:** {', '.join(names)}  ")

        if issue.get('milestone'):
            print(f"**Milestone:** {issue['milestone'].get('title')}  ")

        if issue.get('due_date'):
            print(f"**Due Date:** {issue['due_date']}  ")

        print(f"\n**Created:** {issue.get('created_at')}  ")
        print(f"**Updated:** {issue.get('updated_at')}  ")

        # Print dependencies
        if links:
            blocked_by = links.get('blocked_by', [])
            blocking = links.get('blocking', [])

            if blocked_by:
                print("\n### ⛔ Blocked By\n")
                for link in blocked_by:
                    link_iid = link.get('iid')
                    link_title = link.get('title', 'Untitled')
                    link_state = link.get('state', 'unknown')
                    link_url = link.get('web_url', '')
                    print(f"- [#{link_iid} {link_title}]({link_url}) `[{link_state}]`")

            if blocking:
                print("\n### 🚧 Blocking\n")
                for link in blocking:
                    link_iid = link.get('iid')
                    link_title = link.get('title', 'Untitled')
                    link_state = link.get('state', 'unknown')
                    link_url = link.get('web_url', '')
                    print(f"- [#{link_iid} {link_title}]({link_url}) `[{link_state}]`")

        if epic:
            print(f"\n## Epic &{epic.get('iid')}: {epic.get('title')}\n")
            print(f"**URL:** {epic.get('web_url')}  ")
            print(f"**State:** {epic.get('state')}  ")

        print("\n## Description\n")
        description = issue.get('description', 'No description')
        print(description if description else '*No description*')
        print()

    def print_epic_info(self, data: Dict[str, Any]) -> None:
        """Print epic information in markdown format.

        Args:
            data: Dictionary containing epic and issues data.
        """
        epic = data['epic']
        issues = data.get('issues', [])
        self._print_epic_markdown(epic, issues)

    def _print_epic_markdown(self, epic: Dict[str, Any], issues: List[Dict[str, Any]]) -> None:
        """Print epic info in markdown format."""
        print(f"# Epic &{epic.get('iid')}: {epic.get('title')}\n")

        print(f"**URL:** {epic.get('web_url')}  ")
        print(f"**State:** {epic.get('state')}  ")
        print(f"**Author:** {epic.get('author', {}).get('name', 'Unknown')}  ")

        labels = epic.get('labels', [])
        if labels:
            label_names = [label.get('name', label) if isinstance(label, dict) else label for label in labels]
            print(f"**Labels:** {', '.join([f'`{label}`' for label in label_names])}  ")

        if epic.get('start_date'):
            print(f"**Start Date:** {epic['start_date']}  ")

        if epic.get('due_date'):
            print(f"**Due Date:** {epic['due_date']}  ")

        print(f"\n**Created:** {epic.get('created_at')}  ")
        print(f"**Updated:** {epic.get('updated_at')}  ")

        # Print issues in the epic
        print(f"\n## Issues in Epic ({len(issues)})\n")

        if not issues:
            print("*No issues in this epic*\n")
        else:
            # Group issues by state
            opened_issues = [i for i in issues if i.get('state') == 'opened']
            closed_issues = [i for i in issues if i.get('state') == 'closed']

            if opened_issues:
                print(f"### Opened ({len(opened_issues)})\n")
                for issue in opened_issues:
                    iid = issue.get('iid')
                    title = issue.get('title', 'Untitled')
                    url = issue.get('web_url', '')
                    issue_labels = issue.get('labels', [])
                    label_str = ', '.join([f'`{label}`' for label in issue_labels[:3]]) if issue_labels else '*none*'
                    if len(issue_labels) > 3:
                        label_str += f' *+{len(issue_labels) - 3} more*'
                    print(f"- [#{iid} {title}]({url})")
                    print(f"  - Labels: {label_str}")
                print()

            if closed_issues:
                print(f"### Closed ({len(closed_issues)})\n")
                for issue in closed_issues:
                    iid = issue.get('iid')
                    title = issue.get('title', 'Untitled')
                    url = issue.get('web_url', '')
                    print(f"- [#{iid} {title}]({url})")
                print()

        print("## Description\n")
        description = epic.get('description', 'No description')
        print(description if description else '*No description*')
        print()

    def print_milestone_info(self, data: Dict[str, Any]) -> None:
        """Print milestone information in markdown format.

        Args:
            data: Dictionary containing milestone, issues, and epic mapping.
        """
        milestone = data['milestone']
        issues = data.get('issues', [])
        epic_map = data.get('epic_map', {})
        self._print_milestone_markdown(milestone, issues, epic_map)

    def _print_milestone_markdown(
        self,
        milestone: Dict[str, Any],
        issues: List[Dict[str, Any]],
        epic_map: Dict[int, Dict[str, Any]]
    ) -> None:
        """Print milestone info in markdown format."""
        print(f"# Milestone %{milestone.get('iid')}: {milestone.get('title')}\n")

        print(f"**URL:** {milestone.get('web_url')}  ")
        print(f"**State:** {milestone.get('state')}  ")

        if milestone.get('start_date'):
            print(f"**Start Date:** {milestone['start_date']}  ")

        if milestone.get('due_date'):
            print(f"**Due Date:** {milestone['due_date']}  ")

        # Calculate progress
        total_issues = len(issues)
        closed_issues = len([i for i in issues if i.get('state') == 'closed'])
        print(f"**Progress:** {closed_issues}/{total_issues} issues closed  ")

        print(f"\n**Created:** {milestone.get('created_at')}  ")
        print(f"**Updated:** {milestone.get('updated_at')}  ")

        # Group issues by epic
        print("\n## Epic Breakdown\n")

        if not issues:
            print("*No issues in this milestone*\n")
        else:
            # Group issues by epic_iid
            issues_by_epic: Dict[Any, List[Dict[str, Any]]] = {}
            issues_without_epic = []

            for issue in issues:
                epic_iid = issue.get('epic_iid')
                if epic_iid:
                    if epic_iid not in issues_by_epic:
                        issues_by_epic[epic_iid] = []
                    issues_by_epic[epic_iid].append(issue)
                else:
                    issues_without_epic.append(issue)

            # Print issues grouped by epic
            for epic_iid, epic_issues in sorted(issues_by_epic.items()):
                epic_data = epic_map.get(epic_iid)
                if epic_data:
                    epic_title = epic_data.get('title', 'Unknown')
                    print(f"### Epic &{epic_iid}: {epic_title}\n")
                else:
                    print(f"### Epic &{epic_iid}\n")

                for issue in epic_issues:
                    iid = issue.get('iid')
                    title = issue.get('title', 'Untitled')
                    state = issue.get('state', 'unknown')
                    print(f"- #{iid} {title} `[{state}]`")
                print()

            # Print issues without epic
            if issues_without_epic:
                print("### No Epic\n")
                for issue in issues_without_epic:
                    iid = issue.get('iid')
                    title = issue.get('title', 'Untitled')
                    state = issue.get('state', 'unknown')
                    print(f"- #{iid} {title} `[{state}]`")
                print()

        print("## Description\n")
        description = milestone.get('description', 'No description')
        print(description if description else '*No description*')
        print()

    def load_mr(self, mr_ref: str) -> Dict[str, Any]:
        """Load merge request information from GitLab.

        Args:
            mr_ref: MR reference (number, URL, or !number format).

        Returns:
            Dictionary containing MR data.

        Raises:
            PlatformError: If loading fails.
        """
        # Remove ! prefix if present
        if mr_ref.startswith('!'):
            mr_ref = mr_ref[1:]

        # Parse URL if provided
        if '://' in mr_ref:
            # Extract MR number from URL
            # Format: https://gitlab.../group/project/-/merge_requests/123
            if '/-/merge_requests/' in mr_ref:
                mr_ref = mr_ref.split('/-/merge_requests/')[-1].split('/')[0].split('?')[0]
            else:
                raise ValueError(f"Invalid MR URL format: {mr_ref}")

        if not mr_ref.isdigit():
            raise ValueError(f"Invalid MR reference: {mr_ref}")

        logger.debug("Loading MR !%s", mr_ref)
        # Use glab mr view command
        cmd = ['mr', 'view', mr_ref, '--output', 'json']
        output = self._run_glab_command(cmd)
        mr_data = json.loads(output)

        return {'mr': mr_data}

    def print_mr_info(self, data: Dict[str, Any]) -> None:
        """Print merge request information in markdown format.

        Args:
            data: Dictionary containing MR data.
        """
        mr = data['mr']
        print(f"# MR !{mr.get('iid')}: {mr.get('title')}\n")

        print(f"**URL:** {mr.get('web_url')}  ")
        print(f"**State:** {mr.get('state')}  ")
        print(f"**Author:** {mr.get('author', {}).get('name', 'Unknown')}  ")

        if mr.get('draft'):
            print("**Draft:** Yes  ")

        if mr.get('source_branch'):
            print(f"**Source Branch:** `{mr['source_branch']}`  ")

        if mr.get('target_branch'):
            print(f"**Target Branch:** `{mr['target_branch']}`  ")

        labels = mr.get('labels', [])
        if labels:
            print(f"**Labels:** {', '.join([f'`{label}`' for label in labels])}  ")

        assignees = mr.get('assignees', [])
        if assignees:
            names = [a.get('name', a.get('username')) for a in assignees]
            print(f"**Assignees:** {', '.join(names)}  ")

        reviewers = mr.get('reviewers', [])
        if reviewers:
            names = [r.get('name', r.get('username')) for r in reviewers]
            print(f"**Reviewers:** {', '.join(names)}  ")

        if mr.get('milestone'):
            print(f"**Milestone:** {mr['milestone'].get('title')}  ")

        print(f"\n**Created:** {mr.get('created_at')}  ")
        print(f"**Updated:** {mr.get('updated_at')}  ")

        if mr.get('merged_at'):
            print(f"**Merged:** {mr['merged_at']}  ")

        if mr.get('pipeline'):
            pipeline_status = mr['pipeline'].get('status', 'unknown')
            print(f"**Pipeline Status:** {pipeline_status}  ")

        print("\n## Description\n")
        description = mr.get('description', 'No description')
        print(description if description else '*No description*')
        print()
