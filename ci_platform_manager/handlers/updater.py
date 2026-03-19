"""Ticket (issue/MR/epic/milestone) updater handler."""

import json
import logging
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

from ..config import Config
from .loader import TicketLoader

logger = logging.getLogger(__name__)


class TicketUpdater:
    """Updates GitLab issues, MRs, epics, and milestones using the glab CLI."""

    def __init__(self, config: Config, dry_run: bool = False) -> None:
        """Initialize the updater.

        Args:
            config: Configuration object with defaults.
            dry_run: If True, print what would be sent without executing.
        """
        self.config = config
        self.dry_run = dry_run
        # Reuse loader for reference parsing, label fetching, and glab execution.
        self._loader = TicketLoader(config)

    def _fetch_and_merge_labels(
        self,
        endpoint: str,
        labels_add: Optional[List[str]],
        labels_remove: Optional[List[str]],
        *,
        label_key: str = "labels",
    ) -> Optional[str]:
        """Fetch current labels from a resource and compute the merged set.

        GitLab's PUT endpoint replaces labels entirely, so we must fetch the
        current set and compute the desired result ourselves.

        Args:
            endpoint: GitLab API path for the resource.
            labels_add: Labels to add.
            labels_remove: Labels to remove.
            label_key: JSON key that holds labels (differs for epics vs issues).

        Returns:
            Comma-separated label string, or None if no label change requested.
        """
        if not labels_add and not labels_remove:
            return None

        # pylint: disable=protected-access
        # TicketLoader's _run_glab_command is an internal helper shared between
        # sibling handler classes; no public API exists for command execution.
        current_data = json.loads(self._loader._run_glab_command(["api", endpoint]))
        raw_labels = current_data.get(label_key, [])
        # Epic labels may be dicts with a 'name' key; issue/MR labels are plain strings.
        current_labels: List[str] = [
            lbl["name"] if isinstance(lbl, dict) else lbl for lbl in raw_labels
        ]
        return ",".join(self._merge_labels(current_labels, labels_add, labels_remove))

    def _merge_labels(
        self,
        current: List[str],
        add: Optional[List[str]],
        remove: Optional[List[str]],
    ) -> List[str]:
        """Compute the final label list after additions and removals.

        Args:
            current: Labels currently on the resource.
            add: Labels to add.
            remove: Labels to remove.

        Returns:
            Final label list.
        """
        result = set(current)
        if add:
            result.update(add)
        if remove:
            result.difference_update(remove)
        return sorted(result)

    def _build_put_cmd(
        self,
        endpoint: str,
        fields: Dict[str, Any],
    ) -> List[str]:
        """Build a glab api -X PUT command from an endpoint and field map.

        Args:
            endpoint: GitLab API path (e.g. 'projects/:id/issues/1').
            fields: Mapping of field name to value. None values are skipped.

        Returns:
            Argument list for _loader._run_glab_command (without 'glab' prefix).
        """
        cmd = ["api", "-X", "PUT", endpoint]
        for key, value in fields.items():
            if value is not None:
                cmd.extend(["-f", f"{key}={value}"])
        return cmd

    def _resolve_user_id(self, username: str) -> str:
        """Resolve a GitLab username to its numeric user ID.

        GitLab's PUT API requires numeric IDs for assignee_ids and reviewer_ids,
        not usernames.

        Args:
            username: GitLab username to resolve.

        Returns:
            Numeric user ID as a string.

        Raises:
            ValueError: If no user is found for the given username.
            PlatformError: If the API call fails.
        """
        encoded = urllib.parse.quote(username, safe="")
        # pylint: disable=protected-access
        # TicketLoader's _run_glab_command is an internal helper shared between
        # sibling handler classes; no public API exists for command execution.
        output = self._loader._run_glab_command(["api", f"users?username={encoded}"])
        users = json.loads(output)
        if not users:
            raise ValueError(f"No GitLab user found for username: {username!r}")
        return str(users[0]["id"])

    def _resolve_milestone_id(self, milestone_ref: str, project_path: Optional[str] = None) -> str:
        """Resolve a milestone title or iid string to its numeric ID.

        GitLab's PUT API requires the milestone's database ID (not iid) in the
        milestone_id field. This method searches by iid or title.

        Args:
            milestone_ref: Milestone iid (as string) or title (e.g. "v2.0").
            project_path: Project namespace path. Defaults to ':fullpath' sentinel.

        Returns:
            Numeric milestone ID as a string.

        Raises:
            ValueError: If no matching milestone is found.
            PlatformError: If the API call fails.
        """
        if project_path:
            encoded_path = urllib.parse.quote(project_path, safe="")
        else:
            encoded_path = ":fullpath"

        # pylint: disable=protected-access
        # TicketLoader's _run_glab_command is an internal helper shared between
        # sibling handler classes; no public API exists for command execution.
        output = self._loader._run_glab_command(["api", f"projects/{encoded_path}/milestones"])
        milestones = json.loads(output)
        for m in milestones:
            if str(m.get("iid")) == milestone_ref or m.get("title") == milestone_ref:
                return str(m["id"])
        raise ValueError(f"Milestone not found: {milestone_ref!r}")

    def update_issue(
        self,
        issue_ref: str,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        labels_add: Optional[List[str]] = None,
        labels_remove: Optional[List[str]] = None,
        assignee: Optional[str] = None,
        milestone: Optional[str] = None,
        state_event: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update an existing GitLab issue.

        Args:
            issue_ref: Issue reference (number, URL, or #number format).
            title: New title, or None to leave unchanged.
            description: New description, or None to leave unchanged.
            labels_add: Labels to add to the current set.
            labels_remove: Labels to remove from the current set.
            assignee: Assignee username to set, or None to leave unchanged.
            milestone: Milestone title or iid to set, or None to leave unchanged.
            state_event: 'close' or 'reopen', or None to leave unchanged.

        Returns:
            Updated issue data returned by the API.

        Raises:
            PlatformError: If the update fails.
            ValueError: If the issue reference cannot be parsed.
        """
        # pylint: disable=protected-access
        # TicketLoader's reference-parsing methods are internal helpers shared
        # between sibling handler classes; no public API exists for them.
        project_path, iid = self._loader._parse_issue_reference(issue_ref)

        if project_path:
            encoded_project = urllib.parse.quote(project_path, safe="")
        else:
            encoded_project = ":fullpath"

        endpoint = f"projects/{encoded_project}/issues/{iid}"

        fields: Dict[str, Any] = {
            "title": title,
            "description": description,
            "state_event": state_event,
        }

        if self.dry_run:
            # Show intent for labels without making a live API call.
            if labels_add or labels_remove:
                fields["labels"] = f"<add: {labels_add or []}, remove: {labels_remove or []}>"
            if assignee is not None:
                fields["assignee_ids"] = f"<resolve user: {assignee}>"
            if milestone is not None:
                fields["milestone_id"] = f"<resolve milestone: {milestone}>"
            print(f"[DRY RUN] Would PUT {endpoint} with fields: {fields}")
            return {}

        labels_value = self._fetch_and_merge_labels(endpoint, labels_add, labels_remove)
        if labels_value is not None:
            fields["labels"] = labels_value
        if assignee is not None:
            fields["assignee_ids"] = self._resolve_user_id(assignee)
        if milestone is not None:
            fields["milestone_id"] = self._resolve_milestone_id(milestone, project_path)

        # pylint: disable=protected-access
        # TicketLoader's _run_glab_command is an internal helper shared between
        # sibling handler classes; no public API exists for command execution.
        output = self._loader._run_glab_command(self._build_put_cmd(endpoint, fields))
        result: Dict[str, Any] = json.loads(output)

        ref_display = result.get("iid", iid)
        result_title = result.get("title", "")
        print(f"✓ Updated issue #{ref_display}: {result_title}")
        return result

    def update_mr(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        mr_ref: str,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        labels_add: Optional[List[str]] = None,
        labels_remove: Optional[List[str]] = None,
        assignee: Optional[str] = None,
        reviewer: Optional[str] = None,
        milestone: Optional[str] = None,
        target_branch: Optional[str] = None,
        state_event: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update an existing GitLab merge request.

        MRs have more updatable fields than other resources (reviewer, target_branch),
        so the argument count intentionally exceeds the default pylint limit.

        Args:
            mr_ref: MR reference (number, URL, or !number format).
            title: New title, or None to leave unchanged.
            description: New description, or None to leave unchanged.
            labels_add: Labels to add to the current set.
            labels_remove: Labels to remove from the current set.
            assignee: Assignee username to set, or None to leave unchanged.
            reviewer: Reviewer username to set, or None to leave unchanged.
            milestone: Milestone title or iid to set, or None to leave unchanged.
            target_branch: Target branch to set, or None to leave unchanged.
            state_event: 'close' or 'reopen', or None to leave unchanged.

        Returns:
            Updated MR data returned by the API.

        Raises:
            PlatformError: If the update fails.
            ValueError: If the MR reference cannot be parsed.
        """
        project_path, iid = self._parse_mr_reference(mr_ref)

        if project_path:
            encoded_project = urllib.parse.quote(project_path, safe="")
            endpoint = f"projects/{encoded_project}/merge_requests/{iid}"
        else:
            endpoint = f"projects/:id/merge_requests/{iid}"

        fields: Dict[str, Any] = {
            "title": title,
            "description": description,
            "state_event": state_event,
            "target_branch": target_branch,
        }

        if self.dry_run:
            # Show intent for labels without making a live API call.
            if labels_add or labels_remove:
                fields["labels"] = f"<add: {labels_add or []}, remove: {labels_remove or []}>"
            if assignee is not None:
                fields["assignee_ids"] = f"<resolve user: {assignee}>"
            if reviewer is not None:
                fields["reviewer_ids"] = f"<resolve user: {reviewer}>"
            if milestone is not None:
                fields["milestone_id"] = f"<resolve milestone: {milestone}>"
            print(f"[DRY RUN] Would PUT {endpoint} with fields: {fields}")
            return {}

        labels_value = self._fetch_and_merge_labels(endpoint, labels_add, labels_remove)
        if labels_value is not None:
            fields["labels"] = labels_value
        if assignee is not None:
            fields["assignee_ids"] = self._resolve_user_id(assignee)
        if reviewer is not None:
            fields["reviewer_ids"] = self._resolve_user_id(reviewer)
        if milestone is not None:
            fields["milestone_id"] = self._resolve_milestone_id(milestone, project_path)

        # pylint: disable=protected-access
        # TicketLoader's _run_glab_command is an internal helper shared between
        # sibling handler classes; no public API exists for command execution.
        output = self._loader._run_glab_command(self._build_put_cmd(endpoint, fields))
        result: Dict[str, Any] = json.loads(output)

        ref_display = result.get("iid", iid)
        result_title = result.get("title", "")
        print(f"✓ Updated mr #{ref_display}: {result_title}")
        return result

    def update_epic(
        self,
        epic_ref: str,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        labels_add: Optional[List[str]] = None,
        labels_remove: Optional[List[str]] = None,
        state_event: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update an existing GitLab epic.

        Args:
            epic_ref: Epic reference (number, URL, or &number format).
            title: New title, or None to leave unchanged.
            description: New description, or None to leave unchanged.
            labels_add: Labels to add to the current set.
            labels_remove: Labels to remove from the current set.
            state_event: 'close' or 'reopen', or None to leave unchanged.

        Returns:
            Updated epic data returned by the API.

        Raises:
            PlatformError: If the update fails.
            ValueError: If the epic reference cannot be parsed or group is unavailable.
        """
        # pylint: disable=protected-access
        # TicketLoader's reference-parsing methods are internal helpers shared
        # between sibling handler classes; no public API exists for them.
        parsed_group, iid = self._loader._parse_epic_reference(epic_ref)
        group_path = parsed_group or self.config.get_default_group()

        if not group_path:
            raise ValueError(
                "Group path is required to update epic.\n"
                "Either include the group in the URL or set 'default_group' in your config."
            )

        encoded_group = urllib.parse.quote(group_path, safe="")
        endpoint = f"groups/{encoded_group}/epics/{iid}"

        fields: Dict[str, Any] = {
            "title": title,
            "description": description,
            "state_event": state_event,
        }

        if self.dry_run:
            # Show intent for labels without making a live API call.
            if labels_add or labels_remove:
                fields["labels"] = f"<add: {labels_add or []}, remove: {labels_remove or []}>"
            print(f"[DRY RUN] Would PUT {endpoint} with fields: {fields}")
            return {}

        # Epic labels are returned as dicts with a 'name' key.
        labels_value = self._fetch_and_merge_labels(
            endpoint, labels_add, labels_remove, label_key="labels"
        )
        if labels_value is not None:
            fields["labels"] = labels_value

        # pylint: disable=protected-access
        # TicketLoader's _run_glab_command is an internal helper shared between
        # sibling handler classes; no public API exists for command execution.
        output = self._loader._run_glab_command(self._build_put_cmd(endpoint, fields))
        result: Dict[str, Any] = json.loads(output)

        ref_display = result.get("iid", iid)
        result_title = result.get("title", "")
        print(f"✓ Updated epic #{ref_display}: {result_title}")
        return result

    def update_milestone(
        self,
        milestone_ref: str,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        due_date: Optional[str] = None,
        state_event: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update an existing GitLab milestone.

        Args:
            milestone_ref: Milestone reference (number, URL, or %number format).
            title: New title, or None to leave unchanged.
            description: New description, or None to leave unchanged.
            due_date: New due date in YYYY-MM-DD format, or None to leave unchanged.
            state_event: 'close' or 'activate', or None to leave unchanged.

        Returns:
            Updated milestone data returned by the API.

        Raises:
            PlatformError: If the update fails.
            ValueError: If the milestone reference cannot be parsed.
        """
        # pylint: disable=protected-access
        # TicketLoader's reference-parsing methods are internal helpers shared
        # between sibling handler classes; no public API exists for them.
        parsed_path, iid, is_group_milestone = self._loader._parse_milestone_reference(
            milestone_ref
        )

        # Reuse loader's shared endpoint resolver to avoid duplication.
        api_endpoint, _ = self._loader._resolve_milestone_endpoints(
            parsed_path, iid, is_group_milestone
        )
        # Strip the trailing /issues query part to get the base milestone endpoint.
        endpoint = api_endpoint

        fields: Dict[str, Any] = {
            "title": title,
            "description": description,
            "due_date": due_date,
            "state_event": state_event,
        }
        # None values are skipped by _build_put_cmd; no need for an extra
        # dict comprehension here (consistent with update_issue/update_mr/update_epic).

        if self.dry_run:
            print(f"[DRY RUN] Would PUT {endpoint} with fields: {fields}")
            return {}

        # pylint: disable=protected-access
        # TicketLoader's _run_glab_command is an internal helper shared between
        # sibling handler classes; no public API exists for command execution.
        output = self._loader._run_glab_command(self._build_put_cmd(endpoint, fields))
        result: Dict[str, Any] = json.loads(output)

        ref_display = result.get("iid", iid)
        result_title = result.get("title", "")
        print(f"✓ Updated milestone #{ref_display}: {result_title}")
        return result

    @staticmethod
    def _parse_mr_reference(mr_ref: str) -> Tuple[Optional[str], str]:
        """Parse an MR reference to extract the optional project path and iid.

        Args:
            mr_ref: MR reference (!number, URL, or plain number).

        Returns:
            Tuple of (project_path, iid). project_path is None for non-URL refs.

        Raises:
            ValueError: If the reference cannot be parsed.
        """
        ref = mr_ref.lstrip("!")

        project_path: Optional[str] = None
        if "/-/merge_requests/" in ref:
            parts = ref.split("/-/merge_requests/")
            raw_path = parts[0]
            iid_part = parts[1].split("/")[0].split("?")[0]
            # Strip scheme and host from URL references (e.g. https://gitlab.com/group/project).
            if "://" in raw_path:
                raw_path = "/".join(raw_path.split("/")[3:])
            project_path = raw_path if raw_path else None
            ref = iid_part

        if not ref.isdigit():
            raise ValueError(f"Cannot parse MR reference: {mr_ref}")

        return project_path, ref
