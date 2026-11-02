"""Label-history handler: reads an issue's label add/remove timeline.

Answers questions a current label set cannot: when a ticket left a scope, what
it moved to, and who moved it. The motivating case is iteration close-out —
"which issues left `Iteration::2`, and when" — where the only surviving
evidence is the label event log, since the issue now carries the new label and
nothing records the old one.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import Config
from ..exceptions import PlatformError
from ..utils.git_helpers import encoded_issue_target
from ..utils.glab_runner import run_glab_json_pages

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelEvent:
    """One label add or removal on a resource."""

    date: str
    action: str
    label: str
    username: str

    def as_dict(self) -> Dict[str, str]:
        """Return the event as a plain dict for JSON output."""
        return {
            "date": self.date,
            "action": self.action,
            "label": self.label,
            "username": self.username,
        }


class LabelHistoryHandler:
    """Reads label events for a GitLab issue via the glab CLI."""

    def __init__(self, config: Config) -> None:
        """Initialize the handler.

        Args:
            config: Retained for symmetry with the other handlers; the issue
                endpoint resolves its project from the working directory's
                remote via glab's `:fullpath`, so no group lookup is needed.
        """
        self.config = config

    @staticmethod
    def _parse_events(raw: List[Any]) -> List[LabelEvent]:
        """Convert the API's event objects into LabelEvent records.

        An event whose label has been deleted since arrives with `label: null`;
        it is skipped rather than rendered as an empty name, because a blank row
        reads as a label called "" rather than as a label that no longer exists.

        Args:
            raw: Parsed JSON items from the resource_label_events endpoint.

        Returns:
            Events in API order, which is chronological by `created_at`.

        Raises:
            PlatformError: If an item is not a JSON object.
        """
        events: List[LabelEvent] = []
        for item in raw:
            if not isinstance(item, dict):
                raise PlatformError(f"Expected label event objects, got: {item!r}")
            # The nested objects get the same check as the item: `label: null` is
            # a real shape (a deleted label) handled below, but a non-null
            # non-object would reach .get() and raise AttributeError, which
            # escapes the caller's PlatformError/ValueError handler as a
            # traceback rather than a named error.
            label = item.get("label") or {}
            user = item.get("user") or {}
            if not isinstance(label, dict) or not isinstance(user, dict):
                raise PlatformError(f"Expected label event objects, got: {item!r}")
            name = label.get("name")
            if not name:
                logger.debug("Skipping event %s: label was deleted", item.get("id"))
                continue
            events.append(
                LabelEvent(
                    date=(item.get("created_at") or "")[:10],
                    action=item.get("action") or "?",
                    label=name,
                    username=user.get("username") or "unknown",
                )
            )
        return events

    def issue_events(self, issue_ref: str, label_prefix: str = "") -> List[LabelEvent]:
        """Fetch an issue's label events, oldest first.

        Args:
            issue_ref: Issue reference (#N, N, or URL).
            label_prefix: When set, keep only labels starting with this string.

        Returns:
            Matching events in chronological order.

        Raises:
            ValueError: If the reference cannot be parsed.
            PlatformError: If the API call fails or returns an unexpected shape.
        """
        project, iid = encoded_issue_target(issue_ref)
        path = project if project else ":fullpath"
        endpoint = f"projects/{path}/issues/{iid}/resource_label_events"
        raw = run_glab_json_pages(["api", "--paginate", endpoint])
        events = self._parse_events(raw)
        if label_prefix:
            events = [e for e in events if e.label.startswith(label_prefix)]
        return events

    @staticmethod
    def find_removal(events: List[LabelEvent], label: str) -> Optional[LabelEvent]:
        """Return the removal that left an exact label off, if it is off now.

        The label's **last** event decides, not its last removal. Scope labels
        get re-added, and not rarely: in one real project four issues dropped a
        scope label and carried it again three weeks later, and two more lost
        and regained it inside a single day. Returning the last removal reports
        those as departures while the issue still carries the label — which
        inflates a deferral list with work that never left.

        Args:
            events: Events for one resource, chronological.
            label: Exact label name to look for.

        Returns:
            The removal event that left the label off, or None when the label
            is still applied or was never applied at all.
        """
        for event in reversed(events):
            if event.label == label:
                return event if event.action == "remove" else None
        return None

    @staticmethod
    def epic_unsupported_message(iid: str) -> str:
        """Explain why epic label history is unavailable.

        Args:
            iid: Epic iid, for the message.

        Returns:
            A message naming the cause and the usable alternative.
        """
        return (
            f"Label history is not available for epic &{iid}: GitLab serves no "
            "resource_label_events endpoint for group epics (it answers 404, the same "
            "way the group-epic notes endpoint does), and the work-item GraphQL schema "
            "exposes current labels but not label events. Query the epic's issues "
            "individually instead — their history is complete."
        )
