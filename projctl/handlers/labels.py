"""Labels handler — display configured labels from project config."""

import logging
from collections import defaultdict
from typing import Dict, List, Optional

from ..config import Config

logger = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
# LabelsHandler is intentionally a single-responsibility class with one public
# entry point (print_labels).  Adding artificial methods to satisfy the checker
# would violate the single-responsibility principle.
class LabelsHandler:
    """Displays allowed or default labels from the project configuration."""

    def __init__(self, config: Config) -> None:
        """Initialize the labels handler.

        Args:
            config: Configuration object with label settings.
        """
        self.config = config

    @staticmethod
    def _group_labels(labels: List[str]) -> Dict[str, List[str]]:
        """Group labels by their prefix (text before '::'}).

        Labels containing '::' are grouped under their prefix.  Labels without
        '::' are collected under the sentinel key '(ungrouped)'.

        Args:
            labels: List of label strings to group.

        Returns:
            Ordered dict mapping group name → sorted list of labels.
        """
        groups: Dict[str, List[str]] = defaultdict(list)
        ungrouped: List[str] = []

        for label in labels:
            if "::" in label:
                prefix = label.split("::")[0]
                groups[prefix].append(label)
            else:
                ungrouped.append(label)

        result: Dict[str, List[str]] = {}
        for key in sorted(groups):
            result[key] = sorted(groups[key])
        if ungrouped:
            result["(ungrouped)"] = sorted(ungrouped)

        return result

    @staticmethod
    def _print_groups(groups: Dict[str, List[str]]) -> None:
        """Print grouped labels in human-readable format.

        Args:
            groups: Mapping of group name → list of labels.
        """
        for group, members in groups.items():
            print(f"{group}::" if "::" not in group and group != "(ungrouped)" else group)
            for label in members:
                print(f"  {label}")
            print()

    @staticmethod
    def _print_or_groups(groups: List[List[str]]) -> None:
        """Print OR groups (required pick-one label sets) in human-readable format.

        Args:
            groups: List of OR groups from config.
        """
        for group in groups:
            print(f"  [{' | '.join(group)}]")
        print()

    def print_labels(self) -> None:
        """Print configured labels to stdout.

        If ``allowed`` labels are configured and non-empty they are shown with a
        'Configured labels' heading.  Otherwise the ``default`` labels are shown
        with a note that no allowed list is configured.  Required OR groups from
        ``default`` are always printed regardless of whether ``allowed`` is set.
        """
        allowed: Optional[List[str]] = self.config.get_allowed_labels()
        or_groups = self.config.get_required_label_groups()

        if allowed:
            config_name = (
                self.config.loaded_config_path.name
                if self.config.loaded_config_path
                else "config"
            )
            print(f"Configured labels (from {config_name}):\n")
            self._print_groups(self._group_labels(allowed))
        else:
            default_labels = self.config.get_default_labels()
            if allowed is None:
                print("No allowed labels configured. Showing default labels:\n")
            else:
                # allowed is an empty list — explicitly configured but empty
                print("Allowed labels list is empty. Showing default labels:\n")

            if default_labels:
                self._print_groups(self._group_labels(default_labels))
            elif not or_groups:
                print("(no labels configured)")

        if or_groups:
            print("Required (pick one per group):")
            self._print_or_groups(or_groups)
