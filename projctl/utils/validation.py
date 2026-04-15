"""Input validation helpers."""

from typing import List, Optional, Sequence


def validate_labels(labels: List[str], allowed_labels: Optional[List[str]] = None) -> None:
    """Validate that labels are in the allowed list.

    Args:
        labels: List of label names to validate.
        allowed_labels: List of allowed label names, or None to skip validation.

    Raises:
        ValueError: If any label is not in the allowed list.
    """
    if allowed_labels is None:
        # No validation configured
        return

    if not allowed_labels:
        # Empty allowed list means no labels are allowed
        if labels:
            raise ValueError(
                f"Labels are not allowed (allowed_labels is empty), but found: {', '.join(labels)}"
            )
        return

    # Check each label
    unknown_labels = []
    for label in labels:
        if label not in allowed_labels:
            unknown_labels.append(label)

    if unknown_labels:
        raise ValueError(
            f"Unknown labels found: {', '.join(unknown_labels)}\n"
            f"Allowed labels are: {', '.join(allowed_labels)}"
        )


def validate_required_label_groups(labels: Sequence[str], groups: List[List[str]]) -> None:
    """Validate that exactly one label from each OR group is present.

    Args:
        labels: Final label list applied to the issue.
        groups: OR groups from config. Each group requires exactly one match.

    Raises:
        ValueError: If a group has no matches or more than one match.
    """
    for group in groups:
        matches = [lbl for lbl in labels if lbl in group]
        if not matches:
            raise ValueError(f"Missing required label — exactly one of: {', '.join(group)}")
        if len(matches) > 1:
            raise ValueError(
                f"Conflicting labels — only one of [{', '.join(group)}] is allowed, "
                f"got: {', '.join(matches)}"
            )


def validate_issue_description(
    description: str, required_sections: List[str], issue_title: str = "unknown"
) -> None:
    """Validate that issue description contains required sections.

    Args:
        description: Issue description text.
        required_sections: List of required section names.
        issue_title: Issue title for error messages.

    Raises:
        ValueError: If required sections are missing from description.
    """
    if not required_sections:
        return

    if not description:
        missing = ", ".join(required_sections)
        raise ValueError(
            f"Issue '{issue_title}' has no description. " f"Required sections: {missing}"
        )

    missing_sections = []
    for section in required_sections:
        # Look for "# Section" or "## Section" patterns
        if f"# {section}" not in description:
            missing_sections.append(section)

    if missing_sections:
        missing = ", ".join(missing_sections)
        raise ValueError(f"Issue '{issue_title}' missing required sections: {missing}")
