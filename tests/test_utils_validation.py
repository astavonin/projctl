"""Tests for ci_platform_manager.utils.validation module."""

from typing import List, Optional

import pytest

from ci_platform_manager.utils.validation import validate_issue_description, validate_labels


class TestValidateLabels:
    """Test label validation."""

    def test_validate_labels_success(self) -> None:
        """Valid labels pass validation."""
        labels = ["type::feature", "priority::high"]
        allowed = ["type::feature", "type::bug", "priority::high", "priority::low"]

        # Should not raise
        validate_labels(labels, allowed)

    def test_validate_labels_unknown(self) -> None:
        """Unknown labels raise ValueError."""
        labels = ["type::feature", "unknown::label"]
        allowed = ["type::feature", "type::bug"]

        with pytest.raises(ValueError, match="Unknown labels found: unknown::label"):
            validate_labels(labels, allowed)

    def test_validate_labels_multiple_unknown(self) -> None:
        """Multiple unknown labels are reported."""
        labels = ["type::feature", "unknown1", "unknown2"]
        allowed = ["type::feature"]

        with pytest.raises(ValueError, match="unknown1"):
            validate_labels(labels, allowed)

    def test_validate_labels_none_allowed(self) -> None:
        """None for allowed_labels skips validation."""
        labels = ["any::label", "another::label"]

        # Should not raise
        validate_labels(labels, None)

    def test_validate_labels_empty_allowed_no_labels(self) -> None:
        """Empty allowed list with no labels is valid."""
        labels: List[str] = []
        allowed: List[str] = []

        # Should not raise
        validate_labels(labels, allowed)

    def test_validate_labels_empty_allowed_with_labels(self) -> None:
        """Empty allowed list with labels raises ValueError."""
        labels = ["type::feature"]
        allowed: List[str] = []

        with pytest.raises(ValueError, match="Labels are not allowed"):
            validate_labels(labels, allowed)

    def test_validate_labels_case_sensitive(self) -> None:
        """Label validation is case-sensitive."""
        labels = ["Type::Feature"]
        allowed = ["type::feature"]

        with pytest.raises(ValueError, match="Unknown labels found: Type::Feature"):
            validate_labels(labels, allowed)


class TestValidateIssueDescription:
    """Test issue description validation."""

    def test_validate_description_success(self) -> None:
        """Valid description with all required sections passes."""
        description = """
# Description

This is the description.

# Acceptance Criteria

- Criterion 1
- Criterion 2
"""
        required = ["Description", "Acceptance Criteria"]

        # Should not raise
        validate_issue_description(description, required, "Test Issue")

    def test_validate_description_missing_section(self) -> None:
        """Description missing required section raises ValueError."""
        description = """
# Description

This is the description.
"""
        required = ["Description", "Acceptance Criteria"]

        with pytest.raises(ValueError, match="missing required sections: Acceptance Criteria"):
            validate_issue_description(description, required, "Test Issue")

    def test_validate_description_multiple_missing(self) -> None:
        """Multiple missing sections are reported."""
        description = """
# Notes

Some notes.
"""
        required = ["Description", "Acceptance Criteria", "Technical Details"]

        with pytest.raises(ValueError, match="missing required sections"):
            validate_issue_description(description, required, "Test Issue")

    def test_validate_description_no_requirements(self) -> None:
        """Empty required sections list allows any description."""
        description = "Any description without sections"
        required: List[str] = []

        # Should not raise
        validate_issue_description(description, required)

    def test_validate_description_empty_description(self) -> None:
        """Empty description with requirements raises ValueError."""
        description = ""
        required = ["Description"]

        with pytest.raises(ValueError, match="has no description"):
            validate_issue_description(description, required, "Empty Issue")

    def test_validate_description_with_h2_headers(self) -> None:
        """Description with ## headers is valid."""
        description = """
## Description

This uses h2 headers.

## Acceptance Criteria

- Criterion 1
"""
        required = ["Description", "Acceptance Criteria"]

        # Should not raise
        validate_issue_description(description, required)

    def test_validate_description_case_sensitive(self) -> None:
        """Section names are case-sensitive."""
        description = """
# description

Lower case header.
"""
        required = ["Description"]

        with pytest.raises(ValueError, match="missing required sections: Description"):
            validate_issue_description(description, required)

    def test_validate_description_includes_issue_title(self) -> None:
        """Error message includes issue title."""
        description = ""
        required = ["Description"]

        with pytest.raises(ValueError, match="Issue 'My Test Issue' has no description"):
            validate_issue_description(description, required, "My Test Issue")

    def test_validate_description_default_issue_title(self) -> None:
        """Default issue title used when not provided."""
        description = ""
        required = ["Description"]

        with pytest.raises(ValueError, match="Issue 'unknown' has no description"):
            validate_issue_description(description, required)
