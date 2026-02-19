"""Tests for ci_platform_manager.utils.config_migration module."""

from typing import Any, Dict

import pytest

from ci_platform_manager.utils.config_migration import transform_issue_template


class TestTransformIssueTemplate:
    """Test issue template transformation."""

    def test_transform_simple_template(self) -> None:
        """Transform simple issue template with required sections."""
        old_template = {
            "sections": [
                {"name": "Description", "required": True},
                {"name": "Acceptance Criteria", "required": True},
            ]
        }

        result = transform_issue_template(old_template)

        assert "required_sections" in result
        assert "Description" in result["required_sections"]
        assert "Acceptance Criteria" in result["required_sections"]

    def test_transform_mixed_sections(self) -> None:
        """Transform template with required and optional sections."""
        old_template = {
            "sections": [
                {"name": "Description", "required": True},
                {"name": "Notes", "required": False},
                {"name": "Acceptance Criteria", "required": True},
            ]
        }

        result = transform_issue_template(old_template)

        assert "Description" in result["required_sections"]
        assert "Acceptance Criteria" in result["required_sections"]
        # Optional sections not included
        assert "Notes" not in result["required_sections"]

    def test_transform_no_required_sections(self) -> None:
        """Transform template with no required sections defaults to Description."""
        old_template = {
            "sections": [
                {"name": "Notes", "required": False},
                {"name": "References", "required": False},
            ]
        }

        result = transform_issue_template(old_template)

        # Default to Description when no required sections
        assert "required_sections" in result
        assert "Description" in result["required_sections"]

    def test_transform_empty_template(self) -> None:
        """Transform empty template defaults to Description."""
        old_template: Dict[str, Any] = {}

        result = transform_issue_template(old_template)

        assert "required_sections" in result
        assert "Description" in result["required_sections"]

    def test_transform_missing_sections_key(self) -> None:
        """Transform template without sections key defaults to Description."""
        old_template = {"other_key": "value"}

        result = transform_issue_template(old_template)

        assert "required_sections" in result
        assert "Description" in result["required_sections"]

    def test_transform_preserves_section_order(self) -> None:
        """Transformation preserves order of required sections."""
        old_template = {
            "sections": [
                {"name": "Overview", "required": True},
                {"name": "Details", "required": True},
                {"name": "Summary", "required": True},
            ]
        }

        result = transform_issue_template(old_template)

        assert result["required_sections"] == ["Overview", "Details", "Summary"]

    def test_transform_section_without_required_flag(self) -> None:
        """Section without required flag treated as not required."""
        old_template = {
            "sections": [
                {"name": "Description", "required": True},
                {"name": "Notes"},  # No required flag
            ]
        }

        result = transform_issue_template(old_template)

        assert "Description" in result["required_sections"]
        assert "Notes" not in result["required_sections"]

    def test_transform_empty_sections_list(self) -> None:
        """Empty sections list defaults to Description."""
        old_template = {"sections": []}

        result = transform_issue_template(old_template)

        assert "required_sections" in result
        assert "Description" in result["required_sections"]
