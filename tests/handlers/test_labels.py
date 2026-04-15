"""Tests for projctl.handlers.labels module."""

from pathlib import Path

import pytest
import yaml

from projctl.config import Config
from projctl.handlers.labels import LabelsHandler


def _make_config(tmp_path: Path, config_data: dict) -> Config:
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)
    return Config(config_path)


class TestPrintLabels:
    """Test LabelsHandler.print_labels() output across all branches."""

    def test_allowed_labels_shown_without_or_groups(self, tmp_path: Path, capsys) -> None:
        """When allowed is set and no OR groups, prints allowed labels only."""
        config = _make_config(
            tmp_path,
            {
                "platform": "gitlab",
                "gitlab": {
                    "default_group": "g/p",
                    "labels": {
                        "default": ["type::feature"],
                        "allowed": ["type::feature", "type::bug"],
                    },
                },
            },
        )
        LabelsHandler(config).print_labels()
        out = capsys.readouterr().out
        assert "Configured labels" in out
        assert "type::feature" in out
        assert "type::bug" in out
        assert "Required" not in out

    def test_allowed_labels_shown_with_or_groups(self, tmp_path: Path, capsys) -> None:
        """OR groups are always printed even when allowed list is present (fix for H1)."""
        config = _make_config(
            tmp_path,
            {
                "platform": "gitlab",
                "gitlab": {
                    "default_group": "g/p",
                    "labels": {
                        "default": [
                            ["type::feature", "type::bug"],
                            "development-status::backlog",
                        ],
                        "allowed": ["type::feature", "type::bug", "development-status::backlog"],
                    },
                },
            },
        )
        LabelsHandler(config).print_labels()
        out = capsys.readouterr().out
        assert "Configured labels" in out
        # OR groups must appear regardless of allowed being set
        assert "Required (pick one per group)" in out
        assert "type::feature" in out
        assert "type::bug" in out

    def test_no_allowed_shows_default_labels(self, tmp_path: Path, capsys) -> None:
        """When allowed is None (not configured), default labels are shown with a note."""
        config = _make_config(
            tmp_path,
            {
                "platform": "gitlab",
                "gitlab": {
                    "default_group": "g/p",
                    "labels": {"default": ["type::feature", "development-status::backlog"]},
                },
            },
        )
        LabelsHandler(config).print_labels()
        out = capsys.readouterr().out
        assert "No allowed labels configured" in out
        assert "type::feature" in out

    def test_empty_allowed_shows_default_labels(self, tmp_path: Path, capsys) -> None:
        """When allowed is an empty list, default labels are shown with a note."""
        config = _make_config(
            tmp_path,
            {
                "platform": "gitlab",
                "gitlab": {
                    "default_group": "g/p",
                    "labels": {
                        "default": ["type::feature"],
                        "allowed": [],
                    },
                },
            },
        )
        LabelsHandler(config).print_labels()
        out = capsys.readouterr().out
        assert "Allowed labels list is empty" in out
        assert "type::feature" in out

    def test_only_or_groups_no_flat_defaults(self, tmp_path: Path, capsys) -> None:
        """When default contains only OR groups and allowed is absent, OR groups are shown."""
        config = _make_config(
            tmp_path,
            {
                "platform": "gitlab",
                "gitlab": {
                    "default_group": "g/p",
                    "labels": {
                        "default": [["type::feature", "type::bug"]],
                    },
                },
            },
        )
        LabelsHandler(config).print_labels()
        out = capsys.readouterr().out
        assert "Required (pick one per group)" in out
        assert "type::feature" in out

    def test_no_labels_at_all(self, tmp_path: Path, capsys) -> None:
        """When no labels are configured at all, prints a fallback message."""
        config = _make_config(
            tmp_path,
            {
                "platform": "gitlab",
                "gitlab": {"default_group": "g/p"},
            },
        )
        LabelsHandler(config).print_labels()
        out = capsys.readouterr().out
        assert "no labels configured" in out
