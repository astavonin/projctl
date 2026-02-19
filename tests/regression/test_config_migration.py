"""Regression tests for configuration migration.

Ensures old config format works identically to new format.
"""

import warnings
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from ci_platform_manager.config import Config


class TestConfigMigration:
    """Ensure old config format works identically to new format."""

    def test_legacy_config_loads_without_error(self, legacy_config_path: Path) -> None:
        """Legacy config should load with deprecation warning."""
        with pytest.warns(DeprecationWarning):
            config = Config(legacy_config_path)
        assert config.platform == "gitlab"

    def test_legacy_issue_template_transform(self, legacy_config_path: Path) -> None:
        """Old issue_template.sections should map to required_sections."""
        with pytest.warns(DeprecationWarning):
            config = Config(legacy_config_path)

        required = config.get_required_sections()
        # Old format had "Description" and "Acceptance Criteria" as required
        assert "Description" in required
        assert "Acceptance Criteria" in required

    def test_legacy_labels_preserved(self, legacy_config_path: Path) -> None:
        """Label configuration should be preserved."""
        with pytest.warns(DeprecationWarning):
            config = Config(legacy_config_path)

        gitlab_config = config.get_platform_config("gitlab")
        assert "type::feature" in gitlab_config["labels"]["default"]
        assert "epic" in gitlab_config["labels"]["default_epic"]

    def test_legacy_default_group_preserved(self, legacy_config_path: Path) -> None:
        """Default group should be preserved in migration."""
        with pytest.warns(DeprecationWarning):
            config = Config(legacy_config_path)

        assert config.get_default_group() == "test/group"

    def test_legacy_allowed_labels_preserved(self, legacy_config_path: Path) -> None:
        """Allowed labels list should be preserved."""
        with pytest.warns(DeprecationWarning):
            config = Config(legacy_config_path)

        allowed = config.get_allowed_labels()
        assert allowed is not None
        assert "type::feature" in allowed
        assert "type::bug" in allowed

    def test_new_config_no_warnings(self, new_config_path: Path) -> None:
        """New config format should load without warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            config = Config(new_config_path)

        assert config.platform == "gitlab"

    def test_legacy_and_new_configs_equivalent(
        self, legacy_config_path: Path, new_config_path: Path
    ) -> None:
        """Legacy and new configs produce equivalent configuration."""
        with pytest.warns(DeprecationWarning):
            legacy_config = Config(legacy_config_path)

        new_config = Config(new_config_path)

        # Compare key configuration values
        assert legacy_config.platform == new_config.platform
        assert legacy_config.get_default_group() == new_config.get_default_group()
        assert legacy_config.get_default_labels() == new_config.get_default_labels()
        assert legacy_config.get_default_epic_labels() == new_config.get_default_epic_labels()
        assert set(legacy_config.get_required_sections()) == set(new_config.get_required_sections())

    def test_legacy_optional_sections_not_required(self, legacy_config_path: Path) -> None:
        """Optional sections in legacy format should not be required."""
        with pytest.warns(DeprecationWarning):
            config = Config(legacy_config_path)

        required = config.get_required_sections()
        # Notes and References are optional in fixture
        assert "Notes" not in required
        assert "Technical Notes" not in required
        assert "References" not in required


class TestBackwardCompatibility:
    """Test backward compatibility with legacy config files."""

    def test_legacy_config_file_from_fixtures(self, temp_dir: Path, monkeypatch) -> None:
        """Real legacy config file from fixtures loads correctly."""
        fixtures_dir = Path(__file__).parent.parent / "fixtures"
        legacy_config = fixtures_dir / "config_old.yaml"

        with pytest.warns(DeprecationWarning):
            config = Config(legacy_config)

        # Verify essential config is loaded
        assert config.get_default_group() is not None
        assert len(config.get_default_labels()) > 0
        assert len(config.get_required_sections()) > 0

    def test_new_config_file_from_fixtures(self) -> None:
        """New config file from fixtures loads correctly."""
        fixtures_dir = Path(__file__).parent.parent / "fixtures"
        new_config = fixtures_dir / "config_new.yaml"

        config = Config(new_config)

        # Verify multi-platform config
        assert config.platform == "gitlab"
        assert "gitlab" in config.config_data
        assert "common" in config.config_data

    def test_project_local_legacy_config_priority(
        self,
        temp_dir: Path,
        legacy_config_data: Dict[str, Any],
        new_config_data: Dict[str, Any],
        monkeypatch,
    ) -> None:
        """Project-local glab_config.yaml takes priority over new format."""
        monkeypatch.chdir(temp_dir)

        # Create both legacy and new config in project dir
        legacy_path = temp_dir / "glab_config.yaml"
        new_path = temp_dir / "config.yaml"

        with open(legacy_path, "w", encoding="utf-8") as file:
            yaml.dump(legacy_config_data, file)

        with open(new_path, "w", encoding="utf-8") as file:
            yaml.dump(new_config_data, file)

        # Legacy should be loaded (with warning)
        with pytest.warns(DeprecationWarning, match="legacy config location"):
            config = Config()

        assert config.loaded_config_path == legacy_path

    def test_fallback_to_new_format_when_no_legacy(
        self, temp_dir: Path, new_config_data: Dict[str, Any], monkeypatch
    ) -> None:
        """Fallback to config.yaml when glab_config.yaml doesn't exist."""
        monkeypatch.chdir(temp_dir)

        # Create only new format config
        new_path = temp_dir / "config.yaml"
        with open(new_path, "w", encoding="utf-8") as file:
            yaml.dump(new_config_data, file)

        # Should load without warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            config = Config()

        assert config.loaded_config_path == new_path


class TestMigrationEdgeCases:
    """Test edge cases in config migration."""

    def test_legacy_config_no_issue_template(self, temp_dir: Path) -> None:
        """Legacy config without issue_template section."""
        legacy_no_template = {
            "gitlab": {"default_group": "test/group"},
            "labels": {
                "default": ["type::feature"],
                "default_epic": ["epic"],
                "allowed_labels": ["type::feature"],
            },
        }

        config_path = temp_dir / "glab_config.yaml"
        with open(config_path, "w", encoding="utf-8") as file:
            yaml.dump(legacy_no_template, file)

        with pytest.warns(DeprecationWarning):
            config = Config(config_path)

        # Should default to requiring Description
        required = config.get_required_sections()
        assert "Description" in required

    def test_legacy_config_empty_allowed_labels(self, temp_dir: Path) -> None:
        """Legacy config with empty allowed_labels."""
        legacy_empty_labels = {
            "gitlab": {"default_group": "test/group"},
            "labels": {"default": [], "default_epic": [], "allowed_labels": []},
            "issue_template": {"sections": [{"name": "Description", "required": True}]},
        }

        config_path = temp_dir / "glab_config.yaml"
        with open(config_path, "w", encoding="utf-8") as file:
            yaml.dump(legacy_empty_labels, file)

        with pytest.warns(DeprecationWarning):
            config = Config(config_path)

        # Empty list should be preserved
        allowed = config.get_allowed_labels()
        assert allowed == []

    def test_legacy_config_all_optional_sections(self, temp_dir: Path) -> None:
        """Legacy config with all sections marked as optional."""
        legacy_all_optional = {
            "gitlab": {"default_group": "test/group"},
            "labels": {"default": [], "default_epic": [], "allowed_labels": []},
            "issue_template": {
                "sections": [
                    {"name": "Notes", "required": False},
                    {"name": "References", "required": False},
                ]
            },
        }

        config_path = temp_dir / "glab_config.yaml"
        with open(config_path, "w", encoding="utf-8") as file:
            yaml.dump(legacy_all_optional, file)

        with pytest.warns(DeprecationWarning):
            config = Config(config_path)

        # Should default to requiring Description
        required = config.get_required_sections()
        assert "Description" in required
