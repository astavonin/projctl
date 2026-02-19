"""Tests for ci_platform_manager.config module."""

import warnings
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from ci_platform_manager.config import Config, ConfigurationError


class TestConfigLoading:
    """Test configuration file loading."""

    def test_load_new_format_config(self, new_config_path: Path) -> None:
        """New format config loads without warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            config = Config(new_config_path)

        assert config.platform == "gitlab"
        assert config.loaded_config_path == new_config_path

    def test_load_legacy_format_config(self, legacy_config_path: Path) -> None:
        """Legacy format config loads with deprecation warning."""
        with pytest.warns(DeprecationWarning, match="deprecated format"):
            config = Config(legacy_config_path)

        assert config.platform == "gitlab"

    def test_explicit_config_path_not_found(self, temp_dir: Path) -> None:
        """Raises FileNotFoundError when explicit config doesn't exist."""
        nonexistent = temp_dir / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            Config(nonexistent)

    def test_config_search_order(
        self, temp_dir: Path, new_config_data: Dict[str, Any], monkeypatch
    ) -> None:
        """Config search follows correct priority order."""
        # Change to temp dir
        monkeypatch.chdir(temp_dir)

        # Create config in current directory
        local_config = temp_dir / "glab_config.yaml"
        with open(local_config, "w", encoding="utf-8") as file:
            yaml.dump(new_config_data, file)

        with pytest.warns(DeprecationWarning, match="legacy config location"):
            config = Config()

        assert config.loaded_config_path == local_config

    def test_no_config_found_error(self, temp_dir: Path, monkeypatch) -> None:
        """Raises FileNotFoundError when no config found."""
        monkeypatch.chdir(temp_dir)

        with pytest.raises(FileNotFoundError, match="No config file found"):
            Config()


class TestConfigTransformation:
    """Test legacy config transformation."""

    def test_legacy_to_new_transformation(self, legacy_config_path: Path) -> None:
        """Legacy config transforms to new format correctly."""
        with pytest.warns(DeprecationWarning):
            config = Config(legacy_config_path)

        # Check platform is set
        assert config.platform == "gitlab"

        # Check GitLab config preserved
        gitlab_config = config.get_platform_config("gitlab")
        assert gitlab_config["default_group"] == "test/group"
        assert "type::feature" in gitlab_config["labels"]["default"]

    def test_legacy_issue_template_transform(self, legacy_config_path: Path) -> None:
        """Legacy issue_template.sections maps to required_sections."""
        with pytest.warns(DeprecationWarning):
            config = Config(legacy_config_path)

        required = config.get_required_sections()
        assert "Description" in required
        assert "Acceptance Criteria" in required
        # Notes is not required in legacy fixture
        assert "Notes" not in required

    def test_legacy_labels_preserved(self, legacy_config_path: Path) -> None:
        """Label configuration preserved in transformation."""
        with pytest.warns(DeprecationWarning):
            config = Config(legacy_config_path)

        gitlab_config = config.get_platform_config("gitlab")
        assert "type::feature" in gitlab_config["labels"]["default"]
        assert "epic" in gitlab_config["labels"]["default_epic"]
        assert "type::bug" in gitlab_config["labels"]["allowed"]


class TestConfigGetters:
    """Test configuration getter methods."""

    def test_get_platform_config(self, new_config_path: Path) -> None:
        """get_platform_config returns correct platform data."""
        config = Config(new_config_path)

        gitlab_config = config.get_platform_config("gitlab")
        assert gitlab_config["default_group"] == "test/group"

        # Non-existent platform returns empty dict
        unknown_config = config.get_platform_config("unknown")
        assert unknown_config == {}

    def test_get_common_config(self, new_config_path: Path) -> None:
        """get_common_config returns common configuration."""
        config = Config(new_config_path)

        common_config = config.get_common_config()
        assert "issue_template" in common_config
        assert "required_sections" in common_config["issue_template"]

    def test_get_required_sections_new_format(self, new_config_path: Path) -> None:
        """get_required_sections returns sections from new format."""
        config = Config(new_config_path)

        sections = config.get_required_sections()
        assert "Description" in sections
        assert "Acceptance Criteria" in sections

    def test_get_required_sections_legacy_format(self, legacy_config_path: Path) -> None:
        """get_required_sections works with legacy format."""
        with pytest.warns(DeprecationWarning):
            config = Config(legacy_config_path)

        sections = config.get_required_sections()
        assert "Description" in sections
        assert "Acceptance Criteria" in sections

    def test_get_default_group(self, new_config_path: Path) -> None:
        """get_default_group returns GitLab default group."""
        config = Config(new_config_path)

        group = config.get_default_group()
        assert group == "test/group"

    def test_get_default_labels(self, new_config_path: Path) -> None:
        """get_default_labels returns default labels for platform."""
        config = Config(new_config_path)

        labels = config.get_default_labels()
        assert "type::feature" in labels
        assert "development-status::backlog" in labels

    def test_get_default_epic_labels(self, new_config_path: Path) -> None:
        """get_default_epic_labels returns default epic labels."""
        config = Config(new_config_path)

        labels = config.get_default_epic_labels()
        assert "epic" in labels

    def test_get_allowed_labels(self, new_config_path: Path) -> None:
        """get_allowed_labels returns allowed labels list."""
        config = Config(new_config_path)

        allowed = config.get_allowed_labels()
        assert allowed is not None
        assert "type::feature" in allowed
        assert "type::bug" in allowed

    def test_get_allowed_labels_none(self, temp_dir: Path) -> None:
        """get_allowed_labels returns None when not configured."""
        # Create minimal config without allowed labels
        minimal_config = {"platform": "gitlab", "gitlab": {"default_group": "test"}}
        config_path = temp_dir / "minimal.yaml"
        with open(config_path, "w", encoding="utf-8") as file:
            yaml.dump(minimal_config, file)

        config = Config(config_path)
        allowed = config.get_allowed_labels()
        assert allowed is None


class TestPlatformOverride:
    """Test platform override functionality."""

    def test_platform_override(self, new_config_path: Path) -> None:
        """Platform can be overridden via constructor."""
        config = Config(new_config_path, platform="github")

        assert config.platform == "github"

    def test_platform_from_config(self, new_config_path: Path) -> None:
        """Platform defaults to config value."""
        config = Config(new_config_path)

        assert config.platform == "gitlab"

    def test_platform_default(self, temp_dir: Path) -> None:
        """Platform defaults to 'gitlab' when not in config."""
        minimal_config = {"gitlab": {"default_group": "test"}}
        config_path = temp_dir / "minimal.yaml"
        with open(config_path, "w", encoding="utf-8") as file:
            yaml.dump(minimal_config, file)

        config = Config(config_path)
        assert config.platform == "gitlab"


class TestPlanningSyncConfig:
    """Test planning_sync configuration."""

    def test_planning_sync_in_new_format(self, temp_dir: Path) -> None:
        """planning_sync config is loaded in new format."""
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/project"},
            "planning_sync": {"gdrive_base": "~/GoogleDrive"}
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as file:
            yaml.dump(config_data, file)

        config = Config(config_path)
        assert config.planning_sync == {"gdrive_base": "~/GoogleDrive"}

    def test_planning_sync_in_legacy_format(self, temp_dir: Path) -> None:
        """planning_sync config is preserved during legacy transformation."""
        legacy_config = {
            "gitlab": {"default_group": "test/project"},
            "labels": {
                "default": ["type::feature"],
                "allowed_labels": ["type::feature"]
            },
            "planning_sync": {"gdrive_base": "~/GoogleDrive"}
        }
        config_path = temp_dir / "legacy.yaml"
        with open(config_path, "w", encoding="utf-8") as file:
            yaml.dump(legacy_config, file)

        with pytest.warns(DeprecationWarning):
            config = Config(config_path)

        # Verify planning_sync was preserved during transformation
        assert config.planning_sync == {"gdrive_base": "~/GoogleDrive"}

    def test_planning_sync_missing(self, temp_dir: Path) -> None:
        """planning_sync is empty dict when not configured."""
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/project"}
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as file:
            yaml.dump(config_data, file)

        config = Config(config_path)
        assert config.planning_sync == {}
