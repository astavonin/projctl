"""Tests for projctl.config module."""

import warnings
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest
import yaml

from projctl.config import (
    PROJECT_LOCAL_CONFIG_NAMES,
    Config,
    ConfigurationError,
    config_search_paths,
)


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

        with pytest.warns(DeprecationWarning, match="legacy config name"):
            config = Config()

        assert config.loaded_config_path == local_config

    def test_no_config_found_error(self, temp_dir: Path, monkeypatch) -> None:
        """Raises FileNotFoundError when no config found."""
        monkeypatch.chdir(temp_dir)
        # Redirect Path.home() to the temp dir so the user-level config is not found.
        monkeypatch.setattr(Path, "home", lambda: temp_dir)

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

    def test_get_default_labels_flat_only(self, temp_dir: Path) -> None:
        """get_default_labels returns only flat strings, excluding OR groups."""
        config_data = {
            "platform": "gitlab",
            "gitlab": {
                "default_group": "test/group",
                "labels": {
                    "default": [
                        ["type::feature", "type::bug"],
                        "development-status::backlog",
                    ],
                },
            },
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        config = Config(config_path)
        labels = config.get_default_labels()

        assert "development-status::backlog" in labels
        assert "type::feature" not in labels
        assert "type::bug" not in labels

    def test_get_required_label_groups_returns_or_groups(self, temp_dir: Path) -> None:
        """get_required_label_groups returns inner lists, excluding flat strings."""
        config_data = {
            "platform": "gitlab",
            "gitlab": {
                "default_group": "test/group",
                "labels": {
                    "default": [
                        ["type::feature", "type::bug"],
                        "development-status::backlog",
                        ["priority::high", "priority::low"],
                    ],
                },
            },
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        config = Config(config_path)
        groups = config.get_required_label_groups()

        assert len(groups) == 2
        assert ["type::feature", "type::bug"] in groups
        assert ["priority::high", "priority::low"] in groups

    def test_get_required_label_groups_empty_when_all_flat(self, new_config_path: Path) -> None:
        """get_required_label_groups returns empty list when default has no OR groups."""
        config = Config(new_config_path)
        assert config.get_required_label_groups() == []

    def test_malformed_dict_entry_raises_configuration_error(self, temp_dir: Path) -> None:
        """A dict item in labels.default raises ConfigurationError at access time."""
        config_data = {
            "platform": "gitlab",
            "gitlab": {
                "default_group": "test/group",
                "labels": {"default": [{"type::feature": None}]},
            },
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        config = Config(config_path)
        with pytest.raises(ConfigurationError, match="unexpected type"):
            config.get_default_labels()

    def test_empty_or_group_raises_configuration_error(self, temp_dir: Path) -> None:
        """An empty list item in labels.default raises ConfigurationError."""
        config_data = {
            "platform": "gitlab",
            "gitlab": {
                "default_group": "test/group",
                "labels": {"default": [[], "development-status::backlog"]},
            },
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        config = Config(config_path)
        with pytest.raises(ConfigurationError, match="empty OR group"):
            config.get_default_labels()

    def test_non_string_inner_item_raises_configuration_error(self, temp_dir: Path) -> None:
        """A list whose members include a non-string raises ConfigurationError."""
        config_data = {
            "platform": "gitlab",
            "gitlab": {
                "default_group": "test/group",
                "labels": {"default": [["type::feature", 42]]},
            },
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        config = Config(config_path)
        with pytest.raises(ConfigurationError, match="non-string members"):
            config.get_default_labels()


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
            "planning_sync": {"gdrive_base": "~/GoogleDrive"},
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
            "labels": {"default": ["type::feature"], "allowed_labels": ["type::feature"]},
            "planning_sync": {"gdrive_base": "~/GoogleDrive"},
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
        config_data = {"platform": "gitlab", "gitlab": {"default_group": "test/project"}}
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as file:
            yaml.dump(config_data, file)

        config = Config(config_path)
        assert config.planning_sync == {}


class TestGetGithubRepo:
    """Test get_github_repo() method."""

    def test_get_github_repo_from_config(self, temp_dir: Path) -> None:
        """Returns explicit repo value from config when present."""
        cfg_path = temp_dir / "config.yaml"
        cfg_path.write_text("platform: github\n" "github:\n" "  repo: myorg/myrepo\n")
        config = Config(cfg_path)
        assert config.get_github_repo() == "myorg/myrepo"

    def test_get_github_repo_auto_detected(self, temp_dir: Path) -> None:
        """Falls back to git remote when repo not in config."""
        cfg_path = temp_dir / "config.yaml"
        cfg_path.write_text("platform: github\ngithub: {}\n")
        config = Config(cfg_path)

        with patch("projctl.config.get_current_repo_path", return_value="detected/repo"):
            assert config.get_github_repo() == "detected/repo"

    def test_get_github_repo_raises_when_unresolvable(self, temp_dir: Path) -> None:
        """ConfigurationError raised when both config and git remote are absent."""
        cfg_path = temp_dir / "config.yaml"
        cfg_path.write_text("platform: github\ngithub: {}\n")
        config = Config(cfg_path)

        with patch("projctl.config.get_current_repo_path", return_value=None):
            with pytest.raises(ConfigurationError, match="Cannot determine GitHub repository"):
                config.get_github_repo()


class TestGetRequiredEpicSections:
    """Test get_required_epic_sections() method."""

    def test_key_absent_returns_default(self, temp_dir: Path) -> None:
        """Returns ['Description'] when epic_template key is absent from config."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {"issue_template": {"required_sections": ["Description"]}},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)
        sections = config.get_required_epic_sections()

        # Assert
        assert sections == ["Description"]

    def test_key_present_with_values_returns_configured_list(self, temp_dir: Path) -> None:
        """Returns configured list when epic_template.required_sections is set."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {
                "epic_template": {"required_sections": ["Overview", "Goals"]},
            },
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)
        sections = config.get_required_epic_sections()

        # Assert
        assert sections == ["Overview", "Goals"]

    def test_key_present_but_empty_returns_empty_list(self, temp_dir: Path) -> None:
        """Returns [] when epic_template.required_sections is explicitly empty."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {
                "epic_template": {"required_sections": []},
            },
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)
        sections = config.get_required_epic_sections()

        # Assert
        assert sections == []


class TestGetRequiredMrSections:
    """Test get_required_mr_sections() method."""

    def test_key_absent_returns_default(self, temp_dir: Path) -> None:
        """Returns default MR sections when mr_template key is absent from config."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {"issue_template": {"required_sections": ["Description"]}},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)
        sections = config.get_required_mr_sections()

        # Assert
        assert sections == ["Summary", "Implementation Details", "How It Was Tested"]

    def test_key_present_with_values_returns_configured_list(self, temp_dir: Path) -> None:
        """Returns configured list when mr_template.required_sections is set."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {
                "mr_template": {"required_sections": ["Summary", "Testing"]},
            },
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)
        sections = config.get_required_mr_sections()

        # Assert
        assert sections == ["Summary", "Testing"]

    def test_key_present_but_empty_returns_empty_list(self, temp_dir: Path) -> None:
        """Returns [] when mr_template.required_sections is explicitly empty."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {
                "mr_template": {"required_sections": []},
            },
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)
        sections = config.get_required_mr_sections()

        # Assert
        assert sections == []


class TestGetRequiredFields:
    """Tests for get_required_issue_fields, get_required_epic_fields, get_required_mr_fields."""

    # -------------------------------------------------------------------
    # H1 regression: sibling key must not disturb the other sub-key default
    # -------------------------------------------------------------------

    def test_mr_template_with_only_required_fields_sections_returns_default(
        self, temp_dir: Path
    ) -> None:
        """mr_template present with only required_fields → get_required_mr_sections returns default."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {
                "mr_template": {"required_fields": ["reviewers"]},
            },
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)
        sections = config.get_required_mr_sections()

        # Assert — required_sections absent under mr_template → default returned
        assert sections == ["Summary", "Implementation Details", "How It Was Tested"]

    def test_mr_template_with_only_required_sections_fields_returns_empty(
        self, temp_dir: Path
    ) -> None:
        """mr_template present with only required_sections → get_required_mr_fields returns []."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {
                "mr_template": {"required_sections": ["Summary", "Implementation Details"]},
            },
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)
        fields = config.get_required_mr_fields()

        # Assert — required_fields absent under mr_template → []
        assert fields == []

    def test_epic_template_with_only_required_fields_sections_returns_default(
        self, temp_dir: Path
    ) -> None:
        """epic_template present with only required_fields → get_required_epic_sections returns default."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {
                "epic_template": {"required_fields": ["some_field"]},
            },
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act — suppress the unknown-field warning for the assertion
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = Config(config_path)
            sections = config.get_required_epic_sections()

        # Assert
        assert sections == ["Description"]

    # -------------------------------------------------------------------
    # get_required_issue_fields
    # -------------------------------------------------------------------

    def test_issue_fields_common_absent_gitlab_returns_weight(self, temp_dir: Path) -> None:
        """common: absent, platform=gitlab → returns ["weight"]."""
        # Arrange
        config_data = {"platform": "gitlab", "gitlab": {"default_group": "test/group"}}
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)

        # Assert
        assert config.get_required_issue_fields() == ["weight"]

    def test_issue_fields_issue_template_absent_gitlab_returns_weight(self, temp_dir: Path) -> None:
        """issue_template absent under common, platform=gitlab → returns ["weight"]."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)

        # Assert
        assert config.get_required_issue_fields() == ["weight"]

    def test_issue_fields_only_required_sections_gitlab_returns_weight(
        self, temp_dir: Path
    ) -> None:
        """issue_template has required_sections only (no required_fields), gitlab → ["weight"]."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {
                "issue_template": {"required_sections": ["Description", "Acceptance Criteria"]}
            },
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)

        # Assert
        assert config.get_required_issue_fields() == ["weight"]

    def test_issue_fields_empty_required_fields_returns_empty(self, temp_dir: Path) -> None:
        """required_fields: [] → returns []."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {"issue_template": {"required_fields": []}},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)

        # Assert
        assert config.get_required_issue_fields() == []

    def test_issue_fields_required_fields_weight_returns_weight(self, temp_dir: Path) -> None:
        """required_fields: ["weight"] → returns ["weight"]."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {"issue_template": {"required_fields": ["weight"]}},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)

        # Assert
        assert config.get_required_issue_fields() == ["weight"]

    def test_issue_fields_github_no_required_fields_returns_empty(self, temp_dir: Path) -> None:
        """platform=github, required_fields absent → returns []."""
        # Arrange
        config_data = {
            "platform": "github",
            "github": {"repo": "owner/repo"},
            "common": {"issue_template": {"required_sections": ["Description"]}},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)

        # Assert
        assert config.get_required_issue_fields() == []

    def test_issue_fields_unknown_name_emits_warning(self, temp_dir: Path) -> None:
        """Unknown field name in required_fields → warning emitted, field included in return."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {"issue_template": {"required_fields": ["weight", "unknown_field"]}},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act / Assert
        config = Config(config_path)
        with pytest.warns(UserWarning, match="Unknown required_fields"):
            fields = config.get_required_issue_fields()

        assert "unknown_field" in fields

    # -------------------------------------------------------------------
    # get_required_epic_fields
    # -------------------------------------------------------------------

    def test_epic_fields_key_absent_returns_empty(self, temp_dir: Path) -> None:
        """required_fields key absent from epic_template → []."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {"epic_template": {"required_sections": ["Description"]}},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)

        # Assert
        assert config.get_required_epic_fields() == []

    def test_epic_fields_key_present_with_values_returns_list(self, temp_dir: Path) -> None:
        """required_fields present with values → returns configured list."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {"epic_template": {"required_fields": ["future_field"]}},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act / Assert — suppress unknown-field warning
        config = Config(config_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fields = config.get_required_epic_fields()

        assert fields == ["future_field"]

    def test_epic_fields_key_present_but_empty_returns_empty(self, temp_dir: Path) -> None:
        """required_fields: [] → []."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {"epic_template": {"required_fields": []}},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)

        # Assert
        assert config.get_required_epic_fields() == []

    # -------------------------------------------------------------------
    # get_required_mr_fields
    # -------------------------------------------------------------------

    def test_mr_fields_key_absent_returns_empty(self, temp_dir: Path) -> None:
        """required_fields key absent from mr_template → []."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {"mr_template": {"required_sections": ["Summary"]}},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)

        # Assert
        assert config.get_required_mr_fields() == []

    def test_mr_fields_key_present_with_values_returns_list(self, temp_dir: Path) -> None:
        """required_fields: ["reviewers", "labels"] → returns configured list."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {"mr_template": {"required_fields": ["reviewers", "labels"]}},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)
        fields = config.get_required_mr_fields()

        # Assert
        assert fields == ["reviewers", "labels"]

    def test_mr_fields_key_present_but_empty_returns_empty(self, temp_dir: Path) -> None:
        """required_fields: [] → []."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {"mr_template": {"required_fields": []}},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)

        # Assert
        assert config.get_required_mr_fields() == []

    def test_mr_fields_unknown_name_emits_warning(self, temp_dir: Path) -> None:
        """Unknown field name "reviewer" in mr required_fields → warning emitted, included in return."""
        # Arrange — "reviewer" (singular) is not a known name; "reviewers" (plural) is
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {"mr_template": {"required_fields": ["reviewer"]}},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act / Assert
        config = Config(config_path)
        with pytest.warns(UserWarning, match="Unknown required_fields"):
            fields = config.get_required_mr_fields()

        assert "reviewer" in fields

    # -------------------------------------------------------------------
    # M1: type validation for required_fields (scalar → ConfigurationError, null → [])
    # -------------------------------------------------------------------

    def test_issue_fields_scalar_string_raises_configuration_error(self, temp_dir: Path) -> None:
        """required_fields: 'weight' (scalar string) → raises ConfigurationError."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {"issue_template": {"required_fields": "weight"}},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act / Assert
        config = Config(config_path)
        with pytest.raises(
            ConfigurationError, match="issue_template.required_fields must be a list"
        ):
            config.get_required_issue_fields()

    def test_issue_fields_null_returns_empty(self, temp_dir: Path) -> None:
        """required_fields: null (YAML null → Python None) → returns []."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {"issue_template": {"required_fields": None}},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)
        fields = config.get_required_issue_fields()

        # Assert — None is handled by `or []`; no error
        assert fields == []

    def test_epic_fields_scalar_string_raises_configuration_error(self, temp_dir: Path) -> None:
        """epic required_fields: 'some_field' (scalar string) → raises ConfigurationError."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {"epic_template": {"required_fields": "some_field"}},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act / Assert
        config = Config(config_path)
        with pytest.raises(
            ConfigurationError, match="epic_template.required_fields must be a list"
        ):
            config.get_required_epic_fields()

    def test_mr_fields_scalar_string_raises_configuration_error(self, temp_dir: Path) -> None:
        """mr required_fields: 'reviewers' (scalar string) → raises ConfigurationError."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {"mr_template": {"required_fields": "reviewers"}},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act / Assert
        config = Config(config_path)
        with pytest.raises(ConfigurationError, match="mr_template.required_fields must be a list"):
            config.get_required_mr_fields()


# ---------------------------------------------------------------------------
# TestGetDefaultMrReviewers
# ---------------------------------------------------------------------------


class TestGetDefaultMrReviewers:
    """Tests for Config.get_default_mr_reviewers()."""

    def test_absent_reviewers_returns_empty(self, temp_dir: Path) -> None:
        """No mr_template.reviewers key → []."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {"mr_template": {"required_sections": ["Summary"]}},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)

        # Assert
        assert config.get_default_mr_reviewers() == []

    def test_empty_reviewers_returns_empty(self, temp_dir: Path) -> None:
        """mr_template.reviewers: [] → []."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {"mr_template": {"reviewers": []}},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)

        # Assert
        assert config.get_default_mr_reviewers() == []

    def test_configured_reviewers_returned(self, temp_dir: Path) -> None:
        """mr_template.reviewers: [alice, bob] → ['alice', 'bob']."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {"mr_template": {"reviewers": ["alice", "bob"]}},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act
        config = Config(config_path)

        # Assert
        assert config.get_default_mr_reviewers() == ["alice", "bob"]

    def test_non_list_raises_configuration_error(self, temp_dir: Path) -> None:
        """mr_template.reviewers: 'alice' (scalar) → ConfigurationError."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {"mr_template": {"reviewers": "alice"}},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act / Assert
        config = Config(config_path)
        with pytest.raises(ConfigurationError, match="mr_template.reviewers must be a list"):
            config.get_default_mr_reviewers()

    def test_non_string_element_raises_configuration_error(self, temp_dir: Path) -> None:
        """mr_template.reviewers: [123] (non-string element) → ConfigurationError."""
        # Arrange
        config_data = {
            "platform": "gitlab",
            "gitlab": {"default_group": "test/group"},
            "common": {"mr_template": {"reviewers": [123]}},
        }
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Act / Assert
        config = Config(config_path)
        with pytest.raises(
            ConfigurationError, match="mr_template.reviewers must be a list of strings"
        ):
            config.get_default_mr_reviewers()


# ---------------------------------------------------------------------------
# TestConfigSearchPaths
# ---------------------------------------------------------------------------


class TestConfigSearchPaths:
    """Tests for the config_search_paths() module-level helper."""

    def test_returns_four_entries(self) -> None:
        """config_search_paths() returns exactly 4 (path, label) tuples."""
        paths = config_search_paths()
        assert len(paths) == 4

    def test_all_entries_are_tuples_of_path_and_str(self) -> None:
        """Every entry is a (Path, str) tuple."""
        for path, label in config_search_paths():
            assert isinstance(path, Path)
            assert isinstance(label, str) and label

    def test_first_two_entries_are_cwd_relative(self, monkeypatch, tmp_path: Path) -> None:
        """The first two entries are under the current working directory."""
        monkeypatch.chdir(tmp_path)
        paths = config_search_paths()
        assert paths[0][0].parent == tmp_path
        assert paths[1][0].parent == tmp_path

    def test_last_two_entries_are_home_relative(self) -> None:
        """The last two entries are under the user home directory."""
        home = Path.home()
        paths = config_search_paths()
        assert str(paths[2][0]).startswith(str(home))
        assert str(paths[3][0]).startswith(str(home))

    def test_legacy_name_appears_before_preferred_name(self) -> None:
        """glab_config.yaml (legacy) appears before projctl.yaml (preferred) in order."""
        paths = config_search_paths()
        names = [p.name for p, _ in paths]
        assert names.index("glab_config.yaml") < names.index("projctl.yaml")

    def test_labels_are_human_readable(self) -> None:
        """Every label contains the file name so it is self-describing."""
        for path, label in config_search_paths():
            assert path.name in label

    def test_user_config_appears_before_legacy_user_config(self) -> None:
        """~/.config/projctl/config.yaml appears before ~/.config/glab_config.yaml."""
        paths = config_search_paths()
        user_idx = next(i for i, (p, _) in enumerate(paths) if ".config/projctl" in str(p))
        legacy_idx = next(
            i
            for i, (p, _) in enumerate(paths)
            if p.name == "glab_config.yaml" and ".config" in str(p)
        )
        assert user_idx < legacy_idx


class TestProjectLocalConfigNames:
    """PROJECT_LOCAL_CONFIG_NAMES is the single source config_search_paths()
    and docs_search.py's per-project probe both read, so they cannot disagree
    about order (see design.md §5.2)."""

    def test_legacy_name_precedes_preferred_name(self) -> None:
        assert PROJECT_LOCAL_CONFIG_NAMES == ("glab_config.yaml", "projctl.yaml")

    def test_config_search_paths_first_two_entries_use_the_same_constant(self) -> None:
        paths = config_search_paths()
        assert paths[0][0].name == PROJECT_LOCAL_CONFIG_NAMES[0]
        assert paths[1][0].name == PROJECT_LOCAL_CONFIG_NAMES[1]


class TestSearchConfigAccessor:
    """Config.get_search_config() — the search: section (design.md §5.2, §6)."""

    def test_absent_search_key_returns_defaults(self, temp_dir: Path) -> None:
        cfg_path = temp_dir / "projctl.yaml"
        cfg_path.write_text("planning_sync:\n  gdrive_base: ~/GoogleDrive\n")
        config = Config(cfg_path)
        search_config = config.get_search_config()
        assert search_config.docs_path == "docs"
        assert search_config.docs_path_configured is False
        assert search_config.related == []

    def test_configured_docs_path_string_reports_configured_true(self, temp_dir: Path) -> None:
        cfg_path = temp_dir / "projctl.yaml"
        cfg_path.write_text("search:\n  docs_path: documentation\n")
        config = Config(cfg_path)
        search_config = config.get_search_config()
        assert search_config.docs_path == "documentation"
        assert search_config.docs_path_configured is True

    def test_configured_docs_path_null_reports_configured_true(self, temp_dir: Path) -> None:
        cfg_path = temp_dir / "projctl.yaml"
        cfg_path.write_text("search:\n  docs_path: null\n")
        config = Config(cfg_path)
        search_config = config.get_search_config()
        assert search_config.docs_path is None
        assert search_config.docs_path_configured is True

    def test_related_without_docs_path_keeps_the_documented_default(self, temp_dir: Path) -> None:
        cfg_path = temp_dir / "projctl.yaml"
        cfg_path.write_text("search:\n  related:\n    - ../sibling\n")
        search_config = Config(cfg_path).get_search_config()
        assert search_config.docs_path == "docs"
        assert search_config.docs_path_configured is False

    def test_search_as_a_scalar_raises_naming_the_key(self, temp_dir: Path) -> None:
        cfg_path = temp_dir / "projctl.yaml"
        cfg_path.write_text('search: "on"\n')
        config = Config(cfg_path)
        with pytest.raises(ConfigurationError, match="search"):
            config.get_search_config()

    def test_search_present_as_null_is_rejected_rather_than_read_as_absent(
        self, temp_dir: Path
    ) -> None:
        # A present key asserts a shape; silently defaulting it hides a
        # truncated section from the operator who wrote it (§5.2).
        cfg_path = temp_dir / "projctl.yaml"
        cfg_path.write_text("search:\n")
        config = Config(cfg_path)
        with pytest.raises(ConfigurationError, match="search must be a mapping"):
            config.get_search_config()

    def test_docs_path_wrong_type_raises_naming_the_key(self, temp_dir: Path) -> None:
        cfg_path = temp_dir / "projctl.yaml"
        cfg_path.write_text("search:\n  docs_path: 3\n")
        config = Config(cfg_path)
        with pytest.raises(ConfigurationError, match="docs_path"):
            config.get_search_config()

    def test_related_wrong_type_raises_naming_the_key(self, temp_dir: Path) -> None:
        cfg_path = temp_dir / "projctl.yaml"
        cfg_path.write_text("search:\n  related: ../x\n")
        config = Config(cfg_path)
        with pytest.raises(ConfigurationError, match="related"):
            config.get_search_config()

    def test_related_non_string_member_names_its_index(self, temp_dir: Path) -> None:
        cfg_path = temp_dir / "projctl.yaml"
        cfg_path.write_text("search:\n  related:\n    - ../ok\n    - 7\n")
        config = Config(cfg_path)
        with pytest.raises(ConfigurationError, match=r"related\[1\]"):
            config.get_search_config()

    def test_legacy_config_carrying_search_still_yields_it_with_undeclared_platform(
        self, temp_dir: Path
    ) -> None:
        """A legacy (pre-platform:) config that happens to carry search: must not
        have it silently dropped by _transform_legacy_config's whitelist, and
        its platform must read as undeclared rather than the gitlab default."""
        cfg_path = temp_dir / "glab_config.yaml"
        cfg_path.write_text(
            "labels:\n  default: []\ngitlab:\n  default_group: g\nsearch:\n  docs_path: docs\n"
        )
        with pytest.warns(DeprecationWarning):
            config = Config(cfg_path)
        assert config.get_search_config().docs_path == "docs"
        assert config.get_raw_platform_or_undeclared() == "undeclared"


class TestRawPlatformOrUndeclared:
    def test_declared_platform_is_returned_as_is(self, temp_dir: Path) -> None:
        cfg_path = temp_dir / "projctl.yaml"
        cfg_path.write_text("platform: github\n")
        config = Config(cfg_path)
        assert config.get_raw_platform_or_undeclared() == "github"

    def test_absent_platform_key_is_undeclared_not_the_gitlab_default(self, temp_dir: Path) -> None:
        cfg_path = temp_dir / "projctl.yaml"
        cfg_path.write_text("planning_sync:\n  gdrive_base: ~/GoogleDrive\n")
        config = Config(cfg_path)
        assert config.platform == "gitlab"  # dispatch still defaults to gitlab
        assert config.get_raw_platform_or_undeclared() == "undeclared"

    def test_a_non_string_platform_value_is_rendered_as_written(self, temp_dir: Path) -> None:
        cfg_path = temp_dir / "projctl.yaml"
        cfg_path.write_text("platform: 3\n")
        config = Config(cfg_path)
        assert config.get_raw_platform_or_undeclared() == "3"


class TestRawConfigDataIsolation:
    """raw_config_data is the as-parsed mapping, and stays that way."""

    def test_the_raw_mapping_is_not_the_transformed_one(self, temp_dir: Path) -> None:
        cfg_path = temp_dir / "projctl.yaml"
        cfg_path.write_text("platform: github\nsearch:\n  docs_path: docs\n")
        config = Config(cfg_path)
        assert config.config_data is not config.raw_config_data

    def test_a_later_write_to_config_data_cannot_surface_as_a_declared_platform(
        self, temp_dir: Path
    ) -> None:
        cfg_path = temp_dir / "projctl.yaml"
        cfg_path.write_text("gitlab:\n  default_group: g\n")
        config = Config(cfg_path)
        config.config_data["platform"] = "gitlab"
        assert config.get_raw_platform_or_undeclared() == "undeclared"

    def test_a_nested_write_to_config_data_does_not_reach_the_raw_mapping(
        self, temp_dir: Path
    ) -> None:
        cfg_path = temp_dir / "projctl.yaml"
        cfg_path.write_text("platform: gitlab\nsearch:\n  related:\n    - ../a\n")
        config = Config(cfg_path)
        config.config_data["search"]["related"].append("../injected")
        assert config.get_search_config().related == ["../a"]


# ---------------------------------------------------------------------------
# PRODUCT-SHIPPED compatibility matrix (design.md §6) — the shapes the managed
# repositories use must load exactly as they did before `search:` existed.
# ---------------------------------------------------------------------------

_MANAGED_CONFIG_SHAPES: Dict[str, str] = {
    "gitlab-flat-labels": (
        "platform: gitlab\n"
        "gitlab:\n"
        "  default_group: group/project\n"
        "  labels:\n"
        "    default:\n"
        "      - development-status::backlog\n"
    ),
    "gitlab-or-group": (
        "platform: gitlab\n"
        "gitlab:\n"
        "  default_group: group/project\n"
        "  labels:\n"
        "    default:\n"
        "      - [type::feature, type::bug]\n"
        "      - development-status::backlog\n"
    ),
    "gitlab-allowed-present": (
        "platform: gitlab\n"
        "gitlab:\n"
        "  default_group: group/project\n"
        "  labels:\n"
        "    default: [development-status::backlog]\n"
        "    allowed: [type::feature, type::bug]\n"
    ),
    "gitlab-allowed-empty": (
        "platform: gitlab\n"
        "gitlab:\n"
        "  default_group: group/project\n"
        "  labels:\n"
        "    default: [development-status::backlog]\n"
        "    allowed: []\n"
    ),
    "gitlab-default-epic": (
        "platform: gitlab\n"
        "gitlab:\n"
        "  default_group: group/project\n"
        "  labels:\n"
        "    default: [development-status::backlog]\n"
        "    default_epic: [type::epic]\n"
    ),
    "gitlab-with-planning-sync": (
        "platform: gitlab\n"
        "gitlab:\n"
        "  default_group: group/project\n"
        "  labels:\n"
        "    default: [development-status::backlog]\n"
        "planning_sync:\n"
        "  gdrive_base: ~/GoogleDrive\n"
    ),
    "gitlab-with-templates": (
        "platform: gitlab\n"
        "gitlab:\n"
        "  default_group: group/project\n"
        "  labels:\n"
        "    default: [development-status::backlog]\n"
        "common:\n"
        "  issue_template:\n"
        "    required_sections: [Description, Acceptance Criteria]\n"
        "  mr_template:\n"
        "    required_sections: [Summary]\n"
        "    required_fields: [reviewers, labels]\n"
        "    reviewers: [alice, bob]\n"
    ),
    "github-repo": (
        "platform: github\n"
        "github:\n"
        "  repo: org/repo\n"
        "  labels:\n"
        "    default: [development-status::backlog]\n"
    ),
    "github-or-group": (
        "platform: github\n"
        "github:\n"
        "  repo: org/repo\n"
        "  labels:\n"
        "    default:\n"
        "      - [type::feature, type::bug]\n"
    ),
    "no-platform-key": ("gitlab:\n  default_group: group/project\n  labels:\n    default: []\n"),
    "legacy-allowed-labels": (
        "labels:\n"
        "  default: [development-status::backlog]\n"
        "  allowed_labels: [type::feature]\n"
        "gitlab:\n"
        "  default_group: group/project\n"
    ),
}


def _accessor_snapshot(config: Config) -> Dict[str, Any]:
    """Every existing accessor's value, for a before/after comparison."""
    return {
        "platform": config.platform,
        "default_group": config.get_default_group(),
        "default_labels": config.get_default_labels(),
        "required_label_groups": config.get_required_label_groups(),
        "default_epic_labels": config.get_default_epic_labels(),
        "allowed_labels": config.get_allowed_labels(),
        "required_sections": config.get_required_sections(),
        "required_epic_sections": config.get_required_epic_sections(),
        "required_mr_sections": config.get_required_mr_sections(),
        "required_issue_fields": config.get_required_issue_fields(),
        "required_epic_fields": config.get_required_epic_fields(),
        "required_mr_fields": config.get_required_mr_fields(),
        "default_mr_reviewers": config.get_default_mr_reviewers(),
        "planning_sync": config.planning_sync,
    }


def _load(path: Path, body: str) -> Config:
    path.write_text(body)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return Config(path)


# The pre-change value of every accessor, as literal data rather than as a
# second reading of the same live code: a before/after comparison cancels out
# any regression both sides share, which is exactly the class NFR-4 names.
_BASELINE_ACCESSORS: Dict[str, Any] = {
    "platform": "gitlab",
    "default_group": "group/project",
    "default_labels": ["development-status::backlog"],
    "required_label_groups": [],
    "default_epic_labels": [],
    "allowed_labels": None,
    "required_sections": [],
    "required_epic_sections": ["Description"],
    "required_mr_sections": ["Summary", "Implementation Details", "How It Was Tested"],
    "required_issue_fields": ["weight"],
    "required_epic_fields": [],
    "required_mr_fields": [],
    "default_mr_reviewers": [],
    "planning_sync": {},
}

_SHAPE_ACCESSOR_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "gitlab-flat-labels": {},
    "gitlab-or-group": {"required_label_groups": [["type::feature", "type::bug"]]},
    "gitlab-allowed-present": {"allowed_labels": ["type::feature", "type::bug"]},
    "gitlab-allowed-empty": {"allowed_labels": []},
    "gitlab-default-epic": {"default_epic_labels": ["type::epic"]},
    "gitlab-with-planning-sync": {"planning_sync": {"gdrive_base": "~/GoogleDrive"}},
    "gitlab-with-templates": {
        "required_sections": ["Description", "Acceptance Criteria"],
        "required_mr_sections": ["Summary"],
        "required_mr_fields": ["reviewers", "labels"],
        "default_mr_reviewers": ["alice", "bob"],
    },
    "github-repo": {
        "platform": "github",
        "default_group": None,
        "required_issue_fields": [],
    },
    "github-or-group": {
        "platform": "github",
        "default_group": None,
        "default_labels": [],
        "required_label_groups": [["type::feature", "type::bug"]],
        "required_issue_fields": [],
    },
    "no-platform-key": {"default_labels": []},
    "legacy-allowed-labels": {
        "allowed_labels": ["type::feature"],
        "required_sections": ["Description"],
    },
}


class TestManagedConfigShapeCompatibility:
    """NFR-4: the existing projctl.yaml shapes load exactly as they did before."""

    def test_every_managed_shape_carries_a_pinned_accessor_snapshot(self) -> None:
        assert set(_SHAPE_ACCESSOR_OVERRIDES) == set(_MANAGED_CONFIG_SHAPES)

    @pytest.mark.parametrize("shape", sorted(_MANAGED_CONFIG_SHAPES))
    def test_every_accessor_returns_its_pinned_pre_change_value(
        self, shape: str, temp_dir: Path
    ) -> None:
        config = _load(temp_dir / "projctl.yaml", _MANAGED_CONFIG_SHAPES[shape])
        expected = {**_BASELINE_ACCESSORS, **_SHAPE_ACCESSOR_OVERRIDES[shape]}
        assert _accessor_snapshot(config) == expected

    @pytest.mark.parametrize("shape", sorted(_MANAGED_CONFIG_SHAPES))
    def test_adding_a_search_section_changes_no_existing_accessor(
        self, shape: str, temp_dir: Path
    ) -> None:
        body = _MANAGED_CONFIG_SHAPES[shape]
        before = _accessor_snapshot(_load(temp_dir / "before.yaml", body))
        after_config = _load(
            temp_dir / "after.yaml", body + "search:\n  docs_path: documentation\n"
        )

        assert _accessor_snapshot(after_config) == before
        assert after_config.get_search_config().docs_path == "documentation"

    @pytest.mark.parametrize("shape", sorted(_MANAGED_CONFIG_SHAPES))
    def test_a_shape_carrying_no_search_section_yields_the_documented_defaults(
        self, shape: str, temp_dir: Path
    ) -> None:
        config = _load(temp_dir / "projctl.yaml", _MANAGED_CONFIG_SHAPES[shape])
        search_config = config.get_search_config()
        assert search_config.docs_path == "docs"
        assert search_config.docs_path_configured is False
        assert search_config.related == []

    def test_a_legacy_shape_carrying_search_transforms_exactly_as_it_did_before(
        self, temp_dir: Path
    ) -> None:
        legacy = _MANAGED_CONFIG_SHAPES["legacy-allowed-labels"]
        without_search = _load(temp_dir / "without.yaml", legacy)
        with_search = _load(temp_dir / "with.yaml", legacy + "search:\n  docs_path: docs\n")

        # The transform whitelists gitlab/common/planning_sync, so search:
        # must survive on the raw mapping without appearing on the transformed
        # one — and the transformed one must be untouched by its presence.
        assert with_search.config_data == without_search.config_data
        assert "search" not in with_search.config_data
        assert with_search.raw_config_data["search"] == {"docs_path": "docs"}
