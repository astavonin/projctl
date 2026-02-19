"""Configuration management for CI Platform Manager."""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required. Install with: pip install PyYAML") from exc

from .utils.config_migration import transform_issue_template


logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Configuration-related errors."""


class Config:
    """Multi-platform configuration manager with legacy format support.

    Loads configuration from YAML file with automatic legacy format detection.
    Configuration can be overridden by command-line arguments.

    Config file resolution:
    1. If --config is specified: use that path (fail if not found)
    2. ./glab_config.yaml (project-local legacy, PRIORITY)
    3. ./config.yaml (project-local new format)
    4. ~/.config/ci_platform_manager/config.yaml (user config new)
    5. ~/.config/glab_config.yaml (user config legacy)
    """

    def __init__(self, config_path: Optional[Path] = None, platform: Optional[str] = None) -> None:
        """Initialize configuration.

        Args:
            config_path: Path to config file. If None, uses default search order.
            platform: Platform override (gitlab, github). If None, uses config or defaults to gitlab.

        Raises:
            ConfigurationError: If config file doesn't exist or is invalid.
            FileNotFoundError: If config file not found.
        """
        self.config_data: Dict[str, Any] = {}
        self.loaded_config_path: Optional[Path] = None
        self.planning_sync: Dict[str, Any] = {}

        self.config_data = self._load_config_with_legacy_support(config_path)
        self.platform = platform or self.config_data.get('platform', 'gitlab')
        self._load_planning_sync()

    def _load_config_with_legacy_support(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load config with automatic legacy format detection.

        Search order (preserves backward compatibility):
        1. Explicit --config path if provided
        2. ./glab_config.yaml (project-local, current behavior)
        3. ./config.yaml (project-local, new format)
        4. ~/.config/ci_platform_manager/config.yaml (user config, new)
        5. ~/.config/glab_config.yaml (user config, legacy)

        Args:
            config_path: Explicit config path or None for auto-search.

        Returns:
            Dictionary containing configuration data.

        Raises:
            FileNotFoundError: If no config file found in search order.
        """
        if config_path is not None:
            # Explicit config path specified via --config
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Config file not found: {config_path}\n\n"
                    f"To fix this:\n"
                    f"  - Verify the path specified with --config is correct\n"
                    f"  - OR copy the example config to your desired location"
                )
            return self._load_config_file(config_path)

        # Search order for backward compatibility
        search_paths = [
            Path.cwd() / 'glab_config.yaml',  # Project-local legacy (PRIORITY)
            Path.cwd() / 'config.yaml',        # Project-local new format
            Path.home() / '.config' / 'ci_platform_manager' / 'config.yaml',  # User config new
            Path.home() / '.config' / 'glab_config.yaml',  # User config legacy
        ]

        for candidate in search_paths:
            if candidate.exists():
                if 'glab_config.yaml' in str(candidate):
                    warnings.warn(
                        f"Using legacy config location {candidate}. "
                        f"Consider renaming to config.yaml",
                        DeprecationWarning,
                        stacklevel=2
                    )
                return self._load_config_file(candidate)

        # No config found
        raise FileNotFoundError(
            "No config file found. Searched:\n" +
            "\n".join(f"  - {p}" for p in search_paths)
        )

    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from a file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            Dictionary containing configuration data.

        Raises:
            yaml.YAMLError: If YAML parsing fails.
            IOError: If file cannot be read.
        """
        with open(config_path, 'r', encoding='utf-8') as config_file:
            config = yaml.safe_load(config_file) or {}

        self.loaded_config_path = config_path
        logger.info("Loaded configuration from: %s", config_path)
        # Detect and transform old format
        is_old_format = 'labels' in config and 'platform' not in config

        if is_old_format:
            warnings.warn(
                "Config uses deprecated format. Run migration script to update.",
                DeprecationWarning,
                stacklevel=2
            )
            config = self._transform_legacy_config(config)

        return config

    def _transform_legacy_config(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform old config format to new format in-memory.

        Args:
            old_config: Configuration in legacy format.

        Returns:
            Configuration in new format.
        """
        old_template = old_config.get('issue_template', {})
        new_template = transform_issue_template(old_template)

        # Extract labels section
        labels_config = old_config.get('labels', {})

        # Build labels dictionary, making default_epic optional
        labels = {
            'default': labels_config.get('default', []),
            'default_epic': labels_config.get('default_epic', []),  # Optional
        }

        # Handle both 'allowed' and 'allowed_labels' for backward compatibility
        allowed = labels_config.get('allowed', labels_config.get('allowed_labels', []))
        if allowed:
            labels['allowed'] = allowed

        new_config = {
            'platform': 'gitlab',
            'gitlab': {
                'default_group': old_config.get('gitlab', {}).get('default_group', ''),
                'labels': labels
            },
            'common': {
                'issue_template': new_template
            }
        }

        # Preserve planning_sync section if present
        if 'planning_sync' in old_config:
            new_config['planning_sync'] = old_config['planning_sync']

        return new_config

    def get_platform_config(self, platform: str) -> Dict[str, Any]:
        """Get platform-specific configuration.

        Args:
            platform: Platform name (gitlab, github).

        Returns:
            Platform-specific configuration dictionary.
        """
        return self.config_data.get(platform, {})  # type: ignore[no-any-return]

    def get_common_config(self) -> Dict[str, Any]:
        """Get common (platform-agnostic) configuration.

        Returns:
            Common configuration dictionary.
        """
        return self.config_data.get('common', {})  # type: ignore[no-any-return]

    def get_required_sections(self) -> List[str]:
        """Get required issue template sections.

        Returns:
            List of required section names.
        """
        # New format
        new_sections = self.get_common_config().get('issue_template', {}).get('required_sections', [])
        if new_sections:
            return new_sections  # type: ignore[no-any-return]

        # Legacy format support (if config wasn't transformed)
        legacy_sections = self.config_data.get('issue_template', {}).get('sections', [])
        return [s['name'] for s in legacy_sections if s.get('required', False)]

    def get_default_group(self) -> Optional[str]:
        """Get the default GitLab group path for epic operations.

        Returns:
            Default group path or None.
        """
        return self.get_platform_config('gitlab').get('default_group')

    def get_default_labels(self) -> List[str]:
        """Get the default labels to apply to issues.

        Returns:
            List of default label names.
        """
        # Try new format first
        platform_labels = self.get_platform_config(self.platform).get('labels', {}).get('default', [])
        if platform_labels:
            return platform_labels  # type: ignore[no-any-return]

        # Fallback to legacy format
        return self.config_data.get('labels', {}).get('default', [])  # type: ignore[no-any-return]

    def get_default_epic_labels(self) -> List[str]:
        """Get the default labels to apply to epics.

        Returns:
            List of default epic label names.
        """
        # Try new format first
        platform_labels = self.get_platform_config(self.platform).get('labels', {}).get('default_epic', [])
        if platform_labels:
            return platform_labels  # type: ignore[no-any-return]

        # Fallback to legacy format
        return self.config_data.get('labels', {}).get('default_epic', [])  # type: ignore[no-any-return]

    def get_allowed_labels(self) -> Optional[List[str]]:
        """Get the allowed labels list for validation.

        Returns:
            List of allowed label names, or None if not configured.
        """
        # Try new format first
        platform_labels = self.get_platform_config(self.platform).get('labels', {}).get('allowed', [])
        if platform_labels:
            return platform_labels  # type: ignore[no-any-return]

        # Fallback to legacy format
        allowed = self.config_data.get('labels', {}).get('allowed_labels', [])
        return allowed if allowed else None  # type: ignore[no-any-return]

    def _load_planning_sync(self) -> None:
        """Load planning sync configuration from config file.

        Loads the 'planning_sync' section if present, which contains:
        - gdrive_base: Base path to Google Drive mount point
        """
        if 'planning_sync' in self.config_data:
            self.planning_sync = self.config_data['planning_sync']
            logger.debug("Loaded planning_sync config: %s", self.planning_sync)
        else:
            self.planning_sync = {}
            logger.debug("No planning_sync section in config")
