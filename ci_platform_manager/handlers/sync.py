"""Planning folder sync handler."""

import logging
import os
import subprocess
from pathlib import Path

from ..config import Config
from ..exceptions import PlatformError


logger = logging.getLogger(__name__)


class PlanningSyncHandler:
    """Handles synchronization of planning folders to/from Google Drive.

    Auto-detects current repository name and planning folder location.
    Uses rsync for efficient file synchronization.
    """

    def __init__(self, config: Config, dry_run: bool = False) -> None:
        """Initialize the sync handler.

        Args:
            config: Configuration object with planning_sync settings.
            dry_run: If True, only preview operations without executing them.

        Raises:
            PlatformError: If not in a git repository or planning folder doesn't exist.
        """
        self.config = config
        self.dry_run = dry_run

        # Auto-detect repository and planning folder
        self.repo_name = self._detect_repo_name()
        self.repo_root = self._get_repo_root()
        self.planning_path = self._get_planning_path()

        # Get Google Drive base from config
        planning_sync = getattr(config, 'planning_sync', {})
        gdrive_base = planning_sync.get('gdrive_base')

        if not gdrive_base:
            raise PlatformError(
                "Google Drive path not configured.\n\n"
                "Add to your config file:\n"
                "planning_sync:\n"
                "  gdrive_base: ~/GoogleDrive"
            )

        # Expand user path (~/GoogleDrive -> /home/user/GoogleDrive)
        self.gdrive_base = Path(os.path.expanduser(gdrive_base))
        self.gdrive_planning_base = self.gdrive_base / 'backup' / 'planning'
        self.gdrive_repo_path = self.gdrive_planning_base / self.repo_name

    def _get_repo_root(self) -> Path:
        """Get the git repository root directory.

        Returns:
            Path to repository root.

        Raises:
            PlatformError: If not in a git repository.
        """
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--show-toplevel'],
                capture_output=True,
                text=True,
                check=True
            )
            return Path(result.stdout.strip())
        except subprocess.CalledProcessError as err:
            raise PlatformError(
                "Not in a git repository. Planning sync requires git."
            ) from err

    def _detect_repo_name(self) -> str:
        """Detect current git repository name.

        Uses the repository directory name as the identifier.

        Returns:
            Repository name.

        Raises:
            PlatformError: If not in a git repository.
        """
        repo_root = self._get_repo_root()
        return repo_root.name

    def _get_planning_path(self) -> Path:
        """Get planning folder path (./planning/ from repo root).

        Returns:
            Path to planning folder.

        Raises:
            PlatformError: If planning folder doesn't exist.
        """
        planning_path = self.repo_root / 'planning'

        if not planning_path.exists():
            raise PlatformError(
                f"Planning folder not found: {planning_path}\n\n"
                f"Expected planning folder in repository root."
            )

        if not planning_path.is_dir():
            raise PlatformError(
                f"Planning path exists but is not a directory: {planning_path}"
            )

        return planning_path

    def _verify_rsync_available(self) -> None:
        """Verify rsync command is available.

        Raises:
            PlatformError: If rsync is not installed.
        """
        try:
            subprocess.run(
                ['rsync', '--version'],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as err:
            raise PlatformError(
                "rsync is not installed or not available in PATH.\n\n"
                "Install rsync:\n"
                "  Ubuntu/Debian: sudo apt install rsync\n"
                "  macOS: brew install rsync (or use built-in version)"
            ) from err

    def _run_rsync(self, source: Path, target: Path, description: str) -> None:
        """Run rsync to synchronize directories.

        Args:
            source: Source directory path.
            target: Target directory path.
            description: Human-readable description of the operation.

        Raises:
            PlatformError: If rsync fails.
        """
        # Ensure source and target end with trailing slash for rsync
        source_str = f"{source}/"
        target_str = f"{target}/"

        # Build rsync command
        rsync_cmd = [
            'rsync',
            '-av',  # archive mode, verbose
            '--delete',  # delete files in target that don't exist in source
            '--exclude=*.swp',  # exclude vim swap files
            '--exclude=*~',  # exclude backup files
            '--exclude=.DS_Store',  # exclude macOS metadata
        ]

        if self.dry_run:
            rsync_cmd.append('--dry-run')

        rsync_cmd.extend([source_str, target_str])

        logger.debug("Executing rsync: %s", ' '.join(rsync_cmd))

        if self.dry_run:
            print(f"\n[DRY RUN] {description}")
            print(f"  Source: {source}")
            print(f"  Target: {target}")
            print(f"  Command: {' '.join(rsync_cmd)}\n")

        try:
            result = subprocess.run(
                rsync_cmd,
                capture_output=True,
                text=True,
                check=True
            )

            if self.dry_run:
                print("Preview of changes:")
                print(result.stdout)
            else:
                logger.debug("rsync output: %s", result.stdout)
                logger.info("✓ %s", description)

        except subprocess.CalledProcessError as err:
            raise PlatformError(
                f"rsync failed: {err.stderr}\n\n"
                f"Command: {' '.join(rsync_cmd)}"
            ) from err

    def push(self) -> None:
        """Push local planning folder to Google Drive.

        Syncs ./planning/ → Google Drive backup location.

        Raises:
            PlatformError: If sync fails.
        """
        self._verify_rsync_available()

        # Verify Google Drive base exists
        if not self.gdrive_base.exists():
            raise PlatformError(
                f"Google Drive not found: {self.gdrive_base}\n\n"
                f"Verify Google Drive is mounted and path is correct in config."
            )

        # Create target directory if it doesn't exist
        if not self.dry_run:
            self.gdrive_planning_base.mkdir(parents=True, exist_ok=True)

        description = f"Push {self.repo_name} planning → Google Drive"
        self._run_rsync(
            source=self.planning_path,
            target=self.gdrive_repo_path,
            description=description
        )

        if not self.dry_run:
            print(f"✓ Pushed {self.repo_name} planning to Google Drive")
            print(f"  Local:  {self.planning_path}")
            print(f"  Remote: {self.gdrive_repo_path}")

    def pull(self) -> None:
        """Pull planning folder from Google Drive to local.

        Syncs Google Drive backup → ./planning/

        Raises:
            PlatformError: If sync fails or Google Drive folder doesn't exist.
        """
        self._verify_rsync_available()

        # Verify Google Drive folder exists
        if not self.gdrive_repo_path.exists():
            if self.gdrive_planning_base.exists():
                available = "\n".join(
                    f"  - {p.name}"
                    for p in self.gdrive_planning_base.iterdir()
                    if p.is_dir()
                )
                error_msg = (
                    f"Planning folder not found in Google Drive: {self.gdrive_repo_path}\n\n"
                    f"Available repositories in Google Drive:\n{available}"
                )
            else:
                error_msg = (
                    f"Planning folder not found in Google Drive: {self.gdrive_repo_path}\n\n"
                    f"Google Drive planning backup folder doesn't exist: "
                    f"{self.gdrive_planning_base}"
                )
            raise PlatformError(error_msg)

        # Create local planning folder if it doesn't exist
        if not self.dry_run and not self.planning_path.exists():
            self.planning_path.mkdir(parents=True, exist_ok=True)

        description = f"Pull {self.repo_name} planning ← Google Drive"
        self._run_rsync(
            source=self.gdrive_repo_path,
            target=self.planning_path,
            description=description
        )

        if not self.dry_run:
            print(f"✓ Pulled {self.repo_name} planning from Google Drive")
            print(f"  Remote: {self.gdrive_repo_path}")
            print(f"  Local:  {self.planning_path}")
