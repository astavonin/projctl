"""Regression tests for CLI command parity.

Ensures CLI commands produce identical output to old version.
"""

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ci_platform_manager.cli import main


class TestCLICommandAvailability:
    """Test that all CLI commands are available."""

    def test_help_command_works(self) -> None:
        """Help command should work."""
        result = subprocess.run(
            ["python", "-m", "ci_platform_manager", "--help"], capture_output=True, text=True
        )

        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()

    def test_create_command_exists(self) -> None:
        """Create command should be available."""
        result = subprocess.run(
            ["python", "-m", "ci_platform_manager", "create", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "create" in result.stdout.lower()

    def test_load_command_exists(self) -> None:
        """Load command should be available."""
        result = subprocess.run(
            ["python", "-m", "ci_platform_manager", "load", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

    def test_search_command_exists(self) -> None:
        """Search command should be available."""
        result = subprocess.run(
            ["python", "-m", "ci_platform_manager", "search", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

    def test_comment_command_exists(self) -> None:
        """Comment command should be available."""
        result = subprocess.run(
            ["python", "-m", "ci_platform_manager", "comment", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

    def test_create_mr_command_exists(self) -> None:
        """Create-mr command should be available."""
        result = subprocess.run(
            ["python", "-m", "ci_platform_manager", "create-mr", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0


class TestLegacyWrapperParity:
    """Test that legacy wrapper still works."""

    def test_old_script_wrapper_works(self) -> None:
        """Legacy glab_tasks_management.py should still work."""
        result = subprocess.run(
            ["python", "glab-management/glab_tasks_management.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()

    def test_old_script_shows_deprecation(self) -> None:
        """Legacy script should show deprecation warning."""
        result = subprocess.run(
            ["python", "glab-management/glab_tasks_management.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        # Deprecation warning should be in stderr
        assert "deprecat" in result.stderr.lower() or "DEPRECAT" in result.stderr


class TestCLIOutputFormat:
    """Test CLI output format consistency."""

    @patch("subprocess.run")
    def test_load_issue_output_format(
        self, mock_run: Mock, new_config_path: Path, mock_glab_issue_view: str, capsys, monkeypatch
    ) -> None:
        """Load command should produce markdown format."""
        # Mock glab command
        mock_run.return_value = Mock(stdout=mock_glab_issue_view, stderr="", returncode=0)

        # Run CLI command
        monkeypatch.setattr(
            "sys.argv",
            ["ci-platform-manager", "--config", str(new_config_path), "load", "issue", "#1"],
        )

        try:
            main()
        except SystemExit:
            pass

        captured = capsys.readouterr()

        # Check markdown format
        assert "State:" in captured.out or "Labels:" in captured.out
        assert "Description:" in captured.out or "#" in captured.out

    @patch("subprocess.run")
    def test_search_output_format(
        self, mock_run: Mock, new_config_path: Path, capsys, monkeypatch
    ) -> None:
        """Search command should produce text format."""
        import json

        search_results = [
            {
                "iid": 1,
                "title": "Test Issue",
                "state": "opened",
                "labels": ["type::feature"],
                "web_url": "https://gitlab.example.com/test/project/-/issues/1",
            }
        ]

        mock_run.return_value = Mock(stdout=json.dumps(search_results), returncode=0)

        monkeypatch.setattr(
            "sys.argv",
            ["ci-platform-manager", "--config", str(new_config_path), "search", "issues", "test"],
        )

        try:
            main()
        except SystemExit:
            pass

        captured = capsys.readouterr()

        # Check text format output
        assert "Test Issue" in captured.out


class TestCLIOptions:
    """Test CLI option compatibility."""

    def test_config_option_works(self, new_config_path: Path) -> None:
        """--config option should work."""
        result = subprocess.run(
            ["python", "-m", "ci_platform_manager", "--config", str(new_config_path), "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

    def test_dry_run_option_works(self, new_config_path: Path) -> None:
        """--dry-run option should work for create command."""
        result = subprocess.run(
            [
                "python",
                "-m",
                "ci_platform_manager",
                "--config",
                str(new_config_path),
                "create",
                "--dry-run",
                "--help",
            ],
            capture_output=True,
            text=True,
        )

        # Should not fail (help overrides dry-run)
        assert result.returncode == 0


class TestReferenceFormats:
    """Test that all reference formats are supported."""

    @patch("subprocess.run")
    def test_issue_number_reference(
        self, mock_run: Mock, new_config_path: Path, mock_glab_issue_view: str, monkeypatch
    ) -> None:
        """#123 reference format should work."""
        mock_run.return_value = Mock(stdout=mock_glab_issue_view, returncode=0)

        monkeypatch.setattr(
            "sys.argv",
            ["ci-platform-manager", "--config", str(new_config_path), "load", "issue", "#123"],
        )

        try:
            main()
        except SystemExit:
            pass

        # Should have called glab
        mock_run.assert_called()

    @patch("subprocess.run")
    def test_epic_reference(self, mock_run: Mock, new_config_path: Path, monkeypatch) -> None:
        """&21 reference format should work."""
        # Mock both epic and issues API calls
        mock_run.side_effect = [
            Mock(stdout='{"iid": 21, "title": "Test Epic"}', returncode=0),
            Mock(stdout="[]", returncode=0),
        ]

        monkeypatch.setattr(
            "sys.argv",
            ["ci-platform-manager", "--config", str(new_config_path), "load", "epic", "&21"],
        )

        try:
            main()
        except SystemExit:
            pass

        # Should have called glab API
        assert mock_run.call_count >= 1

    @patch("subprocess.run")
    def test_milestone_reference(self, mock_run: Mock, new_config_path: Path, monkeypatch) -> None:
        """%123 reference format should work."""
        import json

        milestone_data = {"id": 123, "iid": 1, "title": "v1.0", "state": "active"}

        mock_run.side_effect = [
            Mock(stdout=json.dumps(milestone_data), returncode=0),
            Mock(stdout="[]", returncode=0),
        ]

        monkeypatch.setattr(
            "sys.argv",
            ["ci-platform-manager", "--config", str(new_config_path), "load", "milestone", "%123"],
        )

        try:
            main()
        except SystemExit:
            pass

        assert mock_run.call_count >= 1

    @patch("subprocess.run")
    def test_mr_reference(
        self, mock_run: Mock, new_config_path: Path, mock_glab_mr_view: str, monkeypatch
    ) -> None:
        """!134 reference format should work."""
        mock_run.return_value = Mock(stdout=mock_glab_mr_view, returncode=0)

        monkeypatch.setattr(
            "sys.argv",
            ["ci-platform-manager", "--config", str(new_config_path), "load", "mr", "!134"],
        )

        try:
            main()
        except SystemExit:
            pass

        mock_run.assert_called()

    @patch("subprocess.run")
    def test_url_reference(
        self, mock_run: Mock, new_config_path: Path, mock_glab_issue_view: str, monkeypatch
    ) -> None:
        """Full URL reference should work."""
        mock_run.return_value = Mock(stdout=mock_glab_issue_view, returncode=0)

        monkeypatch.setattr(
            "sys.argv",
            [
                "ci-platform-manager",
                "--config",
                str(new_config_path),
                "load",
                "issue",
                "https://gitlab.example.com/test/project/-/issues/123",
            ],
        )

        try:
            main()
        except SystemExit:
            pass

        mock_run.assert_called()


class TestErrorHandling:
    """Test error handling parity."""

    def test_missing_config_error(self, temp_dir: Path, monkeypatch) -> None:
        """Missing config should show helpful error."""
        monkeypatch.chdir(temp_dir)

        result = subprocess.run(
            ["python", "-m", "ci_platform_manager", "load", "issue", "#123"],
            capture_output=True,
            text=True,
        )

        # Should fail with error about missing config
        assert result.returncode != 0
        assert "config" in result.stderr.lower() or "not found" in result.stderr.lower()

    @patch("subprocess.run")
    def test_glab_not_found_error(self, mock_run: Mock, new_config_path: Path, monkeypatch) -> None:
        """Missing glab CLI should show helpful error."""
        mock_run.side_effect = FileNotFoundError()

        monkeypatch.setattr(
            "sys.argv",
            ["ci-platform-manager", "--config", str(new_config_path), "load", "issue", "#123"],
        )

        with pytest.raises(SystemExit):
            main()
