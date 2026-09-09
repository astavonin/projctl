"""Tests for projctl.handlers.updater module."""

# Tests intentionally access protected members to unit-test internal helpers.
# pylint: disable=protected-access

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from projctl.cli import main as cli_main
from projctl.config import Config
from projctl.exceptions import PlatformError
from projctl.handlers.updater import TicketUpdater

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_run(stdout: str = "{}") -> Mock:
    """Return a subprocess.run mock that returns the given stdout."""
    mock = Mock()
    mock.return_value = Mock(stdout=stdout, stderr="", returncode=0)
    return mock


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestTicketUpdaterInit:
    """Test TicketUpdater initialisation."""

    def test_init_defaults(self, new_config_path: Path) -> None:
        """Updater stores config and defaults dry_run to False."""
        config = Config(new_config_path)
        updater = TicketUpdater(config)

        assert updater.config is config
        assert updater.dry_run is False

    def test_init_dry_run(self, new_config_path: Path) -> None:
        """Updater stores dry_run=True when requested."""
        config = Config(new_config_path)
        updater = TicketUpdater(config, dry_run=True)

        assert updater.dry_run is True


# ---------------------------------------------------------------------------
# update_issue
# ---------------------------------------------------------------------------


class TestUpdateIssue:
    """Tests for TicketUpdater.update_issue."""

    @patch("subprocess.run")
    def test_update_issue_title(self, mock_run: Mock, new_config_path: Path) -> None:
        """update_issue sends PUT to the correct endpoint with title field."""
        mock_run.return_value = Mock(
            stdout='{"iid": 231, "title": "New title"}', stderr="", returncode=0
        )
        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_issue("231", title="New title")

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "glab"
        assert "api" in args
        assert "-X" in args
        assert "PUT" in args
        # Endpoint should reference issues/231
        endpoint = [a for a in args if "issues/231" in str(a)]
        assert endpoint, f"Expected issues/231 in args: {args}"
        # Field should be present
        assert "-f" in args
        title_field = [args[i + 1] for i, a in enumerate(args) if a == "-f"]
        assert any("title=New title" in f for f in title_field)

    @patch("subprocess.run")
    def test_update_issue_state_event(self, mock_run: Mock, new_config_path: Path) -> None:
        """update_issue passes state_event=close correctly."""
        mock_run.return_value = Mock(
            stdout='{"iid": 231, "title": "Some issue"}', stderr="", returncode=0
        )
        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_issue("231", state_event="close")

        args = mock_run.call_args[0][0]
        fields = [args[i + 1] for i, a in enumerate(args) if a == "-f"]
        assert any("state_event=close" in f for f in fields)

    @patch("subprocess.run")
    def test_update_issue_due_date(self, mock_run: Mock, new_config_path: Path) -> None:
        """update_issue sends due_date, which GitLab supports on issues as well as milestones."""
        mock_run.return_value = Mock(
            stdout='{"iid": 478, "due_date": "2026-08-11"}', stderr="", returncode=0
        )
        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_issue("478", due_date="2026-08-11")

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        fields = [args[i + 1] for i, a in enumerate(args) if a == "-f"]
        assert any("due_date=2026-08-11" in f for f in fields)

    @patch("subprocess.run")
    def test_update_issue_due_date_alone_triggers_put(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """A due date with no other field still counts as an update.

        due_date must appear in has_put_fields; omitting it there makes a
        due-date-only call silently no-op instead of issuing the PUT.
        """
        mock_run.return_value = Mock(stdout='{"iid": 478}', stderr="", returncode=0)
        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_issue("478", due_date="2026-08-11")

        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_update_issue_label_merge(self, mock_run: Mock, new_config_path: Path) -> None:
        """update_issue fetches current labels and merges add/remove correctly."""
        # First call: GET current labels; second call: PUT update
        get_response = Mock(
            stdout='{"iid": 231, "title": "T", "labels": ["type::feature", "keep"]}',
            stderr="",
            returncode=0,
        )
        put_response = Mock(stdout='{"iid": 231, "title": "T"}', stderr="", returncode=0)
        mock_run.side_effect = [get_response, put_response]

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_issue("231", labels_add=["type::bug"], labels_remove=["type::feature"])

        assert mock_run.call_count == 2
        # Second call (PUT) should contain the merged label set.
        put_args = mock_run.call_args_list[1][0][0]
        fields = [put_args[i + 1] for i, a in enumerate(put_args) if a == "-f"]
        label_fields = [f for f in fields if f.startswith("labels=")]
        assert label_fields, "Expected labels field in PUT call"
        label_value = label_fields[0][len("labels=") :]
        label_set = set(label_value.split(","))
        assert "type::bug" in label_set
        assert "keep" in label_set
        assert "type::feature" not in label_set

    @patch("subprocess.run")
    def test_update_issue_dry_run(self, mock_run: Mock, new_config_path: Path, capsys) -> None:
        """Dry run prints intent without executing any glab command."""
        config = Config(new_config_path)
        updater = TicketUpdater(config, dry_run=True)

        updater.update_issue("231", title="Preview")

        mock_run.assert_not_called()
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "issues/231" in captured.out

    @patch("subprocess.run")
    def test_update_issue_dry_run_with_labels_no_api_call(
        self, mock_run: Mock, new_config_path: Path, capsys
    ) -> None:
        """Dry run with labels must not make any live API call."""
        config = Config(new_config_path)
        updater = TicketUpdater(config, dry_run=True)

        updater.update_issue("231", labels_add=["type::bug"], labels_remove=["type::feature"])

        mock_run.assert_not_called()
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "add:" in captured.out

    @patch("subprocess.run")
    def test_update_issue_command_failure(self, mock_run: Mock, new_config_path: Path) -> None:
        """PlatformError is raised when the glab command fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, ["glab", "api"], stderr="error")
        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with pytest.raises(PlatformError):
            updater.update_issue("231", title="Fail")

    def test_glab_not_found_raises_platform_error(self, new_config_path: Path) -> None:
        """PlatformError is raised when the glab binary is missing.

        Delegates through update_issue → loader._run_glab_command, so the
        FileNotFoundError surfaces as a PlatformError from the loader.
        """
        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(PlatformError, match="glab command not found"):
                updater.update_issue("231", title="Fail")

    @patch("subprocess.run")
    def test_update_issue_assignee_resolves_user_id(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """update_issue resolves username to numeric ID before sending PUT."""
        # First call: users API; second call: PUT
        users_response = Mock(stdout='[{"id": 42, "username": "alice"}]', stderr="", returncode=0)
        put_response = Mock(stdout='{"iid": 231, "title": "T"}', stderr="", returncode=0)
        mock_run.side_effect = [users_response, put_response]

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_issue("231", assignee="alice")

        # PUT call must use the numeric ID, not the username string.
        put_args = mock_run.call_args_list[1][0][0]
        fields = [put_args[i + 1] for i, a in enumerate(put_args) if a == "-f"]
        assert any("assignee_ids=42" in f for f in fields)

    @patch("subprocess.run")
    def test_update_issue_milestone_resolves_id(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """update_issue resolves milestone title to numeric ID before sending PUT."""
        # First call: milestones API; second call: PUT
        milestones_response = Mock(
            stdout='[{"id": 99, "iid": 10, "title": "v2.0"}]', stderr="", returncode=0
        )
        put_response = Mock(stdout='{"iid": 231, "title": "T"}', stderr="", returncode=0)
        mock_run.side_effect = [milestones_response, put_response]

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_issue("231", milestone="v2.0")

        put_args = mock_run.call_args_list[1][0][0]
        fields = [put_args[i + 1] for i, a in enumerate(put_args) if a == "-f"]
        assert any("milestone_id=99" in f for f in fields)


    def test_update_issue_invalid_label_raises(self, new_config_path: Path) -> None:
        """labels_add containing a label not in the allowed list raises ValueError."""
        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with pytest.raises(ValueError, match="Unknown labels"):
            updater.update_issue("231", labels_add=["not-allowed::label"])

    def test_update_issue_invalid_label_raises_in_dry_run(self, new_config_path: Path) -> None:
        """Label validation fires before dry-run output — bad labels fail even with dry_run=True."""
        config = Config(new_config_path)
        updater = TicketUpdater(config, dry_run=True)

        with pytest.raises(ValueError, match="Unknown labels"):
            updater.update_issue("231", labels_add=["not-allowed::label"])

    @patch("subprocess.run")
    def test_update_issue_valid_label_does_not_raise(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """labels_add containing only allowed labels proceeds without error."""
        mock_run.side_effect = [
            Mock(stdout='{"iid": 231, "labels": []}', stderr="", returncode=0),
            Mock(stdout='{"iid": 231, "title": "T"}', stderr="", returncode=0),
        ]
        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_issue("231", labels_add=["type::bug"])  # in allowed list — must not raise


# ---------------------------------------------------------------------------
# update_mr
# ---------------------------------------------------------------------------


class TestUpdateMr:
    """Tests for TicketUpdater.update_mr."""

    @patch("subprocess.run")
    def test_update_mr_title_and_description(self, mock_run: Mock, new_config_path: Path) -> None:
        """update_mr sends PUT to merge_requests endpoint with title and description."""
        mock_run.return_value = Mock(
            stdout='{"iid": 144, "title": "New MR title"}', stderr="", returncode=0
        )
        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_mr("144", title="New MR title", description="New desc")

        args = mock_run.call_args[0][0]
        assert "merge_requests/144" in " ".join(args)
        fields = [args[i + 1] for i, a in enumerate(args) if a == "-f"]
        assert any("title=New MR title" in f for f in fields)
        assert any("description=New desc" in f for f in fields)

    @patch("subprocess.run")
    def test_update_mr_state_event(self, mock_run: Mock, new_config_path: Path) -> None:
        """update_mr passes state_event correctly."""
        mock_run.return_value = Mock(stdout='{"iid": 144, "title": "T"}', stderr="", returncode=0)
        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_mr("144", state_event="reopen")

        args = mock_run.call_args[0][0]
        fields = [args[i + 1] for i, a in enumerate(args) if a == "-f"]
        assert any("state_event=reopen" in f for f in fields)

    @patch("subprocess.run")
    def test_update_mr_label_merge(self, mock_run: Mock, new_config_path: Path) -> None:
        """update_mr fetches and merges labels correctly."""
        get_response = Mock(
            stdout='{"iid": 144, "title": "T", "labels": ["type::feature"]}',
            stderr="",
            returncode=0,
        )
        put_response = Mock(stdout='{"iid": 144, "title": "T"}', stderr="", returncode=0)
        mock_run.side_effect = [get_response, put_response]

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_mr("144", labels_add=["type::bug"], labels_remove=["type::feature"])

        assert mock_run.call_count == 2
        put_args = mock_run.call_args_list[1][0][0]
        fields = [put_args[i + 1] for i, a in enumerate(put_args) if a == "-f"]
        label_fields = [f for f in fields if f.startswith("labels=")]
        assert label_fields
        label_set = set(label_fields[0][len("labels=") :].split(","))
        assert "type::bug" in label_set
        assert "type::feature" not in label_set

    @patch("subprocess.run")
    def test_update_mr_dry_run(self, mock_run: Mock, new_config_path: Path, capsys) -> None:
        """Dry run prints intent without executing."""
        config = Config(new_config_path)
        updater = TicketUpdater(config, dry_run=True)

        updater.update_mr("144", title="Preview MR")

        mock_run.assert_not_called()
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "merge_requests/144" in captured.out

    @patch("subprocess.run")
    def test_update_mr_dry_run_with_labels_no_api_call(
        self, mock_run: Mock, new_config_path: Path, capsys
    ) -> None:
        """Dry run with labels must not make any live API call."""
        config = Config(new_config_path)
        updater = TicketUpdater(config, dry_run=True)

        updater.update_mr("144", labels_add=["type::bug"], labels_remove=["type::feature"])

        mock_run.assert_not_called()
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "add:" in captured.out

    @patch("subprocess.run")
    def test_update_mr_reviewer_and_target_branch(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """update_mr resolves reviewer username to numeric ID and sets target_branch."""
        # First call: users API for reviewer; second call: PUT
        users_response = Mock(stdout='[{"id": 7, "username": "alice"}]', stderr="", returncode=0)
        put_response = Mock(stdout='{"iid": 144, "title": "T"}', stderr="", returncode=0)
        mock_run.side_effect = [users_response, put_response]

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_mr("144", reviewer="alice", target_branch="main")

        put_args = mock_run.call_args_list[1][0][0]
        fields = [put_args[i + 1] for i, a in enumerate(put_args) if a == "-f"]
        assert any("reviewer_ids=7" in f for f in fields)
        assert any("target_branch=main" in f for f in fields)

    @patch("subprocess.run")
    def test_update_mr_assignee_resolves_user_id(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """update_mr resolves assignee username to numeric ID before sending PUT."""
        users_response = Mock(stdout='[{"id": 5, "username": "bob"}]', stderr="", returncode=0)
        put_response = Mock(stdout='{"iid": 144, "title": "T"}', stderr="", returncode=0)
        mock_run.side_effect = [users_response, put_response]

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_mr("144", assignee="bob")

        put_args = mock_run.call_args_list[1][0][0]
        fields = [put_args[i + 1] for i, a in enumerate(put_args) if a == "-f"]
        assert any("assignee_ids=5" in f for f in fields)

    @patch("subprocess.run")
    def test_update_mr_url_extracts_project_path(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """URL MR reference sets project-specific endpoint instead of :id sentinel."""
        mock_run.return_value = Mock(stdout='{"iid": 144, "title": "T"}', stderr="", returncode=0)
        config = Config(new_config_path)
        updater = TicketUpdater(config)

        url = "https://gitlab.example.com/mygroup/myproject/-/merge_requests/144"
        updater.update_mr(url, title="T")

        args = mock_run.call_args[0][0]
        endpoint_str = " ".join(args)
        assert "mygroup" in endpoint_str
        assert "merge_requests/144" in endpoint_str
        # Must NOT use the :id sentinel when a real path is available.
        assert ":id" not in endpoint_str


    def test_update_mr_invalid_label_raises(self, new_config_path: Path) -> None:
        """labels_add containing a disallowed label raises ValueError."""
        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with pytest.raises(ValueError, match="Unknown labels"):
            updater.update_mr("144", labels_add=["not-allowed::label"])

    def test_update_mr_invalid_label_raises_in_dry_run(self, new_config_path: Path) -> None:
        """Label validation fires before dry-run output for MRs too."""
        config = Config(new_config_path)
        updater = TicketUpdater(config, dry_run=True)

        with pytest.raises(ValueError, match="Unknown labels"):
            updater.update_mr("144", labels_add=["not-allowed::label"])


# ---------------------------------------------------------------------------
# update_epic
# ---------------------------------------------------------------------------


class TestUpdateEpic:
    """Tests for TicketUpdater.update_epic."""

    @patch("subprocess.run")
    def test_update_epic_title(self, mock_run: Mock, new_config_path: Path) -> None:
        """update_epic sends PUT to groups/:group/epics/:iid endpoint."""
        mock_run.return_value = Mock(
            stdout='{"iid": 37, "title": "New epic title"}', stderr="", returncode=0
        )
        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_epic("37", title="New epic title")

        args = mock_run.call_args[0][0]
        # The config fixture has default_group='test/group', so the endpoint must
        # reference that group and epics/37.
        endpoint_str = " ".join(args)
        assert "epics/37" in endpoint_str
        fields = [args[i + 1] for i, a in enumerate(args) if a == "-f"]
        assert any("title=New epic title" in f for f in fields)

    @patch("subprocess.run")
    def test_update_epic_state_event(self, mock_run: Mock, new_config_path: Path) -> None:
        """update_epic passes state_event correctly."""
        title_response = Mock(stdout='{"iid": 37, "title": "T"}', stderr="", returncode=0)
        put_response = Mock(stdout='{"iid": 37, "title": "T"}', stderr="", returncode=0)
        mock_run.side_effect = [title_response, put_response]
        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_epic("37", state_event="close")

        args = mock_run.call_args[0][0]
        fields = [args[i + 1] for i, a in enumerate(args) if a == "-f"]
        assert any("state_event=close" in f for f in fields)

    @patch("subprocess.run")
    def test_update_epic_label_merge(self, mock_run: Mock, new_config_path: Path) -> None:
        """update_epic fetches current labels (as dicts) and merges correctly."""
        title_response = Mock(
            stdout='{"iid": 37, "title": "T"}',
            stderr="",
            returncode=0,
        )
        get_response = Mock(
            stdout='{"iid": 37, "title": "T", "labels": [{"name": "epic"}, {"name": "old"}]}',
            stderr="",
            returncode=0,
        )
        put_response = Mock(stdout='{"iid": 37, "title": "T"}', stderr="", returncode=0)
        mock_run.side_effect = [title_response, get_response, put_response]

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_epic("37", labels_add=["type::bug"], labels_remove=["old"])

        assert mock_run.call_count == 3
        put_args = mock_run.call_args_list[2][0][0]
        fields = [put_args[i + 1] for i, a in enumerate(put_args) if a == "-f"]
        label_fields = [f for f in fields if f.startswith("labels=")]
        assert label_fields
        label_set = set(label_fields[0][len("labels=") :].split(","))
        assert "type::bug" in label_set
        assert "epic" in label_set
        assert "old" not in label_set

    @patch("subprocess.run")
    def test_update_epic_dry_run(self, mock_run: Mock, new_config_path: Path, capsys) -> None:
        """Dry run prints intent without executing."""
        config = Config(new_config_path)
        updater = TicketUpdater(config, dry_run=True)

        updater.update_epic("37", title="Preview epic")

        mock_run.assert_not_called()
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "epics/37" in captured.out

    @patch("subprocess.run")
    def test_update_epic_dry_run_with_labels_no_api_call(
        self, mock_run: Mock, new_config_path: Path, capsys
    ) -> None:
        """Dry run with labels must not make any live API call."""
        config = Config(new_config_path)
        updater = TicketUpdater(config, dry_run=True)

        updater.update_epic("37", labels_add=["type::bug"], labels_remove=["old"])

        mock_run.assert_not_called()
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "add:" in captured.out

    def test_update_epic_no_group_raises(self, tmp_path: Path) -> None:
        """ValueError is raised when no group is available for an epic update."""
        # Config without default_group set.
        config_data = {
            "platform": "gitlab",
            "gitlab": {
                "labels": {
                    "default": ["type::feature"],
                }
            },
        }
        config_path = tmp_path / "config_no_group.yaml"
        with open(config_path, "w", encoding="utf-8") as fh:
            yaml.dump(config_data, fh)

        config = Config(config_path)
        updater = TicketUpdater(config)

        with pytest.raises(ValueError, match="Group path is required"):
            updater.update_epic("37", title="Should fail")

    @patch("subprocess.run")
    def test_update_epic_milestone_resolves_group_id(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """update_epic resolves milestone via group milestones endpoint and sends milestone_id."""
        # First call: GET epic (title fetch); second call: GET group milestones; third call: PUT epic
        title_response = Mock(stdout='{"iid": 37, "title": "T"}', stderr="", returncode=0)
        milestones_response = Mock(
            stdout='[{"id": 77, "iid": 24, "title": "Sprint 1"}]', stderr="", returncode=0
        )
        put_response = Mock(stdout='{"iid": 37, "title": "T"}', stderr="", returncode=0)
        mock_run.side_effect = [title_response, milestones_response, put_response]

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_epic("37", milestone="Sprint 1")

        assert mock_run.call_count == 3
        # Second call must hit the group milestones endpoint, not projects.
        get_args = mock_run.call_args_list[1][0][0]
        assert "groups/" in " ".join(get_args)
        assert "milestones" in " ".join(get_args)
        assert "projects/" not in " ".join(get_args)
        # PUT call must carry the numeric database ID.
        put_args = mock_run.call_args_list[2][0][0]
        fields = [put_args[i + 1] for i, a in enumerate(put_args) if a == "-f"]
        assert any("milestone_id=77" in f for f in fields)

    @patch("subprocess.run")
    def test_update_epic_milestone_resolves_by_iid(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """update_epic resolves milestone when referenced by iid string."""
        title_response = Mock(stdout='{"iid": 37, "title": "T"}', stderr="", returncode=0)
        milestones_response = Mock(
            stdout='[{"id": 88, "iid": 5, "title": "v3.0"}]', stderr="", returncode=0
        )
        put_response = Mock(stdout='{"iid": 37, "title": "T"}', stderr="", returncode=0)
        mock_run.side_effect = [title_response, milestones_response, put_response]

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_epic("37", milestone="5")

        put_args = mock_run.call_args_list[2][0][0]
        fields = [put_args[i + 1] for i, a in enumerate(put_args) if a == "-f"]
        assert any("milestone_id=88" in f for f in fields)

    @patch("subprocess.run")
    def test_update_epic_milestone_not_found_raises(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """ValueError is raised when the specified milestone does not exist in the group."""
        title_response = Mock(stdout='{"iid": 37, "title": "T"}', stderr="", returncode=0)
        milestones_response = Mock(stdout="[]", stderr="", returncode=0)
        mock_run.side_effect = [title_response, milestones_response]

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with pytest.raises(ValueError, match="Group milestone not found"):
            updater.update_epic("37", milestone="nonexistent")

    @patch("subprocess.run")
    def test_update_epic_milestone_dry_run(
        self, mock_run: Mock, new_config_path: Path, capsys
    ) -> None:
        """Dry run with milestone shows intent and makes no API call."""
        config = Config(new_config_path)
        updater = TicketUpdater(config, dry_run=True)

        updater.update_epic("37", milestone="Sprint 1")

        mock_run.assert_not_called()
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "milestone_id" in captured.out
        assert "Sprint 1" in captured.out


    def test_update_epic_invalid_label_raises(self, new_config_path: Path) -> None:
        """labels_add containing a disallowed label raises ValueError for epics."""
        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with pytest.raises(ValueError, match="Unknown labels"):
            updater.update_epic("37", labels_add=["not-allowed::label"])

    def test_update_epic_invalid_label_raises_in_dry_run(self, new_config_path: Path) -> None:
        """Label validation fires before dry-run output for epics."""
        config = Config(new_config_path)
        updater = TicketUpdater(config, dry_run=True)

        with pytest.raises(ValueError, match="Unknown labels"):
            updater.update_epic("37", labels_add=["not-allowed::label"])


# ---------------------------------------------------------------------------
# update_milestone
# ---------------------------------------------------------------------------


class TestUpdateMilestone:
    """Tests for TicketUpdater.update_milestone."""

    @patch("subprocess.run")
    def test_update_milestone_due_date(self, mock_run: Mock, new_config_path: Path) -> None:
        """update_milestone sends PUT to projects/:id/milestones/:iid endpoint."""
        mock_run.return_value = Mock(stdout='{"iid": 10, "title": "v1.0"}', stderr="", returncode=0)
        config = Config(new_config_path)
        updater = TicketUpdater(config)

        # The config fixture has default_group set, so plain number resolves as
        # a group milestone.  Patch the group-milestone-id lookup.

        with patch.object(updater._loader, "_get_group_milestone_id", return_value="99"):
            updater.update_milestone("10", due_date="2026-04-01")

        args = mock_run.call_args[0][0]
        fields = [args[i + 1] for i, a in enumerate(args) if a == "-f"]
        assert any("due_date=2026-04-01" in f for f in fields)

    @patch("subprocess.run")
    def test_update_milestone_state_event_activate(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """update_milestone passes state_event=activate correctly."""
        mock_run.return_value = Mock(stdout='{"iid": 10, "title": "v1.0"}', stderr="", returncode=0)
        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with patch.object(updater._loader, "_get_group_milestone_id", return_value="99"):
            updater.update_milestone("10", state_event="activate")

        args = mock_run.call_args[0][0]
        fields = [args[i + 1] for i, a in enumerate(args) if a == "-f"]
        assert any("state_event=activate" in f for f in fields)

    @patch("subprocess.run")
    def test_update_milestone_dry_run(self, mock_run: Mock, new_config_path: Path, capsys) -> None:
        """Dry run prints intent without executing."""
        config = Config(new_config_path)
        updater = TicketUpdater(config, dry_run=True)

        with patch.object(updater._loader, "_get_group_milestone_id", return_value="99"):
            updater.update_milestone("10", title="Preview milestone")

        mock_run.assert_not_called()
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "milestones" in captured.out

    @patch("subprocess.run")
    def test_update_milestone_group_milestone_not_found(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """PlatformError is raised when group milestone iid cannot be resolved."""
        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with patch.object(updater._loader, "_get_group_milestone_id", return_value=None):
            with pytest.raises(PlatformError, match="not found"):
                updater.update_milestone("10", title="Should fail")

        mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# _merge_labels (unit)
# ---------------------------------------------------------------------------


class TestMergeLabels:
    """Unit tests for TicketUpdater._merge_labels."""

    def test_add_labels(self, new_config_path: Path) -> None:
        """Labels are added to the existing set."""
        updater = TicketUpdater(Config(new_config_path))
        result = updater._merge_labels(["a", "b"], add=["c"], remove=None)
        assert "a" in result
        assert "b" in result
        assert "c" in result

    def test_remove_labels(self, new_config_path: Path) -> None:
        """Labels are removed from the existing set."""
        updater = TicketUpdater(Config(new_config_path))
        result = updater._merge_labels(["a", "b"], add=None, remove=["b"])
        assert "a" in result
        assert "b" not in result

    def test_add_and_remove(self, new_config_path: Path) -> None:
        """Add and remove can be combined; result is deterministic."""
        updater = TicketUpdater(Config(new_config_path))
        result = updater._merge_labels(["a", "b"], add=["c"], remove=["a"])
        assert set(result) == {"b", "c"}

    def test_empty_operations(self, new_config_path: Path) -> None:
        """No-op returns current labels unchanged."""
        updater = TicketUpdater(Config(new_config_path))
        result = updater._merge_labels(["a", "b"], add=[], remove=[])

        assert set(result) == {"a", "b"}


# ---------------------------------------------------------------------------
# _parse_mr_reference (unit)
# ---------------------------------------------------------------------------


class TestParseMrReference:
    """Unit tests for TicketUpdater._parse_mr_reference (now returns tuple)."""

    def test_plain_number(self) -> None:
        """Plain number returns (None, iid)."""
        path, iid = TicketUpdater._parse_mr_reference("144")
        assert path is None
        assert iid == "144"

    def test_bang_prefix(self) -> None:
        """!number strips the prefix and returns (None, iid)."""
        path, iid = TicketUpdater._parse_mr_reference("!144")
        assert path is None
        assert iid == "144"

    def test_url_reference(self) -> None:
        """URL reference extracts both project path and iid."""
        url = "https://gitlab.example.com/group/project/-/merge_requests/144"
        path, iid = TicketUpdater._parse_mr_reference(url)
        assert path == "group/project"
        assert iid == "144"

    def test_url_reference_preserves_nested_group(self) -> None:
        """URL with nested group path preserves the full project path."""
        url = "https://gitlab.example.com/top/sub/project/-/merge_requests/7"
        path, iid = TicketUpdater._parse_mr_reference(url)
        assert path == "top/sub/project"
        assert iid == "7"

    def test_invalid_reference_raises(self) -> None:
        """Non-numeric reference raises ValueError."""
        with pytest.raises(ValueError, match="Cannot parse MR reference"):
            TicketUpdater._parse_mr_reference("not-a-number")


# ---------------------------------------------------------------------------
# _resolve_user_id (unit)
# ---------------------------------------------------------------------------


class TestResolveUserId:
    """Unit tests for TicketUpdater._resolve_user_id."""

    @patch("subprocess.run")
    def test_resolve_known_user(self, mock_run: Mock, new_config_path: Path) -> None:
        """Returns numeric ID string for a known username."""
        mock_run.return_value = Mock(
            stdout='[{"id": 42, "username": "alice"}]', stderr="", returncode=0
        )
        updater = TicketUpdater(Config(new_config_path))
        result = updater._resolve_user_id("alice")
        assert result == "42"

    @patch("subprocess.run")
    def test_resolve_unknown_user_raises(self, mock_run: Mock, new_config_path: Path) -> None:
        """ValueError is raised when the API returns an empty list."""
        mock_run.return_value = Mock(stdout="[]", stderr="", returncode=0)
        updater = TicketUpdater(Config(new_config_path))

        with pytest.raises(ValueError, match="No GitLab user found"):
            updater._resolve_user_id("ghost")


# ---------------------------------------------------------------------------
# _resolve_milestone_id (unit)
# ---------------------------------------------------------------------------


class TestResolveMilestoneId:
    """Unit tests for TicketUpdater._resolve_milestone_id."""

    @patch("subprocess.run")
    def test_resolve_by_title(self, mock_run: Mock, new_config_path: Path) -> None:
        """Returns milestone database ID when matched by title."""
        mock_run.return_value = Mock(
            stdout='[{"id": 99, "iid": 10, "title": "v2.0"}]', stderr="", returncode=0
        )
        updater = TicketUpdater(Config(new_config_path))
        result = updater._resolve_milestone_id("v2.0")
        assert result == "99"

    @patch("subprocess.run")
    def test_resolve_by_iid_string(self, mock_run: Mock, new_config_path: Path) -> None:
        """Returns milestone database ID when matched by iid string."""
        mock_run.return_value = Mock(
            stdout='[{"id": 99, "iid": 10, "title": "v2.0"}]', stderr="", returncode=0
        )
        updater = TicketUpdater(Config(new_config_path))
        result = updater._resolve_milestone_id("10")
        assert result == "99"

    @patch("subprocess.run")
    def test_resolve_not_found_raises(self, mock_run: Mock, new_config_path: Path) -> None:
        """ValueError is raised when no milestone matches the reference."""
        mock_run.return_value = Mock(stdout="[]", stderr="", returncode=0)
        updater = TicketUpdater(Config(new_config_path))

        with pytest.raises(ValueError, match="Milestone not found"):
            updater._resolve_milestone_id("nonexistent")


# ---------------------------------------------------------------------------
# update_issue — additional field coverage
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _resolve_epic_global_id (unit)
# ---------------------------------------------------------------------------


class TestResolveEpicGlobalId:
    """Unit tests for TicketUpdater._resolve_epic_global_id."""

    @patch("subprocess.run")
    def test_resolve_by_ampersand_ref(self, mock_run: Mock, new_config_path: Path) -> None:
        """Returns (global_id, iid) for a plain &number epic reference."""
        mock_run.return_value = Mock(stdout='{"id": 999, "iid": 47}', stderr="", returncode=0)
        updater = TicketUpdater(Config(new_config_path))

        global_id, iid = updater._resolve_epic_global_id("&47")

        assert global_id == "999"
        assert iid == "47"

    @patch("subprocess.run")
    def test_resolve_uses_default_group(self, mock_run: Mock, new_config_path: Path) -> None:
        """Endpoint uses the config default_group when ref has no group path."""
        mock_run.return_value = Mock(stdout='{"id": 999, "iid": 47}', stderr="", returncode=0)
        updater = TicketUpdater(Config(new_config_path))
        updater._resolve_epic_global_id("47")

        args = mock_run.call_args[0][0]
        # config has default_group = 'test/group'
        assert "test" in " ".join(args) or "group" in " ".join(args)
        assert "epics/47" in " ".join(args)

    def test_resolve_no_group_raises(self, tmp_path: Path) -> None:
        """ValueError is raised when no group is available for epic resolution."""
        config_data = {
            "platform": "gitlab",
            "gitlab": {"labels": {"default": ["type::feature"]}},
        }
        config_path = tmp_path / "no_group.yaml"
        with open(config_path, "w", encoding="utf-8") as fh:
            yaml.dump(config_data, fh)

        updater = TicketUpdater(Config(config_path))
        with pytest.raises(ValueError, match="Group path is required"):
            updater._resolve_epic_global_id("47")


# ---------------------------------------------------------------------------
# _assign_issue_to_epic (unit)
# ---------------------------------------------------------------------------


class TestAssignIssueToEpic:  # pylint: disable=too-few-public-methods
    """Unit tests for TicketUpdater._assign_issue_to_epic."""

    @patch("subprocess.run")
    def test_assigns_issue_to_epic(self, mock_run: Mock, new_config_path: Path, capsys) -> None:
        """Sends PUT with epic_id and prints confirmation."""
        # Call 1: fetch epic data; Call 2: PUT issue with epic_id
        epic_response = Mock(stdout='{"id": 999, "iid": 47}', stderr="", returncode=0)
        put_response = Mock(stdout='{"iid": 231, "title": "T"}', stderr="", returncode=0)
        mock_run.side_effect = [epic_response, put_response]

        updater = TicketUpdater(Config(new_config_path))
        updater._assign_issue_to_epic("231", "&47")

        assert mock_run.call_count == 2
        put_args = mock_run.call_args_list[1][0][0]
        fields = [put_args[i + 1] for i, a in enumerate(put_args) if a == "-f"]
        assert any("epic_id=999" in f for f in fields)
        captured = capsys.readouterr()
        assert "231" in captured.out
        assert "47" in captured.out


# ---------------------------------------------------------------------------
# update_issue — epic assignment
# ---------------------------------------------------------------------------


class TestUpdateIssueEpic:
    """Tests for update_issue epic assignment path."""

    @patch("subprocess.run")
    def test_update_issue_epic_only(self, mock_run: Mock, new_config_path: Path) -> None:
        """update_issue with epic only skips PUT and calls _assign_issue_to_epic."""
        epic_response = Mock(stdout='{"id": 999, "iid": 47}', stderr="", returncode=0)
        put_response = Mock(stdout='{"iid": 231, "title": "T"}', stderr="", returncode=0)
        mock_run.side_effect = [epic_response, put_response]

        updater = TicketUpdater(Config(new_config_path))
        result = updater.update_issue("231", epic="&47")

        # No PUT for issue fields (only epic fetch + issue PUT with epic_id)
        assert mock_run.call_count == 2
        put_args = mock_run.call_args_list[1][0][0]
        fields = [put_args[i + 1] for i, a in enumerate(put_args) if a == "-f"]
        assert any("epic_id=999" in f for f in fields)
        assert result == {}  # no PUT-based result when only epic is set

    @patch("subprocess.run")
    def test_update_issue_title_and_epic(self, mock_run: Mock, new_config_path: Path) -> None:
        """update_issue with title + epic performs PUT first, then epic assignment."""
        # Call order: PUT issue, fetch epic, PUT issue with epic_id
        put_issue_response = Mock(stdout='{"iid": 231, "title": "New"}', stderr="", returncode=0)
        epic_response = Mock(stdout='{"id": 999, "iid": 47}', stderr="", returncode=0)
        epic_put_response = Mock(stdout='{"iid": 231, "title": "New"}', stderr="", returncode=0)
        mock_run.side_effect = [put_issue_response, epic_response, epic_put_response]

        updater = TicketUpdater(Config(new_config_path))
        result = updater.update_issue("231", title="New", epic="&47")

        assert mock_run.call_count == 3
        assert result.get("iid") == 231

    @patch("subprocess.run")
    def test_update_issue_dry_run_epic(self, mock_run: Mock, new_config_path: Path, capsys) -> None:
        """Dry run with epic prints intent for both title and epic without API calls."""
        updater = TicketUpdater(Config(new_config_path), dry_run=True)

        updater.update_issue("231", title="Preview", epic="&47")

        mock_run.assert_not_called()
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "47" in captured.out

    @patch("subprocess.run")
    def test_update_issue_dry_run_epic_only(
        self, mock_run: Mock, new_config_path: Path, capsys
    ) -> None:
        """Dry run with epic only (no other fields) prints epic intent without API calls."""
        updater = TicketUpdater(Config(new_config_path), dry_run=True)

        updater.update_issue("231", epic="&47")

        mock_run.assert_not_called()
        captured = capsys.readouterr()
        assert "47" in captured.out


# ---------------------------------------------------------------------------
# cmd_update — --epic CLI validation
# ---------------------------------------------------------------------------


class TestCmdUpdateEpicValidation:
    """Tests for cmd_update --epic flag validation."""

    def test_epic_rejected_for_mr(self, new_config_path: Path) -> None:
        """--epic is rejected with an error when used on an MR resource."""
        old_argv = sys.argv
        try:
            sys.argv = [
                "projctl",
                "--config",
                str(new_config_path),
                "update",
                "mr",
                "144",
                "--epic",
                "&47",
            ]
            result = cli_main()
        finally:
            sys.argv = old_argv

        assert result == 1

    def test_epic_rejected_for_milestone(self, new_config_path: Path) -> None:
        """--epic is rejected with an error when used on a milestone resource."""
        old_argv = sys.argv
        try:
            sys.argv = [
                "projctl",
                "--config",
                str(new_config_path),
                "update",
                "milestone",
                "10",
                "--epic",
                "&47",
            ]
            result = cli_main()
        finally:
            sys.argv = old_argv

        assert result == 1

    def test_epic_alone_counts_as_update_field(self, new_config_path: Path) -> None:
        """--epic alone satisfies the 'at least one update field' requirement."""
        old_argv = sys.argv
        try:
            sys.argv = [
                "projctl",
                "--config",
                str(new_config_path),
                "update",
                "issue",
                "231",
                "--epic",
                "&47",
            ]
            epic_response = Mock(stdout='{"id": 999, "iid": 47}', stderr="", returncode=0)
            put_response = Mock(stdout='{"iid": 231, "title": "T"}', stderr="", returncode=0)
            with patch("subprocess.run", side_effect=[epic_response, put_response]):
                result = cli_main()
        finally:
            sys.argv = old_argv

        assert result == 0

    def test_milestone_accepted_for_epic(self, new_config_path: Path) -> None:
        """--milestone is accepted for epic resources and triggers update_epic with milestone."""
        old_argv = sys.argv
        try:
            sys.argv = [
                "projctl",
                "--config",
                str(new_config_path),
                "update",
                "epic",
                "43",
                "--milestone",
                "24",
            ]
            title_response = Mock(stdout='{"iid": 43, "title": "My Epic"}', stderr="", returncode=0)
            milestones_response = Mock(
                stdout='[{"id": 200, "iid": 24, "title": "Sprint 2"}]', stderr="", returncode=0
            )
            put_response = Mock(stdout='{"iid": 43, "title": "My Epic"}', stderr="", returncode=0)
            with patch(
                "subprocess.run", side_effect=[title_response, milestones_response, put_response]
            ):
                result = cli_main()
        finally:
            sys.argv = old_argv

        assert result == 0

    def test_milestone_rejected_for_milestone_resource(self, new_config_path: Path) -> None:
        """--milestone is still rejected when the resource type is milestone."""
        old_argv = sys.argv
        try:
            sys.argv = [
                "projctl",
                "--config",
                str(new_config_path),
                "update",
                "milestone",
                "10",
                "--milestone",
                "24",
            ]
            result = cli_main()
        finally:
            sys.argv = old_argv

        assert result == 1


class TestUpdateIssueFields:
    """Additional update_issue tests covering assignee, milestone, and URL refs."""

    @patch("subprocess.run")
    def test_update_issue_assignee_and_milestone(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """update_issue resolves assignee and milestone before sending PUT."""
        # Call order: users API, milestones API, PUT
        users_response = Mock(stdout='[{"id": 42, "username": "alice"}]', stderr="", returncode=0)
        milestones_response = Mock(
            stdout='[{"id": 99, "iid": 42, "title": "v1.0"}]', stderr="", returncode=0
        )
        put_response = Mock(stdout='{"iid": 231, "title": "T"}', stderr="", returncode=0)
        mock_run.side_effect = [users_response, milestones_response, put_response]

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_issue("231", assignee="alice", milestone="42")

        put_args = mock_run.call_args_list[2][0][0]
        fields = [put_args[i + 1] for i, a in enumerate(put_args) if a == "-f"]
        assert any("assignee_ids=42" in f for f in fields)
        assert any("milestone_id=99" in f for f in fields)

    @patch("subprocess.run")
    def test_update_issue_url_reference(self, mock_run: Mock, new_config_path: Path) -> None:
        """URL issue reference resolves to the correct project-encoded endpoint."""
        mock_run.return_value = Mock(stdout='{"iid": 231, "title": "T"}', stderr="", returncode=0)
        config = Config(new_config_path)
        updater = TicketUpdater(config)

        url = "https://gitlab.example.com/mygroup/myproject/-/issues/231"
        updater.update_issue(url, title="T")

        args = mock_run.call_args[0][0]
        endpoint_str = " ".join(args)
        # URL-encoded project path must appear in the endpoint.
        assert "mygroup" in endpoint_str
        assert "issues/231" in endpoint_str


# ---------------------------------------------------------------------------
# add_issue_link / remove_issue_link (unit)
# ---------------------------------------------------------------------------


class TestAddIssueLink:
    """Tests for TicketUpdater.add_issue_link."""

    @patch("subprocess.run")
    def test_add_link_plain_numbers(self, mock_run: Mock, new_config_path: Path, capsys) -> None:
        """POSTs to the source issue's /links endpoint with the resolved project ID."""
        # Call 1: project ID resolution. Call 2: POST to /links.
        project_response = Mock(stdout='{"id": 555}', stderr="", returncode=0)
        post_response = Mock(stdout="{}", stderr="", returncode=0)
        mock_run.side_effect = [project_response, post_response]

        updater = TicketUpdater(Config(new_config_path))
        updater.add_issue_link("376", "385")

        assert mock_run.call_count == 2
        post_args = mock_run.call_args_list[1][0][0]
        joined = " ".join(post_args)
        assert "-X" in post_args and "POST" in post_args
        assert "issues/376/links" in joined
        fields = [post_args[i + 1] for i, a in enumerate(post_args) if a == "-f"]
        assert "target_project_id=555" in fields
        assert "target_issue_iid=385" in fields
        assert "link_type=is_blocked_by" in fields
        captured = capsys.readouterr()
        assert "376" in captured.out and "385" in captured.out

    @patch("subprocess.run")
    def test_add_link_custom_link_type(self, mock_run: Mock, new_config_path: Path) -> None:
        """A non-default link_type is passed through to the POST body."""
        project_response = Mock(stdout='{"id": 555}', stderr="", returncode=0)
        post_response = Mock(stdout="{}", stderr="", returncode=0)
        mock_run.side_effect = [project_response, post_response]

        updater = TicketUpdater(Config(new_config_path))
        updater.add_issue_link("376", "385", link_type="relates_to")

        post_args = mock_run.call_args_list[1][0][0]
        fields = [post_args[i + 1] for i, a in enumerate(post_args) if a == "-f"]
        assert "link_type=relates_to" in fields

    @patch("subprocess.run")
    def test_add_link_hash_prefix_target(self, mock_run: Mock, new_config_path: Path) -> None:
        """#N syntax on the target strips the prefix in the POST body."""
        project_response = Mock(stdout='{"id": 555}', stderr="", returncode=0)
        post_response = Mock(stdout="{}", stderr="", returncode=0)
        mock_run.side_effect = [project_response, post_response]

        updater = TicketUpdater(Config(new_config_path))
        updater.add_issue_link("376", "#385")

        post_args = mock_run.call_args_list[1][0][0]
        fields = [post_args[i + 1] for i, a in enumerate(post_args) if a == "-f"]
        assert "target_issue_iid=385" in fields

    @patch("subprocess.run")
    def test_add_link_url_source_uses_encoded_project(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """URL source ref produces a URL-encoded project path in the endpoint."""
        project_response = Mock(stdout='{"id": 555}', stderr="", returncode=0)
        post_response = Mock(stdout="{}", stderr="", returncode=0)
        mock_run.side_effect = [project_response, post_response]

        updater = TicketUpdater(Config(new_config_path))
        url = "https://gitlab.example.com/mygroup/myproject/-/issues/376"
        updater.add_issue_link(url, "385")

        post_args = mock_run.call_args_list[1][0][0]
        joined = " ".join(post_args)
        # URL-encoded path 'mygroup%2Fmyproject' appears; and IID resolves to 376.
        assert "mygroup" in joined
        assert "issues/376/links" in joined

    @patch("subprocess.run")
    def test_add_link_dry_run_no_api_calls(
        self, mock_run: Mock, new_config_path: Path, capsys
    ) -> None:
        """Dry run prints intent and makes no API calls (including no project ID lookup)."""
        updater = TicketUpdater(Config(new_config_path), dry_run=True)

        updater.add_issue_link("376", "385")

        mock_run.assert_not_called()
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "issues/376/links" in captured.out
        assert "385" in captured.out

    def test_add_link_invalid_source_ref_raises(self, new_config_path: Path) -> None:
        """Unparseable source reference surfaces a ValueError from the loader."""
        updater = TicketUpdater(Config(new_config_path))

        with pytest.raises(ValueError, match="Cannot parse issue reference"):
            updater.add_issue_link("not-a-ref", "385")

    def test_add_link_invalid_target_ref_raises(self, new_config_path: Path) -> None:
        """Unparseable target reference surfaces a ValueError from the loader."""
        updater = TicketUpdater(Config(new_config_path))

        with pytest.raises(ValueError, match="Cannot parse issue reference"):
            updater.add_issue_link("376", "not-a-ref")


class TestRemoveIssueLink:
    """Tests for TicketUpdater.remove_issue_link."""

    @patch("subprocess.run")
    def test_remove_link_matches_by_iid_and_uses_issue_link_id(
        self, mock_run: Mock, new_config_path: Path, capsys
    ) -> None:
        """DELETE targets links/<issue_link_id> — NOT links/<linked-issue-id>."""
        # Call 1: GET current links. Call 2: DELETE by link ID.
        links_payload = (
            '[{"id": 9999, "iid": 385, "issue_link_id": 42, "title": "Blocker"},'
            ' {"id": 8888, "iid": 400, "issue_link_id": 43, "title": "Other"}]'
        )
        get_response = Mock(stdout=links_payload, stderr="", returncode=0)
        delete_response = Mock(stdout="", stderr="", returncode=0)
        mock_run.side_effect = [get_response, delete_response]

        updater = TicketUpdater(Config(new_config_path))
        updater.remove_issue_link("376", "385")

        assert mock_run.call_count == 2
        delete_args = mock_run.call_args_list[1][0][0]
        joined = " ".join(delete_args)
        assert "-X" in delete_args and "DELETE" in delete_args
        # Uses issue_link_id (42), not the linked issue's database id (9999).
        assert "issues/376/links/42" in joined
        assert "issues/376/links/9999" not in joined
        captured = capsys.readouterr()
        assert "376" in captured.out and "385" in captured.out

    @patch("subprocess.run")
    def test_remove_link_no_matching_link_raises(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """ValueError is raised when the target IID is not present in the links list."""
        # Only one link, not matching target 385.
        links_payload = '[{"id": 8888, "iid": 400, "issue_link_id": 43}]'
        mock_run.return_value = Mock(stdout=links_payload, stderr="", returncode=0)

        updater = TicketUpdater(Config(new_config_path))

        with pytest.raises(ValueError, match="No link found"):
            updater.remove_issue_link("376", "385")

        # Exactly one call — the DELETE never happens.
        assert mock_run.call_count == 1

    @patch("subprocess.run")
    def test_remove_link_empty_links_list_raises(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """ValueError is raised when the source issue has no links at all."""
        mock_run.return_value = Mock(stdout="[]", stderr="", returncode=0)

        updater = TicketUpdater(Config(new_config_path))

        with pytest.raises(ValueError, match="No link found"):
            updater.remove_issue_link("376", "385")

    @patch("subprocess.run")
    def test_remove_link_dry_run_reads_but_does_not_delete(
        self, mock_run: Mock, new_config_path: Path, capsys
    ) -> None:
        """Dry-run must fetch links to resolve the link_id, but must not issue the DELETE."""
        links_payload = '[{"id": 9999, "iid": 385, "issue_link_id": 42}]'
        mock_run.return_value = Mock(stdout=links_payload, stderr="", returncode=0)

        updater = TicketUpdater(Config(new_config_path), dry_run=True)
        updater.remove_issue_link("376", "385")

        # Exactly one call — the GET. No DELETE followed.
        assert mock_run.call_count == 1
        get_args = mock_run.call_args_list[0][0][0]
        assert "-X" not in get_args  # a bare GET, no method override
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "issues/376/links/42" in captured.out

    @patch("subprocess.run")
    def test_remove_link_url_source_uses_encoded_project(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """URL source ref produces a URL-encoded project path in both endpoints."""
        links_payload = '[{"id": 9999, "iid": 385, "issue_link_id": 42}]'
        get_response = Mock(stdout=links_payload, stderr="", returncode=0)
        delete_response = Mock(stdout="", stderr="", returncode=0)
        mock_run.side_effect = [get_response, delete_response]

        updater = TicketUpdater(Config(new_config_path))
        url = "https://gitlab.example.com/mygroup/myproject/-/issues/376"
        updater.remove_issue_link(url, "385")

        get_args = mock_run.call_args_list[0][0][0]
        delete_args = mock_run.call_args_list[1][0][0]
        assert "mygroup" in " ".join(get_args)
        assert "mygroup" in " ".join(delete_args)


class TestResolveProjectId:  # pylint: disable=too-few-public-methods
    """Unit test for TicketUpdater._resolve_project_id."""

    @patch("subprocess.run")
    def test_resolve_project_id_returns_stringified_id(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """Numeric id from the projects API is returned as a string."""
        mock_run.return_value = Mock(
            stdout='{"id": 555, "path_with_namespace": "mygroup/myproject"}',
            stderr="",
            returncode=0,
        )
        updater = TicketUpdater(Config(new_config_path))

        result = updater._resolve_project_id()

        assert result == "555"
        args = mock_run.call_args[0][0]
        assert "projects/:fullpath" in " ".join(args)


# ---------------------------------------------------------------------------
# cmd_update — --add-blocker / --remove-blocker CLI validation
# ---------------------------------------------------------------------------


class TestCmdUpdateBlockerValidation:
    """Tests for cmd_update --add-blocker / --remove-blocker CLI wiring."""

    def _run_cli(self, args_list, side_effect=None):
        """Invoke cli_main with the given argv, optionally mocking subprocess.run."""
        old_argv = sys.argv
        try:
            sys.argv = args_list
            if side_effect is None:
                return cli_main()
            with patch("subprocess.run", side_effect=side_effect):
                return cli_main()
        finally:
            sys.argv = old_argv

    def test_add_blocker_rejected_for_mr(self, new_config_path: Path) -> None:
        """--add-blocker is rejected on MR resources."""
        result = self._run_cli(
            [
                "projctl", "--config", str(new_config_path),
                "update", "mr", "144", "--add-blocker", "252",
            ]
        )
        assert result == 1

    def test_add_blocker_rejected_for_epic(self, new_config_path: Path) -> None:
        """--add-blocker is rejected on epic resources."""
        result = self._run_cli(
            [
                "projctl", "--config", str(new_config_path),
                "update", "epic", "37", "--add-blocker", "252",
            ]
        )
        assert result == 1

    def test_remove_blocker_rejected_for_milestone(self, new_config_path: Path) -> None:
        """--remove-blocker is rejected on milestone resources."""
        result = self._run_cli(
            [
                "projctl", "--config", str(new_config_path),
                "update", "milestone", "10", "--remove-blocker", "252",
            ]
        )
        assert result == 1

    def test_add_blocker_alone_counts_as_update_field(self, new_config_path: Path) -> None:
        """--add-blocker with no other flags satisfies the at-least-one-field rule."""
        # add_issue_link makes 2 calls: project-id lookup, then POST.
        project_response = Mock(stdout='{"id": 555}', stderr="", returncode=0)
        post_response = Mock(stdout="{}", stderr="", returncode=0)

        result = self._run_cli(
            [
                "projctl", "--config", str(new_config_path),
                "update", "issue", "376", "--add-blocker", "385",
            ],
            side_effect=[project_response, post_response],
        )
        assert result == 0

    def test_remove_blocker_alone_counts_as_update_field(self, new_config_path: Path) -> None:
        """--remove-blocker with no other flags satisfies the at-least-one-field rule."""
        # remove_issue_link makes 2 calls: GET links, then DELETE.
        links_payload = '[{"id": 9999, "iid": 385, "issue_link_id": 42}]'
        get_response = Mock(stdout=links_payload, stderr="", returncode=0)
        delete_response = Mock(stdout="", stderr="", returncode=0)

        result = self._run_cli(
            [
                "projctl", "--config", str(new_config_path),
                "update", "issue", "376", "--remove-blocker", "385",
            ],
            side_effect=[get_response, delete_response],
        )
        assert result == 0

    def test_add_blocker_with_title_triggers_both_operations(
        self, new_config_path: Path
    ) -> None:
        """--add-blocker combined with --title triggers PUT (title) then POST (link)."""
        put_response = Mock(stdout='{"iid": 376, "title": "New"}', stderr="", returncode=0)
        project_response = Mock(stdout='{"id": 555}', stderr="", returncode=0)
        post_response = Mock(stdout="{}", stderr="", returncode=0)

        result = self._run_cli(
            [
                "projctl", "--config", str(new_config_path),
                "update", "issue", "376", "--title", "New", "--add-blocker", "385",
            ],
            side_effect=[put_response, project_response, post_response],
        )
        assert result == 0

    def test_add_and_remove_blocker_together(self, new_config_path: Path) -> None:
        """Both --remove-blocker and --add-blocker in one call trigger DELETE then POST."""
        # Order in cli.py: remove first, then add.
        links_payload = '[{"id": 9999, "iid": 252, "issue_link_id": 42}]'
        get_response = Mock(stdout=links_payload, stderr="", returncode=0)
        delete_response = Mock(stdout="", stderr="", returncode=0)
        project_response = Mock(stdout='{"id": 555}', stderr="", returncode=0)
        post_response = Mock(stdout="{}", stderr="", returncode=0)

        result = self._run_cli(
            [
                "projctl", "--config", str(new_config_path),
                "update", "issue", "376",
                "--remove-blocker", "252", "--add-blocker", "385",
            ],
            side_effect=[get_response, delete_response, project_response, post_response],
        )
        assert result == 0

    def test_no_flags_still_errors(self, new_config_path: Path) -> None:
        """update issue with no flags still errors even after adding blocker flags."""
        result = self._run_cli(
            [
                "projctl", "--config", str(new_config_path),
                "update", "issue", "376",
            ]
        )
        assert result == 1


# ---------------------------------------------------------------------------
# update_issue --status (work-item Status field)
# ---------------------------------------------------------------------------

_STATUS_ISSUE_URL = "https://gitlab.example.com/mygroup/myproject/-/issues/231"


def _status_resolve_response(status_names_and_ids, item_type="Issue", extra_types=()):
    """Build a mocked 'workItems + workItemTypes' GraphQL resolve response.

    Mirrors the shape a GitLab Premium instance returns: workItems(iid)
    resolves the target's work-item GID and its own type, and the matching
    entry in workItemTypes carries the live allowedStatuses list.
    """
    body = {
        "data": {
            "project": {
                "workItems": {
                    "nodes": [
                        {
                            "id": "gid://gitlab/WorkItem/205512",
                            "workItemType": {"name": item_type},
                        }
                    ]
                },
                "workItemTypes": {
                    "nodes": [
                        *extra_types,
                        {
                            "name": item_type,
                            "widgetDefinitions": [
                                {"type": "ASSIGNEES"},
                                {
                                    "type": "STATUS",
                                    "allowedStatuses": [
                                        {"id": sid, "name": name}
                                        for name, sid in status_names_and_ids
                                    ],
                                },
                            ],
                        },
                    ]
                },
            }
        }
    }
    return Mock(stdout=json.dumps(body), stderr="", returncode=0)


_DEFAULT_STATUSES = [
    ("To do", "gid://gitlab/WorkItems::Statuses::SystemDefined::Status/1"),
    ("In progress", "gid://gitlab/WorkItems::Statuses::SystemDefined::Status/2"),
    ("Done", "gid://gitlab/WorkItems::Statuses::SystemDefined::Status/3"),
]


class TestUpdateIssueStatus:
    """Tests for TicketUpdater.update_issue(status=...)."""

    @patch("subprocess.run")
    def test_status_case_insensitive_match(self, mock_run: Mock, new_config_path: Path) -> None:
        """A lowercase status name still resolves to the correctly-cased status GID."""
        resolve_response = _status_resolve_response(_DEFAULT_STATUSES)
        mutation_response = Mock(
            stdout='{"data": {"workItemUpdate": {"workItem": {"title": "T"}, "errors": []}}}',
            stderr="",
            returncode=0,
        )
        mock_run.side_effect = [resolve_response, mutation_response]

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_issue(_STATUS_ISSUE_URL, status="in progress")

        assert mock_run.call_count == 2
        resolve_args = mock_run.call_args_list[0][0][0]
        fields = [resolve_args[i + 1] for i, a in enumerate(resolve_args) if a == "-f"]
        assert "fullPath=mygroup/myproject" in fields
        assert "iid=231" in fields

        mutation_args = mock_run.call_args_list[1][0][0]
        mutation_fields = [
            mutation_args[i + 1] for i, a in enumerate(mutation_args) if a == "-f"
        ]
        mutation_str = " ".join(mutation_fields)
        # Field positions, not membership: substring assertions pass when the two
        # GIDs are transposed, which is the whole contract with GitLab.
        assert "workItemUpdate(input:" in mutation_str
        assert 'id: "gid://gitlab/WorkItem/205512"' in mutation_str
        assert (
            'statusWidget: { status: '
            '"gid://gitlab/WorkItems::Statuses::SystemDefined::Status/2" }'
        ) in mutation_str

    @patch("subprocess.run")
    def test_status_unknown_name_lists_valid_statuses(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """An unrecognised status name raises ValueError listing every valid name."""
        mock_run.return_value = _status_resolve_response(_DEFAULT_STATUSES)

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with pytest.raises(ValueError, match="Unknown status") as excinfo:
            updater.update_issue(_STATUS_ISSUE_URL, status="Bogus")

        message = str(excinfo.value)
        assert "To do" in message
        assert "In progress" in message
        assert "Done" in message
        # Only the read-only resolve call happened — no mutation was sent.
        assert mock_run.call_count == 1

    @patch("subprocess.run")
    def test_status_empty_allowed_statuses_raises_platform_error(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """A present STATUS widget with an empty allowedStatuses list raises."""
        mock_run.return_value = _status_resolve_response([])

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with pytest.raises(PlatformError, match="No Status field is configured"):
            updater.update_issue(_STATUS_ISSUE_URL, status="In progress")

    @patch("subprocess.run")
    def test_status_project_not_found_raises_platform_error(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """A null 'project' in the GraphQL response raises PlatformError."""
        mock_run.return_value = Mock(
            stdout='{"data": {"project": null}}', stderr="", returncode=0
        )

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with pytest.raises(PlatformError, match=r"Project .* was not found"):
            updater.update_issue(_STATUS_ISSUE_URL, status="In progress")

    @patch("subprocess.run")
    def test_status_no_status_widget_at_all_raises_platform_error(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """The genuinely-absent case: the type exists but carries no STATUS widget."""
        body = {
            "data": {
                "project": {
                    "workItems": {
                        "nodes": [
                            {
                                "id": "gid://gitlab/WorkItem/205512",
                                "workItemType": {"name": "Issue"},
                            }
                        ]
                    },
                    "workItemTypes": {
                        "nodes": [
                            {
                                "name": "Issue",
                                "widgetDefinitions": [
                                    {"type": "ASSIGNEES"},
                                    {"type": "LABELS"},
                                ],
                            }
                        ]
                    },
                }
            }
        }
        mock_run.return_value = Mock(stdout=json.dumps(body), stderr="", returncode=0)

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with pytest.raises(PlatformError, match="No Status field is configured"):
            updater.update_issue(_STATUS_ISSUE_URL, status="In progress")

    @patch("subprocess.run")
    def test_status_no_matching_work_item_type_raises_platform_error(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """The item's type is absent from workItemTypes entirely."""
        mock_run.return_value = _status_resolve_response(
            _DEFAULT_STATUSES, item_type="Issue"
        )
        body = json.loads(mock_run.return_value.stdout)
        body["data"]["project"]["workItems"]["nodes"][0]["workItemType"]["name"] = "Incident"
        mock_run.return_value = Mock(stdout=json.dumps(body), stderr="", returncode=0)

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with pytest.raises(PlatformError, match="No Status field is configured"):
            updater.update_issue(_STATUS_ISSUE_URL, status="In progress")

    @patch("subprocess.run")
    def test_status_issue_not_found_raises_platform_error(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """A valid project with a wrong iid: empty nodes must raise, not IndexError."""
        body = json.loads(_status_resolve_response(_DEFAULT_STATUSES).stdout)
        body["data"]["project"]["workItems"]["nodes"] = []
        mock_run.return_value = Mock(stdout=json.dumps(body), stderr="", returncode=0)

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with pytest.raises(PlatformError, match=r"Issue #231 was not found"):
            updater.update_issue(_STATUS_ISSUE_URL, status="In progress")

    @patch("subprocess.run")
    def test_status_top_level_graphql_errors_raise(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """GitLab reports query errors in an HTTP 200 body, with no `data` at all."""
        mock_run.return_value = Mock(
            stdout='{"errors": [{"message": "Forbidden"}]}', stderr="", returncode=0
        )

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with pytest.raises(PlatformError, match="GraphQL query failed"):
            updater.update_issue(_STATUS_ISSUE_URL, status="In progress")

    @patch("subprocess.run")
    def test_status_null_data_raises_platform_error_not_attributeerror(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """`.get(k, {})` returns the default only for an absent key, never a null one."""
        mock_run.return_value = Mock(
            stdout='{"data": null, "errors": [{"message": "x"}]}', stderr="", returncode=0
        )

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with pytest.raises(PlatformError):
            updater.update_issue(_STATUS_ISSUE_URL, status="In progress")

    @patch("subprocess.run")
    def test_status_mutation_with_no_work_item_is_indeterminate_not_success(
        self, mock_run: Mock, new_config_path: Path, capsys
    ) -> None:
        """Empty errors plus a null work item is a real shape and is NOT a success."""
        mutation_response = Mock(
            stdout='{"data": {"workItemUpdate": {"workItem": null, "errors": []}}}',
            stderr="",
            returncode=0,
        )
        mock_run.side_effect = [
            _status_resolve_response(_DEFAULT_STATUSES),
            mutation_response,
        ]

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with pytest.raises(PlatformError, match="indeterminate"):
            updater.update_issue(_STATUS_ISSUE_URL, status="In progress")

        assert "✓" not in capsys.readouterr().out

    @patch("subprocess.run")
    def test_status_mutation_top_level_errors_raise_not_print_success(
        self, mock_run: Mock, new_config_path: Path, capsys
    ) -> None:
        """A refused write comes back as a top-level errors body with no `data`."""
        mock_run.side_effect = [
            _status_resolve_response(_DEFAULT_STATUSES),
            Mock(
                stdout='{"errors": [{"message": "Forbidden"}]}', stderr="", returncode=0
            ),
        ]

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with pytest.raises(PlatformError, match="GraphQL query failed"):
            updater.update_issue(_STATUS_ISSUE_URL, status="In progress")

        assert "✓" not in capsys.readouterr().out

    @patch("subprocess.run")
    def test_status_dry_run_prints_mutation_without_mutating(
        self, mock_run: Mock, new_config_path: Path, capsys
    ) -> None:
        """Dry run resolves real GIDs and prints the mutation, but never sends it.

        Unlike --assignee/--milestone, whose dry-run preview is a placeholder
        with no API call at all, --status still performs its one read-only
        resolve call — that call is also what validates the status name, and
        showing the operator the literal resolved GIDs is the point of the
        preview. Only the workItemUpdate mutation itself is skipped.
        """
        mock_run.return_value = _status_resolve_response(_DEFAULT_STATUSES)

        config = Config(new_config_path)
        updater = TicketUpdater(config, dry_run=True)

        updater.update_issue(_STATUS_ISSUE_URL, status="In progress")

        assert mock_run.call_count == 1
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert 'id: "gid://gitlab/WorkItem/205512"' in captured.out
        assert (
            'statusWidget: { status: '
            '"gid://gitlab/WorkItems::Statuses::SystemDefined::Status/2" }'
        ) in captured.out

    @patch("projctl.handlers.updater.get_current_repo_path")
    @patch("subprocess.run")
    def test_status_bare_ref_resolves_project_from_git_remote(
        self, mock_run: Mock, mock_repo_path: Mock, new_config_path: Path
    ) -> None:
        """A bare issue reference (no URL) resolves fullPath via the git remote."""
        mock_repo_path.return_value = "mygroup/subgroup/myproject"
        resolve_response = _status_resolve_response(_DEFAULT_STATUSES)
        mutation_response = Mock(
            stdout='{"data": {"workItemUpdate": {"workItem": {"title": "T"}, "errors": []}}}',
            stderr="",
            returncode=0,
        )
        mock_run.side_effect = [resolve_response, mutation_response]

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_issue("523", status="In progress")

        resolve_args = mock_run.call_args_list[0][0][0]
        fields = [resolve_args[i + 1] for i, a in enumerate(resolve_args) if a == "-f"]
        assert "fullPath=mygroup/subgroup/myproject" in fields
        assert "iid=523" in fields

    @patch("projctl.handlers.updater.get_current_repo_path")
    def test_status_bare_ref_no_git_remote_raises(
        self, mock_repo_path: Mock, new_config_path: Path
    ) -> None:
        """A bare issue reference with no resolvable git remote raises ValueError."""
        mock_repo_path.return_value = None

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with pytest.raises(ValueError, match="Cannot determine the project"):
            updater.update_issue("523", status="In progress")

    @patch("subprocess.run")
    def test_status_mutation_errors_raise_platform_error(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """A non-empty 'errors' array in the mutation response raises PlatformError."""
        resolve_response = _status_resolve_response(_DEFAULT_STATUSES)
        mutation_response = Mock(
            stdout='{"data": {"workItemUpdate": {"workItem": null, '
            '"errors": ["Status is not valid"]}}}',
            stderr="",
            returncode=0,
        )
        mock_run.side_effect = [resolve_response, mutation_response]

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with pytest.raises(PlatformError, match="GraphQL status update failed"):
            updater.update_issue(_STATUS_ISSUE_URL, status="In progress")

    @patch("subprocess.run")
    def test_unknown_status_with_other_fields_writes_nothing(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """A mistyped status must abort before the PUT, not after it.

        Resolving the status after the REST write left the title committed while the
        command exited 1, so the operator could not tell what had landed.
        """
        mock_run.return_value = _status_resolve_response(_DEFAULT_STATUSES)

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with pytest.raises(ValueError, match="Unknown status"):
            updater.update_issue(_STATUS_ISSUE_URL, title="New title", status="Bogus")

        assert all("PUT" not in call[0][0] for call in mock_run.call_args_list)
        # Only the read-only resolve happened.
        assert mock_run.call_count == 1

    @patch("subprocess.run")
    def test_missing_status_widget_with_other_fields_writes_nothing(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """The same ordering guarantee for a project with no Status widget."""
        mock_run.return_value = _status_resolve_response([])

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with pytest.raises(PlatformError, match="No Status field is configured"):
            updater.update_issue(_STATUS_ISSUE_URL, state_event="close", status="In progress")

        assert all("PUT" not in call[0][0] for call in mock_run.call_args_list)

    @patch("subprocess.run")
    def test_status_is_validated_against_the_items_own_work_item_type(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """Tasks share the issue iid namespace but have their own status lifecycle.

        Validating against a hardcoded 'Issue' accepted a name the Task's own
        lifecycle does not contain, then sent it a GID from the wrong type.
        """
        issue_only = {
            "name": "Issue",
            "widgetDefinitions": [
                {
                    "type": "STATUS",
                    "allowedStatuses": [
                        {"id": "gid://gitlab/Custom::Status/38", "name": "Blocked"}
                    ],
                }
            ],
        }
        mock_run.return_value = _status_resolve_response(
            [("In dev", "gid://gitlab/Custom::Status/61")],
            item_type="Task",
            extra_types=(issue_only,),
        )

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        with pytest.raises(ValueError) as excinfo:
            updater.update_issue(_STATUS_ISSUE_URL, status="Blocked")

        message = str(excinfo.value)
        assert "Task" in message
        assert "In dev" in message
        # The Issue lifecycle's status must not be offered or sent.
        assert "Custom::Status/38" not in message
        assert mock_run.call_count == 1

    @patch("subprocess.run")
    def test_status_valid_for_the_items_own_type_is_applied(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """The positive half: a Task's own status resolves to the Task's GID."""
        mutation_response = Mock(
            stdout='{"data": {"workItemUpdate": {"workItem": {"title": "T"}, "errors": []}}}',
            stderr="",
            returncode=0,
        )
        mock_run.side_effect = [
            _status_resolve_response(
                [("In dev", "gid://gitlab/Custom::Status/61")], item_type="Task"
            ),
            mutation_response,
        ]

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_issue(_STATUS_ISSUE_URL, status="in dev")

        mutation_args = mock_run.call_args_list[1][0][0]
        mutation_str = " ".join(mutation_args)
        assert 'statusWidget: { status: "gid://gitlab/Custom::Status/61" }' in mutation_str

    @patch("subprocess.run")
    def test_status_matching_uses_unicode_case_folding(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """`.lower()` leaves 'Straße' != 'STRASSE'; a custom lifecycle may use either."""
        mutation_response = Mock(
            stdout='{"data": {"workItemUpdate": {"workItem": {"title": "T"}, "errors": []}}}',
            stderr="",
            returncode=0,
        )
        mock_run.side_effect = [
            _status_resolve_response([("Straße", "gid://gitlab/Custom::Status/7")]),
            mutation_response,
        ]

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_issue(_STATUS_ISSUE_URL, status="STRASSE")

        mutation_args = mock_run.call_args_list[1][0][0]
        assert 'status: "gid://gitlab/Custom::Status/7"' in " ".join(mutation_args)

    @patch("subprocess.run")
    def test_confirmation_reports_the_servers_canonical_casing(
        self, mock_run: Mock, new_config_path: Path, capsys
    ) -> None:
        """Echoing the typed casing gave no confirmation that the name was matched."""
        mutation_response = Mock(
            stdout='{"data": {"workItemUpdate": {"workItem": {"title": "T"}, "errors": []}}}',
            stderr="",
            returncode=0,
        )
        mock_run.side_effect = [
            _status_resolve_response(_DEFAULT_STATUSES),
            mutation_response,
        ]

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_issue(_STATUS_ISSUE_URL, status="in PROGRESS")

        captured = capsys.readouterr()
        assert "'In progress'" in captured.out
        assert "in PROGRESS" not in captured.out

    @patch("subprocess.run")
    def test_status_alone_triggers_no_put_call(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """--status alone does not go through the issue PUT endpoint at all."""
        resolve_response = _status_resolve_response(_DEFAULT_STATUSES)
        mutation_response = Mock(
            stdout='{"data": {"workItemUpdate": {"workItem": {"title": "T"}, "errors": []}}}',
            stderr="",
            returncode=0,
        )
        mock_run.side_effect = [resolve_response, mutation_response]

        config = Config(new_config_path)
        updater = TicketUpdater(config)

        updater.update_issue(_STATUS_ISSUE_URL, status="In progress")

        for call in mock_run.call_args_list:
            assert "PUT" not in call[0][0]


# ---------------------------------------------------------------------------
# cmd_update — --status CLI validation
# ---------------------------------------------------------------------------


class TestCmdUpdateStatusValidation:
    """Tests for cmd_update --status flag validation and dispatch."""

    def _run_cli(self, args_list, side_effect=None):
        """Invoke cli_main with the given argv, optionally mocking subprocess.run."""
        old_argv = sys.argv
        try:
            sys.argv = args_list
            if side_effect is None:
                return cli_main()
            with patch("subprocess.run", side_effect=side_effect):
                return cli_main()
        finally:
            sys.argv = old_argv

    @pytest.mark.parametrize(
        "resource, ref", [("mr", "144"), ("epic", "37"), ("milestone", "10")]
    )
    def test_status_rejected_for_non_issue_resources(
        self, new_config_path: Path, caplog, resource: str, ref: str
    ) -> None:
        """--status is rejected on every non-issue resource.

        The message, not the exit code: `cmd_update` returns 1 for this argv anyway
        via the pre-existing "No fields to update" branch, so asserting only the code
        leaves the guard deletable with the suite green.
        """
        with caplog.at_level("ERROR"):
            result = self._run_cli(
                [
                    "projctl", "--config", str(new_config_path),
                    "update", resource, ref, "--status", "In progress",
                ]
            )
        assert result == 1
        assert "--status is only valid for issue resources" in caplog.text

    def test_empty_status_is_rejected_by_name_on_an_issue(
        self, new_config_path: Path, caplog
    ) -> None:
        """`--status "$VAR"` with VAR unset must not read as "no field specified"."""
        with caplog.at_level("ERROR"):
            result = self._run_cli(
                [
                    "projctl", "--config", str(new_config_path),
                    "update", "issue", _STATUS_ISSUE_URL, "--status", "",
                ]
            )
        assert result == 1
        assert "--status requires a non-empty status name" in caplog.text

    def test_empty_status_on_an_mr_is_rejected_not_silently_dropped(
        self, new_config_path: Path, caplog
    ) -> None:
        """This exited 0 with the flag silently discarded — a silent accept."""
        with caplog.at_level("ERROR"):
            result = self._run_cli(
                [
                    "projctl", "--config", str(new_config_path),
                    "update", "mr", "144", "--title", "T", "--status", "",
                ]
            )
        assert result == 1
        assert "--status is only valid for issue resources" in caplog.text

    def test_whitespace_only_status_is_rejected(
        self, new_config_path: Path, caplog
    ) -> None:
        """Whitespace would otherwise reach the resolver as Unknown status '   '."""
        with caplog.at_level("ERROR"):
            result = self._run_cli(
                [
                    "projctl", "--config", str(new_config_path),
                    "update", "issue", _STATUS_ISSUE_URL, "--status", "   ",
                ]
            )
        assert result == 1
        assert "--status requires a non-empty status name" in caplog.text

    def test_status_alone_counts_as_update_field(self, new_config_path: Path) -> None:
        """--status alone (no other flags) satisfies the at-least-one-field rule."""
        resolve_response = _status_resolve_response(_DEFAULT_STATUSES)
        mutation_response = Mock(
            stdout='{"data": {"workItemUpdate": {"workItem": {"title": "T"}, "errors": []}}}',
            stderr="",
            returncode=0,
        )

        result = self._run_cli(
            [
                "projctl", "--config", str(new_config_path),
                "update", "issue", _STATUS_ISSUE_URL, "--status", "In progress",
            ],
            side_effect=[resolve_response, mutation_response],
        )
        assert result == 0

    def test_status_dry_run_makes_no_mutation_call(self, new_config_path: Path) -> None:
        """update --dry-run with --status resolves but never sends the mutation."""
        resolve_response = _status_resolve_response(_DEFAULT_STATUSES)

        result = self._run_cli(
            [
                "projctl", "--config", str(new_config_path),
                "update", "issue", _STATUS_ISSUE_URL,
                "--status", "In progress", "--dry-run",
            ],
            side_effect=[resolve_response],
        )
        assert result == 0

    def test_empty_status_rejected_on_github_platform(
        self, tmp_path: Path, caplog
    ) -> None:
        """Truthiness here let `--status ""` through to the GitHub updater silently."""
        cfg_path = tmp_path / "github_config.yaml"
        cfg_path.write_text("platform: github\ngithub:\n  repo: owner/repo\n")

        with caplog.at_level("ERROR"):
            result = self._run_cli(
                [
                    "projctl", "--config", str(cfg_path),
                    "update", "issue", "231", "--status", "",
                ]
            )
        assert result == 1
        assert "not supported on GitHub" in caplog.text

    def test_status_rejected_on_github_platform(self, tmp_path: Path, caplog) -> None:
        """--status is rejected outright when the configured platform is GitHub."""
        cfg_path = tmp_path / "github_config.yaml"
        cfg_path.write_text("platform: github\ngithub:\n  repo: owner/repo\n")

        with caplog.at_level("ERROR"):
            result = self._run_cli(
                [
                    "projctl", "--config", str(cfg_path),
                    "update", "issue", "231", "--status", "In progress",
                ]
            )
        assert result == 1
        assert "not supported on GitHub" in caplog.text
