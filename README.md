# CI Platform Manager

Multi-platform CI automation tool for GitLab/GitHub workflow management.

## Features

- Create issues and epics from YAML definitions with dependency tracking
- Load and display issues, epics, milestones, and merge requests
- Search issues, epics, and milestones by text query
- Create merge requests with reviewer and label assignment
- Post structured review comments on merge requests
- Sync planning folders with Google Drive across machines

## Installation

```bash
pipx install git+https://github.com/astavonin/ci-platform-manager.git
```

To upgrade:

```bash
pipx upgrade ci-platform-manager
```

For development (editable install from local clone):

```bash
git clone git@github.com:astavonin/ci-platform-manager.git ~/projects/ci-platform-manager
pipx install -e ~/projects/ci-platform-manager
```

## Configuration

Create `config.yaml` in your project root or at `~/.config/ci_platform_manager/config.yaml`:

```yaml
platform: gitlab  # or github

gitlab:
  default_group: "group/project"
  labels:
    default: ["type::feature", "development-status::backlog"]

planning_sync:
  gdrive_base: ~/GoogleDrive
```

See `CLAUDE.md` for full configuration reference and all available commands.

## Quick Start

```bash
# Create issues from YAML (dry-run first)
ci-platform-manager create --dry-run epic_definition.yaml
ci-platform-manager create epic_definition.yaml

# Load an issue, epic, milestone, or MR
ci-platform-manager load #113
ci-platform-manager load &21
ci-platform-manager load %5
ci-platform-manager load !134

# Search
ci-platform-manager search issues "streaming"

# Planning sync
ci-platform-manager sync push
ci-platform-manager sync pull --dry-run
```

## Documentation

See `CLAUDE.md` for the full command reference, YAML format, and troubleshooting guide.
