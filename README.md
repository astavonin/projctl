# CI Platform Manager

Multi-platform CI automation tool for GitLab/GitHub workflow management.

## Features

- Create issues and epics from YAML with dependency tracking
- Load issues, epics, milestones, and MRs
- Search across issues, epics, and milestones
- Create merge requests and post review comments
- Sync planning folders with Google Drive across machines

## Installation

```bash
pipx install git+https://github.com/astavonin/ci-platform-manager.git
```

**Development:**

```bash
git clone git@github.com:astavonin/ci-platform-manager.git
pipx install -e ./ci-platform-manager
```

## Usage

```bash
ci-platform-manager create --dry-run epic.yaml   # preview issue creation
ci-platform-manager create epic.yaml             # create issues

ci-platform-manager load #113    # load issue
ci-platform-manager load &21     # load epic
ci-platform-manager load !134    # load MR

ci-platform-manager search issues "streaming"

ci-platform-manager sync push    # push planning → Google Drive
ci-platform-manager sync pull    # pull Google Drive → planning
```

See `CLAUDE.md` for full configuration reference, YAML format, and all commands.
