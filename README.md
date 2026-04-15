# CI Platform Manager

Multi-platform CI automation tool for GitLab/GitHub workflow management.

## Features

- Create issues and epics from YAML with dependency tracking
- Load issues, epics, milestones, and MRs
- Search across issues, epics, and milestones
- Create merge requests and post review comments
- Display configured labels from the project config
- Sync planning folders with Google Drive across machines

## Installation

```bash
pipx install git+https://github.com/astavonin/projctl.git
```

**Development:**

```bash
git clone git@github.com:astavonin/projctl.git
pipx install -e ./projctl
```

## Usage

```bash
projctl create --dry-run epic.yaml   # preview issue creation
projctl create epic.yaml             # create issues

projctl load #113    # load issue
projctl load &21     # load epic
projctl load !134    # load MR

projctl search issues "streaming"

projctl labels       # show configured labels

projctl sync push    # push planning → Google Drive
projctl sync pull    # pull Google Drive → planning
```

See `CLAUDE.md` for full configuration reference, YAML format, and all commands.
