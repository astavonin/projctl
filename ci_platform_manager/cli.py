"""Command-line interface for CI Platform Manager."""

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install PyYAML")
    sys.exit(1)

from .config import Config
from .exceptions import PlatformError
from .handlers.creator import EpicIssueCreator
from .handlers.loader import TicketLoader
from .handlers.pipeline_handler import PipelineHandler
from .handlers.search import SearchHandler
from .handlers.sync import PlanningSyncHandler

logger = logging.getLogger(__name__)


@dataclass
class CommentPosition:
    """Position data for posting inline MR comments."""

    base_sha: str
    head_sha: str


def _normalize_location(loc: str) -> str:
    """Normalize a location string by adding default line number if missing.

    Args:
        loc: Location string (e.g., "file.py" or "file.py:123")

    Returns:
        Normalized location with line number (e.g., "file.py:1" or "file.py:123")
    """
    if ":" not in loc:
        return f"{loc}:1"
    return loc


def _process_finding_locations(finding: Dict[str, Any]) -> list:
    """Process a finding's locations and group by file.

    Args:
        finding: Finding dictionary from review YAML

    Returns:
        List of tuples (modified_finding, [location]) for posting
    """
    # Get locations (could be single 'location' or multiple 'locations')
    locations = finding.get("locations", [])
    if "location" in finding:
        locations = [finding["location"]]

    if not locations:
        return []

    # Normalize locations
    normalized_locations = [_normalize_location(loc) for loc in locations]

    # Group locations by file to avoid duplicates
    files_seen = {}
    inline_findings = []

    for loc in normalized_locations:
        file_path = loc.rsplit(":", 1)[0]
        if file_path not in files_seen:
            files_seen[file_path] = loc
            # Create a modified finding with all locations in this file
            file_locations = [
                location
                for location in normalized_locations
                if location.startswith(file_path + ":")
            ]
            modified_finding = finding.copy()
            if len(file_locations) > 1:
                # Add note about other lines in the same file
                modified_finding["_extra_locations"] = file_locations[1:]
            inline_findings.append((modified_finding, [files_seen[file_path]]))

    return inline_findings


def cmd_create(args) -> int:
    """Handle the 'create' subcommand.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    if not args.yaml_file.exists():
        logger.error("YAML file not found: %s", args.yaml_file)
        return 1

    try:
        # Load configuration
        config_path = Path(args.config) if args.config else None
        config = Config(config_path)

        creator = EpicIssueCreator(config=config, dry_run=args.dry_run)
        creator.process_yaml_file(args.yaml_file)
        creator.print_summary()
        return 0
    except FileNotFoundError as err:
        logger.error(str(err))
        return 1
    except (PlatformError, ValueError, yaml.YAMLError) as err:
        logger.error("Error: %s", err)
        return 1


def cmd_load(args) -> int:
    """Handle the 'load' subcommand.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        # Load configuration
        config_path = Path(args.config) if args.config else None
        config = Config(config_path)

        loader = TicketLoader(config=config)

        # Determine resource type
        reference = args.reference
        resource_type = "issue"  # default

        # Check if --type is specified
        if hasattr(args, "type") and args.type:
            resource_type = args.type
        else:
            # Auto-detect based on reference format
            if reference.startswith("!"):
                resource_type = "mr"
            elif reference.startswith("&"):
                resource_type = "epic"
            elif reference.startswith("%"):
                resource_type = "milestone"
            elif "/-/merge_requests/" in reference:
                resource_type = "mr"
            elif "/-/epics/" in reference:
                resource_type = "epic"
            elif "/-/milestones/" in reference:
                resource_type = "milestone"
            # Otherwise assume issue (default behavior)

        if resource_type == "mr":
            # Load merge request
            data = loader.load_mr(reference)
            loader.print_mr_info(data)
        elif resource_type == "epic":
            # Load epic with issues
            data = loader.load_epic_with_issues(reference)
            loader.print_epic_info(data)
        elif resource_type == "milestone":
            # Load milestone with issues and epics
            data = loader.load_milestone_with_issues(reference)
            loader.print_milestone_info(data)
        else:
            # Load issue with epic
            data = loader.load_ticket_with_epic(reference)
            loader.print_ticket_info(data)

        return 0
    except FileNotFoundError as err:
        logger.error(str(err))
        return 1
    except (PlatformError, ValueError, json.JSONDecodeError) as err:
        logger.error("Error: %s", err)
        return 1


def cmd_search(args) -> int:
    """Handle the 'search' subcommand.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        # Load configuration
        config_path = Path(args.config) if args.config else None
        config = Config(config_path)

        searcher = SearchHandler(config=config)

        if args.type == "issues":
            results = searcher.search_issues(query=args.query, state=args.state, limit=args.limit)
            searcher.print_issues(results, args.query)
        elif args.type == "epics":
            results = searcher.search_epics(query=args.query, state=args.state, limit=args.limit)
            searcher.print_epics(results, args.query)
        elif args.type == "milestones":
            results = searcher.search_milestones(
                query=args.query, state=args.state, limit=args.limit
            )
            searcher.print_milestones(results, args.query)
        else:
            logger.error("Unknown search type: %s", args.type)
            return 1

        return 0
    except FileNotFoundError as err:
        logger.error(str(err))
        return 1
    except (PlatformError, ValueError, json.JSONDecodeError) as err:
        logger.error("Error: %s", err)
        return 1


def cmd_comment(args) -> int:
    """Handle the 'comment' subcommand - post review from YAML file to MR.

    Posts individual comments on specific lines in the MR diff for each finding.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        # Check if review file exists
        if not Path(args.review_file).exists():
            logger.error("Review file not found: %s", args.review_file)
            return 1

        # Load review YAML
        with open(args.review_file, "r", encoding="utf-8") as yaml_file:
            review_data = yaml.safe_load(yaml_file)

        # Validate required fields
        if "findings" not in review_data:
            logger.error("Review YAML must contain 'findings' field")
            return 1

        # Get MR number from args or from YAML
        mr_number = args.mr_number or review_data.get("mr_number")
        if mr_number is None:
            logger.error("MR number must be specified via --mr or in review YAML")
            return 1

        # Get MR details to obtain commit SHAs
        logger.debug("Fetching MR !%s details", mr_number)
        mr_info_cmd = ["glab", "mr", "view", str(mr_number), "--output", "json"]
        result = subprocess.run(mr_info_cmd, capture_output=True, text=True, check=True)
        mr_info = json.loads(result.stdout)

        # Extract commit SHAs for posting diff comments
        head_sha = mr_info.get("sha") or mr_info.get("diff_refs", {}).get("head_sha")
        base_sha = mr_info.get("diff_refs", {}).get("base_sha")

        if not head_sha or not base_sha:
            logger.error("Could not get commit SHAs from MR. Falling back to general comment.")
            # Fallback to single comment
            return post_general_comment(mr_number, review_data, args.dry_run)

        # Process findings and group by file
        findings = review_data.get("findings", [])
        inline_findings = []

        for finding in findings:
            finding_results = _process_finding_locations(finding)
            if finding_results:
                inline_findings.extend(finding_results)
            else:
                logger.warning("Finding '%s' has no location, skipping", finding.get("title"))

        # Post all findings as inline comments
        posted_count, failed_count = 0, 0
        position = CommentPosition(base_sha=base_sha, head_sha=head_sha)

        for finding, locations in inline_findings:
            for location in locations:
                success = post_inline_comment(
                    mr_number=mr_number,
                    finding=finding,
                    location=location,
                    position=position,
                    dry_run=args.dry_run,
                )

                if success:
                    posted_count += 1
                else:
                    failed_count += 1

        if args.dry_run:
            print(
                f"\n[DRY RUN] Would post {posted_count} inline comments ({failed_count} failed) to MR !{mr_number}"
            )
        else:
            logger.info(
                "✓ Posted %d inline comments to MR !%s (%d failed)",
                posted_count,
                mr_number,
                failed_count,
            )

        return 0

    except subprocess.CalledProcessError as err:
        logger.error("Command failed: %s", err.stderr)
    except (FileNotFoundError, ValueError, yaml.YAMLError, json.JSONDecodeError) as err:
        logger.error("Error: %s", err)

    return 1


def post_inline_comment(
    mr_number: int,
    finding: Dict[str, Any],
    location: str,
    position: CommentPosition,
    dry_run: bool = False,
) -> bool:
    """Post an inline comment on a specific line in the MR diff.

    Args:
        mr_number: MR number
        finding: Finding dictionary from review YAML
        location: File location string (e.g., "path/to/file.cc:123")
        position: Comment position data (base and head SHAs)
        dry_run: If True, only print what would be done

    Returns:
        True if comment posted successfully, False otherwise
    """
    try:
        # Parse location (format: "path/to/file.cc:123" or "path/to/file.cc:123-145")
        if ":" not in location:
            logger.warning("Invalid location format: %s (expected 'file:line')", location)
            return False

        file_path, line_part = location.rsplit(":", 1)

        # Handle line ranges (use start line)
        if "-" in line_part:
            line_num = int(line_part.split("-")[0])
        else:
            line_num = int(line_part)

        # Format comment body
        severity = finding.get("severity", "Unknown")
        title = finding.get("title", "Untitled")
        description = finding.get("description", "").strip()
        fix = finding.get("fix", "").strip()
        extra_locations = finding.get("_extra_locations", [])

        comment_body = f"**{severity}: {title}**\n\n{description}"

        # Add other affected lines in the same file
        if extra_locations:
            lines = [loc.split(":")[-1] for loc in extra_locations]
            comment_body += f"\n\n**Also affects lines:** {', '.join(lines)}"

        if fix:
            comment_body += f"\n\n**Fix:**\n```\n{fix}\n```"

        if dry_run:
            print(f"\n[DRY RUN] Would post comment on {file_path}:{line_num}")
            print(f"  Severity: {severity}")
            print(f"  Title: {title}")
            return True

        # Prepare JSON payload with position data including old_line: null
        payload = {
            "body": comment_body,
            "position": {
                "position_type": "text",
                "old_path": file_path,
                "new_path": file_path,
                "old_line": None,  # null for new files
                "new_line": line_num,
                "base_sha": position.base_sha,
                "start_sha": position.base_sha,
                "head_sha": position.head_sha,
            },
        }

        # Post using GitLab API via glab with JSON input and Content-Type header
        cmd = [
            "glab",
            "api",
            f"projects/:id/merge_requests/{mr_number}/discussions",
            "--method",
            "POST",
            "--header",
            "Content-Type: application/json",
            "--input",
            "-",
        ]

        logger.debug("Posting comment to %s:%d", file_path, line_num)
        subprocess.run(cmd, input=json.dumps(payload), capture_output=True, text=True, check=True)

        logger.info("✓ Posted comment on %s:%d", file_path, line_num)
        return True

    except (json.JSONDecodeError, KeyError, TypeError) as err:
        logger.error("Error posting comment on %s: %s", location, err)
        return False
    except ValueError as err:
        logger.error("Invalid line number in location: %s: %s", location, err)
        return False
    except subprocess.CalledProcessError as err:
        logger.error("Failed to post comment on %s: %s", location, err.stderr)
        return False


def post_general_comment(mr_number: int, review_data: Dict[str, Any], dry_run: bool = False) -> int:
    """Post a general comment with all findings (fallback when inline comments fail).

    Args:
        mr_number: MR number
        review_data: Review data from YAML
        dry_run: If True, only print what would be done

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        comment = format_review_comment(review_data)

        if dry_run:
            print(f"[DRY RUN] Would post general comment to MR !{mr_number}:")
            print("=" * 80)
            print(comment)
            print("=" * 80)
            return 0

        cmd = ["glab", "mr", "comment", str(mr_number), "--message", comment]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        logger.info("✓ Posted general comment to MR !%s", mr_number)
        print(f"Comment posted: {result.stdout.strip()}")
        return 0

    except subprocess.CalledProcessError as err:
        logger.error("Failed to post general comment: %s", err.stderr)
        return 1


def format_review_comment(review_data: Dict[str, Any]) -> str:
    """Format review YAML data into a markdown comment.

    Args:
        review_data: Review data from YAML file.

    Returns:
        Formatted markdown comment.
    """
    lines = []

    # Header
    title = review_data.get("title", "Code Review")
    review_date = review_data.get("review_date", "")
    lines.append(f"# Code Review: {title}")
    if review_date:
        lines.append(f"**Review Date:** {review_date}")
    lines.append("")

    # Group findings by severity
    findings = review_data.get("findings", [])
    severity_groups: Dict[str, List[Dict[str, Any]]] = {}
    for finding in findings:
        severity = finding.get("severity", "Unknown")
        if severity not in severity_groups:
            severity_groups[severity] = []
        severity_groups[severity].append(finding)

    # Output findings by severity (Critical, High, Medium, Low)
    severity_order = ["Critical", "High", "Medium", "Low"]
    finding_num = 1

    for severity in severity_order:
        if severity not in severity_groups:
            continue

        lines.append(f"## {severity} Priority Issues")
        lines.append("")

        for finding in severity_groups[severity]:
            title = finding.get("title", "Untitled")
            description = finding.get("description", "").strip()
            location = finding.get("location")
            locations = finding.get("locations", [])
            fix = finding.get("fix", "").strip()

            lines.append(f"### {finding_num}. {title}")
            finding_num += 1

            if description:
                lines.append(f"{description}")
                lines.append("")

            if location:
                lines.append(f"**Location:** `{location}`")
            elif locations:
                lines.append("**Locations:**")
                for loc in locations:
                    lines.append(f"- `{loc}`")

            if fix:
                lines.append("\n**Fix:**")
                lines.append("```")
                lines.append(fix)
                lines.append("```")

            lines.append("")

    return "\n".join(lines)


def cmd_create_mr(args) -> int:
    """Handle the 'create-mr' subcommand - create merge request from current branch.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        # Build glab mr create command
        cmd = ["glab", "mr", "create"]

        if args.title:
            cmd.extend(["--title", args.title])

        if args.description:
            cmd.extend(["--description", args.description])

        # Add boolean flags
        for flag in ["draft", "fill", "web"]:
            if getattr(args, flag, False):
                cmd.append(f"--{flag}")

        if args.assignee:
            for assignee in args.assignee:
                cmd.extend(["--assignee", assignee])

        if args.reviewer:
            for reviewer in args.reviewer:
                cmd.extend(["--reviewer", reviewer])

        if args.label:
            for label in args.label:
                cmd.extend(["--label", label])

        if args.milestone:
            cmd.extend(["--milestone", args.milestone])

        if args.target_branch:
            cmd.extend(["--target-branch", args.target_branch])

        if args.dry_run:
            print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            return 0

        # Execute command
        logger.debug("Executing: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        logger.info("✓ Merge request created")
        print(result.stdout.strip())
        return 0

    except subprocess.CalledProcessError as err:
        logger.error("Command failed: %s", err.stderr)
        return 1
    except (ValueError, OSError) as err:
        logger.error("Error: %s", err)
        return 1


def cmd_sync(args) -> int:
    """Handle the 'sync' subcommand - sync planning folder with Google Drive.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        # Load configuration
        config_path = Path(args.config) if args.config else None
        config = Config(config_path)

        handler = PlanningSyncHandler(config, dry_run=args.dry_run)

        if args.sync_command == "push":
            handler.push()
        elif args.sync_command == "pull":
            handler.pull()
        else:
            logger.error("Unknown sync command: %s", args.sync_command)
            return 1

        return 0
    except FileNotFoundError as err:
        logger.error(str(err))
        return 1
    except (PlatformError, ValueError) as err:
        logger.error("Error: %s", err)
        return 1


def cmd_pipeline_debug(args) -> int:
    """Handle the 'pipeline-debug' subcommand - debug failed pipeline jobs.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        # Load configuration
        config_path = Path(args.config) if args.config else None
        config = Config(config_path)

        handler = PipelineHandler(config)

        # Get branch name
        branch = args.branch if args.branch else handler.get_current_branch()
        logger.info("Debugging pipeline for branch: %s", branch)

        # Get pipeline
        pipeline = handler.get_current_pipeline(branch)
        pipeline_id = pipeline.get("id")
        pipeline_status = pipeline.get("status")
        pipeline_url = pipeline.get("web_url")

        if not isinstance(pipeline_id, int):
            logger.error("Invalid pipeline ID: %s", pipeline_id)
            return 1

        # Get failed jobs
        failed_jobs = handler.get_failed_jobs(pipeline_id)

        if not failed_jobs:
            print(f"\n✓ No failed jobs in pipeline #{pipeline_id}")
            print(f"Pipeline status: {pipeline_status}")
            print(f"URL: {pipeline_url}\n")
            return 0

        # Output formatted results
        print("\n# Pipeline Debug Results\n")
        print(f"**Branch:** {branch}")
        print(f"**Pipeline:** #{pipeline_id} - {pipeline_status}")
        print(f"**URL:** {pipeline_url}\n")
        print(f"## Failed Jobs ({len(failed_jobs)})\n")

        # Fetch and display logs for each failed job
        for job in failed_jobs:
            job_id = job.get("id")
            job_name = job.get("name")
            job_stage = job.get("stage")
            job_status = job.get("status")
            job_duration = job.get("duration") or 0

            print(f"### Job: {job_name}\n")
            print(f"- **Stage:** {job_stage}")
            print(f"- **Status:** {job_status}")
            print(f"- **Duration:** {job_duration:.1f}s\n")
            print("**Logs:**\n")
            print("```")

            if not isinstance(job_id, int):
                logger.warning("Invalid job ID for job %s: %s", job_name, job_id)
                print("(Job ID unavailable)")
                print("```\n")
                continue

            try:
                logs = handler.get_job_logs(job_id)
                print(logs)
            except PlatformError as err:
                logger.warning("Failed to fetch logs for job %s: %s", job_name, err)
                print(f"Error fetching logs: {err}")

            print("```\n")

        return 0

    except FileNotFoundError as err:
        logger.error(str(err))
        return 1
    except (PlatformError, ValueError) as err:
        logger.error("Error: %s", err)
        return 1


def main() -> int:
    """Main entry point for the script.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    # Compute path to CLAUDE.md documentation
    claude_md_path = (Path(__file__).parent / "CLAUDE.md").resolve()

    parser = argparse.ArgumentParser(
        description="GitLab Epic and Issue management tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Create issues from YAML
  %(prog)s create epic_definition.yaml
  %(prog)s create --dry-run epic_definition.yaml

  # Load issue information (with dependencies, markdown output)
  %(prog)s load 113
  %(prog)s load https://gitlab.example.com/group/project/-/issues/113

  # Load epic information (with all issues in the epic, markdown output)
  %(prog)s load &21
  %(prog)s load https://gitlab.example.com/groups/mygroup/-/epics/21
  %(prog)s load 21 --type epic

  # Load milestone information (with all issues and epics, markdown output)
  %(prog)s load %%123
  %(prog)s load https://gitlab.example.com/group/project/-/milestones/123
  %(prog)s load 123 --type milestone

  # Load merge request information (markdown output)
  %(prog)s load !134
  %(prog)s load 134 --type mr
  %(prog)s load https://gitlab.example.com/group/project/-/merge_requests/134

  # Search issues, epics, and milestones (text output)
  %(prog)s search issues "streaming"
  %(prog)s search issues "SRT" --state opened
  %(prog)s search epics "video"
  %(prog)s search milestones "v1.0" --state active

  # Post review comment from YAML to merge request
  %(prog)s comment planning/reviews/MR134-review.yaml
  %(prog)s comment planning/reviews/MR134-review.yaml --mr 134 --dry-run

  # Create merge request from current branch
  %(prog)s create-mr --title "Add feature X" --draft
  %(prog)s create-mr --fill --reviewer alice --label "type::feature"

  # Sync planning folder with Google Drive
  %(prog)s sync push
  %(prog)s sync pull
  %(prog)s sync push --dry-run

  # Debug failed pipeline jobs
  %(prog)s pipeline-debug
  %(prog)s pipeline-debug --branch feature/my-branch

Documentation:
  For comprehensive usage instructions, architecture details, and troubleshooting:
  {claude_md_path}
        """,
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (default: ./glab_config.yaml in current directory)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create subcommand
    create_parser = subparsers.add_parser(
        "create",
        help="Create issues from YAML and link to epic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
YAML format:
  epic:
    title: "My Epic Title"
    description: "Epic description"
    # OR use existing epic:
    # id: 123

  issues:
    - title: "Issue 1"
      description: "Description"
      labels:
        - "bug"
        - "priority::high"
      assignee: "username"
      milestone: "v1.0"
      due_date: "2025-01-15"
        """,
    )
    create_parser.add_argument(
        "yaml_file", type=Path, help="Path to YAML file containing epic and issue definitions"
    )
    create_parser.add_argument(
        "--dry-run", action="store_true", help="Preview commands without executing them"
    )

    # Load subcommand
    load_parser = subparsers.add_parser(
        "load",
        help="Load ticket (issue), epic, or milestone information from GitLab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load issue (auto-detected, markdown output)
  glab_tasks_management.py load 113
  glab_tasks_management.py load https://gitlab.example.com/group/project/-/issues/113

  # Load epic (auto-detected from & prefix or URL, markdown output)
  glab_tasks_management.py load &21
  glab_tasks_management.py load https://gitlab.example.com/groups/mygroup/-/epics/21

  # Load milestone (auto-detected from %% prefix or URL, markdown output)
  glab_tasks_management.py load %%123
  glab_tasks_management.py load https://gitlab.example.com/group/project/-/milestones/123

  # Load with explicit type specification (markdown output)
  glab_tasks_management.py load 21 --type epic
  glab_tasks_management.py load 123 --type milestone
        """,
    )
    load_parser.add_argument(
        "reference",
        type=str,
        help="Resource reference: number, URL, #number (issue), &number (epic), or %%number (milestone)",
    )
    load_parser.add_argument(
        "--type",
        choices=["issue", "epic", "milestone", "mr"],
        help="Resource type (auto-detected if not specified)",
    )

    # Search subcommand
    search_parser = subparsers.add_parser(
        "search",
        help="Search for issues, epics, or milestones by text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search issues (all states by default, text output)
  glab_tasks_management.py search issues "streaming"
  glab_tasks_management.py search issues "SRT" --state opened
  glab_tasks_management.py search issues "SRT" --state closed

  # Search epics (text output)
  glab_tasks_management.py search epics "video"
  glab_tasks_management.py search epics "streaming" --state opened

  # Search milestones (text output)
  glab_tasks_management.py search milestones "v1.0"
  glab_tasks_management.py search milestones "release" --state active

  # Limit results
  glab_tasks_management.py search issues "camera" --limit 10
        """,
    )
    search_parser.add_argument(
        "type", choices=["issues", "epics", "milestones"], help="Type of resource to search"
    )
    search_parser.add_argument(
        "query", type=str, help="Search query text (searches title and description)"
    )
    search_parser.add_argument(
        "--state",
        choices=["opened", "closed", "active", "all"],
        default="all",
        help='Filter by state (default: all). Use "active" for milestones.',
    )
    search_parser.add_argument(
        "--limit", type=int, default=20, help="Maximum number of results (default: 20)"
    )

    # Comment subcommand
    comment_parser = subparsers.add_parser(
        "comment",
        help="Post review comment from YAML file to merge request",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Post review to MR (MR number from YAML)
  glab_tasks_management.py comment planning/reviews/MR134-review.yaml

  # Post review to specific MR (override YAML)
  glab_tasks_management.py comment planning/reviews/MR134-review.yaml --mr 134

  # Preview comment without posting
  glab_tasks_management.py comment planning/reviews/MR134-review.yaml --dry-run
        """,
    )
    comment_parser.add_argument(
        "review_file",
        type=str,
        help="Path to review YAML file (e.g., planning/reviews/MR134-review.yaml)",
    )
    comment_parser.add_argument(
        "--mr", dest="mr_number", type=int, help="MR number (overrides value from YAML)"
    )
    comment_parser.add_argument(
        "--dry-run", action="store_true", help="Preview comment without posting"
    )

    # Create-MR subcommand
    create_mr_parser = subparsers.add_parser(
        "create-mr",
        help="Create merge request from current branch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create MR with interactive prompts
  glab_tasks_management.py create-mr

  # Create MR with title and description
  glab_tasks_management.py create-mr --title "Add feature X" --description "Implements feature X"

  # Create draft MR
  glab_tasks_management.py create-mr --draft --fill

  # Create MR with reviewers and labels
  glab_tasks_management.py create-mr --reviewer alice --reviewer bob --label "type::feature"

  # Preview without creating
  glab_tasks_management.py create-mr --title "Test" --dry-run
        """,
    )
    create_mr_parser.add_argument("--title", type=str, help="MR title")
    create_mr_parser.add_argument("--description", type=str, help="MR description")
    create_mr_parser.add_argument("--draft", action="store_true", help="Mark MR as draft")
    create_mr_parser.add_argument(
        "--assignee", action="append", help="Assignee username (can be repeated)"
    )
    create_mr_parser.add_argument(
        "--reviewer", action="append", help="Reviewer username (can be repeated)"
    )
    create_mr_parser.add_argument("--label", action="append", help="Label to add (can be repeated)")
    create_mr_parser.add_argument("--milestone", type=str, help="Milestone title")
    create_mr_parser.add_argument(
        "--target-branch", type=str, help="Target branch (default: default branch)"
    )
    create_mr_parser.add_argument(
        "--fill", action="store_true", help="Fill in title and description from commits"
    )
    create_mr_parser.add_argument(
        "--web", action="store_true", help="Open MR in web browser after creation"
    )
    create_mr_parser.add_argument(
        "--dry-run", action="store_true", help="Preview command without creating MR"
    )

    # Sync subcommand
    sync_parser = subparsers.add_parser(
        "sync",
        help="Sync planning folder for current repository with Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Push local planning folder to Google Drive
  ci-platform-manager sync push

  # Pull planning folder from Google Drive to local
  ci-platform-manager sync pull

  # Preview sync operation without executing
  ci-platform-manager sync push --dry-run
  ci-platform-manager sync pull --dry-run

Notes:
  - Auto-detects current repository name from git
  - Planning folder must be ./planning/ in repository root
  - Google Drive path configured in config file (planning_sync.gdrive_base)
  - Uses rsync for efficient synchronization (last write wins)
        """,
    )
    sync_subparsers = sync_parser.add_subparsers(dest="sync_command", required=True)

    # Push subcommand
    push_parser = sync_subparsers.add_parser(
        "push", help="Push local planning folder to Google Drive"
    )
    push_parser.add_argument(
        "--dry-run", action="store_true", help="Preview sync without executing"
    )

    # Pull subcommand
    pull_parser = sync_subparsers.add_parser(
        "pull", help="Pull planning folder from Google Drive to local"
    )
    pull_parser.add_argument(
        "--dry-run", action="store_true", help="Preview sync without executing"
    )

    # Pipeline-debug subcommand
    pipeline_parser = subparsers.add_parser(
        "pipeline-debug",
        help="Debug failed pipeline jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Debug pipeline for current branch
  ci-platform-manager pipeline-debug

  # Debug pipeline for specific branch
  ci-platform-manager pipeline-debug --branch feature/my-feature

Notes:
  - Auto-detects current branch if --branch not specified
  - Fetches complete logs for all failed jobs
  - Output is formatted in markdown for agent consumption
        """,
    )
    pipeline_parser.add_argument(
        "--branch", type=str, help="Branch name (default: current git branch)"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not args.command:
        parser.print_help()
        return 1

    # Command dispatch table
    commands = {
        "create": cmd_create,
        "load": cmd_load,
        "search": cmd_search,
        "comment": cmd_comment,
        "create-mr": cmd_create_mr,
        "sync": cmd_sync,
        "pipeline-debug": cmd_pipeline_debug,
    }

    try:
        handler = commands.get(args.command)
        if handler:
            return handler(args)
        parser.print_help()
        return 1
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        return 130
