"""Comment/review posting handler."""

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required. Install with: pip install PyYAML") from exc


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
    """Process a finding's locations into a single (finding, location) pair.

    One comment is posted per finding regardless of how many locations are
    listed. The comment is anchored at the first location; remaining locations
    are stored in ``_extra_locations`` and rendered in the comment body.

    Args:
        finding: Finding dictionary from review YAML

    Returns:
        Single-element list containing a (modified_finding, [primary_location])
        tuple, or an empty list if the finding has no location.
    """
    locations = finding.get("locations", [])
    if "location" in finding:
        locations = [finding["location"]]

    if not locations:
        return []

    normalized = [_normalize_location(loc) for loc in locations]
    primary = normalized[0]

    modified_finding = finding.copy()
    if len(normalized) > 1:
        modified_finding["_extra_locations"] = normalized[1:]

    return [(modified_finding, [primary])]


def _load_review_data(review_file: str) -> tuple:
    """Load and validate review YAML, returning (review_data, mr_number) or raising on error.

    Args:
        review_file: Path to the review YAML file.

    Returns:
        Tuple of (review_data dict, mr_number or None).

    Raises:
        FileNotFoundError: If the review file does not exist.
        ValueError: If required fields are missing.
        yaml.YAMLError: If the YAML is invalid.
    """
    if not Path(review_file).exists():
        raise FileNotFoundError(f"Review file not found: {review_file}")

    with open(review_file, "r", encoding="utf-8") as yaml_file:
        review_data = yaml.safe_load(yaml_file)

    if "findings" not in review_data:
        raise ValueError("Review YAML must contain 'findings' field")

    return review_data, review_data.get("mr_number")


def _fetch_mr_position(mr_number: int) -> "CommentPosition | None":
    """Fetch base and head SHAs for an MR to build inline comment positions.

    Args:
        mr_number: The MR iid.

    Returns:
        CommentPosition if both SHAs are available, None otherwise.
    """
    logger.debug("Fetching MR !%s details", mr_number)
    mr_info_cmd = ["glab", "mr", "view", str(mr_number), "--output", "json"]
    result = subprocess.run(mr_info_cmd, capture_output=True, text=True, check=True)
    mr_info = json.loads(result.stdout)

    head_sha = mr_info.get("sha") or mr_info.get("diff_refs", {}).get("head_sha")
    base_sha = mr_info.get("diff_refs", {}).get("base_sha")

    if not head_sha or not base_sha:
        return None
    return CommentPosition(base_sha=base_sha, head_sha=head_sha)


def _post_inline_findings(
    mr_number: int,
    findings: list,
    position: "CommentPosition",
    dry_run: bool,
) -> tuple:
    """Process findings and post them as inline comments.

    Args:
        mr_number: The MR iid.
        findings: List of finding dicts from review YAML.
        position: CommentPosition with base and head SHAs.
        dry_run: If True, only print what would be done.

    Returns:
        Tuple of (posted_count, failed_count).
    """
    posted_count, failed_count = 0, 0
    for finding in findings:
        finding_results = _process_finding_locations(finding)
        if not finding_results:
            logger.warning("Finding '%s' has no location, skipping", finding.get("title"))
            continue
        for located_finding, locations in finding_results:
            for location in locations:
                success = post_inline_comment(
                    mr_number=mr_number,
                    finding=located_finding,
                    location=location,
                    position=position,
                    dry_run=dry_run,
                )
                if success:
                    posted_count += 1
                else:
                    failed_count += 1
    return posted_count, failed_count


def cmd_comment(args) -> int:
    """Handle the 'comment' subcommand - post review from YAML file to MR.

    Posts individual comments on specific lines in the MR diff for each finding.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        review_data, yaml_mr_number = _load_review_data(args.review_file)

        mr_number = args.mr_number or yaml_mr_number
        if mr_number is None:
            logger.error("MR number must be specified via --mr or in review YAML")
            return 1

        position = _fetch_mr_position(mr_number)
        if position is None:
            logger.error("Could not get commit SHAs from MR. Falling back to general comment.")
            return post_general_comment(mr_number, review_data, args.dry_run)

        findings = review_data.get("findings", [])
        posted_count, failed_count = _post_inline_findings(
            mr_number, findings, position, args.dry_run
        )

        if args.dry_run:
            print(
                f"\n[DRY RUN] Would post {posted_count} inline comments "
                f"({failed_count} failed) to MR !{mr_number}"
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


def _post_note_fallback(mr_number: int, location: str, comment_body: str) -> bool:
    """Post a general MR note when inline comment posting fails.

    Used when GitLab rejects an inline comment (e.g. the line is not in the
    diff). Prepends the original location so context is preserved.

    Args:
        mr_number: The MR iid.
        location: Original file:line location string.
        comment_body: Formatted comment body to post as a note.

    Returns:
        True if the note was posted successfully, False otherwise.
    """
    note_body = f"**Location:** `{location}`\n\n{comment_body}"
    cmd = [
        "glab",
        "api",
        f"projects/:id/merge_requests/{mr_number}/notes",
        "--method",
        "POST",
        "-f",
        f"body={note_body}",
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("✓ Posted note (fallback) for %s", location)
        return True
    except subprocess.CalledProcessError as err:
        logger.error("Failed to post note fallback on %s: %s", location, err.stderr)
        return False


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

        if extra_locations:
            locs_str = ", ".join(f"`{loc}`" for loc in extra_locations)
            comment_body += f"\n\n**Also affects:** {locs_str}"

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
        try:
            subprocess.run(
                cmd, input=json.dumps(payload), capture_output=True, text=True, check=True
            )
            logger.info("✓ Posted inline comment on %s:%d", file_path, line_num)
            return True
        except subprocess.CalledProcessError as err:
            # GitLab returns 500 when the line is not part of the diff.
            # Fall back to a general note so the finding is never silently dropped.
            logger.warning(
                "Inline comment failed for %s:%d (%s) — falling back to note",
                file_path,
                line_num,
                err.stderr.strip(),
            )
            return _post_note_fallback(mr_number, location, comment_body)

    except (json.JSONDecodeError, KeyError, TypeError) as err:
        logger.error("Error posting comment on %s: %s", location, err)
        return False
    except ValueError as err:
        logger.error("Invalid line number in location: %s: %s", location, err)
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
