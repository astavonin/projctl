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
    if ':' not in loc:
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
    locations = finding.get('locations', [])
    if 'location' in finding:
        locations = [finding['location']]

    if not locations:
        return []

    # Normalize locations
    normalized_locations = [_normalize_location(loc) for loc in locations]

    # Group locations by file to avoid duplicates
    files_seen = {}
    inline_findings = []

    for loc in normalized_locations:
        file_path = loc.rsplit(':', 1)[0]
        if file_path not in files_seen:
            files_seen[file_path] = loc
            # Create a modified finding with all locations in this file
            file_locations = [location for location in normalized_locations if location.startswith(file_path + ':')]
            modified_finding = finding.copy()
            if len(file_locations) > 1:
                # Add note about other lines in the same file
                modified_finding['_extra_locations'] = file_locations[1:]
            inline_findings.append((modified_finding, [files_seen[file_path]]))

    return inline_findings


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
        with open(args.review_file, 'r', encoding='utf-8') as yaml_file:
            review_data = yaml.safe_load(yaml_file)

        # Validate required fields
        if 'findings' not in review_data:
            logger.error("Review YAML must contain 'findings' field")
            return 1

        # Get MR number from args or from YAML
        mr_number = args.mr_number or review_data.get('mr_number')
        if mr_number is None:
            logger.error("MR number must be specified via --mr or in review YAML")
            return 1

        # Get MR details to obtain commit SHAs
        logger.debug("Fetching MR !%s details", mr_number)
        mr_info_cmd = ['glab', 'mr', 'view', str(mr_number), '--output', 'json']
        result = subprocess.run(mr_info_cmd, capture_output=True, text=True, check=True)
        mr_info = json.loads(result.stdout)

        # Extract commit SHAs for posting diff comments
        head_sha = mr_info.get('sha') or mr_info.get('diff_refs', {}).get('head_sha')
        base_sha = mr_info.get('diff_refs', {}).get('base_sha')

        if not head_sha or not base_sha:
            logger.error("Could not get commit SHAs from MR. Falling back to general comment.")
            # Fallback to single comment
            return post_general_comment(mr_number, review_data, args.dry_run)

        # Process findings and group by file
        findings = review_data.get('findings', [])
        inline_findings = []

        for finding in findings:
            finding_results = _process_finding_locations(finding)
            if finding_results:
                inline_findings.extend(finding_results)
            else:
                logger.warning("Finding '%s' has no location, skipping", finding.get('title'))

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
                    dry_run=args.dry_run
                )

                if success:
                    posted_count += 1
                else:
                    failed_count += 1

        if args.dry_run:
            print(f"\n[DRY RUN] Would post {posted_count} inline comments ({failed_count} failed) to MR !{mr_number}")
        else:
            logger.info("✓ Posted %d inline comments to MR !%s (%d failed)", posted_count, mr_number, failed_count)

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
    dry_run: bool = False
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
        if ':' not in location:
            logger.warning("Invalid location format: %s (expected 'file:line')", location)
            return False

        file_path, line_part = location.rsplit(':', 1)

        # Handle line ranges (use start line)
        if '-' in line_part:
            line_num = int(line_part.split('-')[0])
        else:
            line_num = int(line_part)

        # Format comment body
        severity = finding.get('severity', 'Unknown')
        title = finding.get('title', 'Untitled')
        description = finding.get('description', '').strip()
        fix = finding.get('fix', '').strip()
        extra_locations = finding.get('_extra_locations', [])

        comment_body = f"**{severity}: {title}**\n\n{description}"

        # Add other affected lines in the same file
        if extra_locations:
            lines = [loc.split(':')[-1] for loc in extra_locations]
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
                "head_sha": position.head_sha
            }
        }

        # Post using GitLab API via glab with JSON input and Content-Type header
        cmd = [
            'glab', 'api',
            f'projects/:id/merge_requests/{mr_number}/discussions',
            '--method', 'POST',
            '--header', 'Content-Type: application/json',
            '--input', '-'
        ]

        logger.debug("Posting comment to %s:%d", file_path, line_num)
        subprocess.run(
            cmd,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True
        )

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

        cmd = ['glab', 'mr', 'comment', str(mr_number), '--message', comment]
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
    title = review_data.get('title', 'Code Review')
    review_date = review_data.get('review_date', '')
    lines.append(f"# Code Review: {title}")
    if review_date:
        lines.append(f"**Review Date:** {review_date}")
    lines.append("")

    # Group findings by severity
    findings = review_data.get('findings', [])
    severity_groups: Dict[str, List[Dict[str, Any]]] = {}
    for finding in findings:
        severity = finding.get('severity', 'Unknown')
        if severity not in severity_groups:
            severity_groups[severity] = []
        severity_groups[severity].append(finding)

    # Output findings by severity (Critical, High, Medium, Low)
    severity_order = ['Critical', 'High', 'Medium', 'Low']
    finding_num = 1

    for severity in severity_order:
        if severity not in severity_groups:
            continue

        lines.append(f"## {severity} Priority Issues")
        lines.append("")

        for finding in severity_groups[severity]:
            title = finding.get('title', 'Untitled')
            description = finding.get('description', '').strip()
            location = finding.get('location')
            locations = finding.get('locations', [])
            fix = finding.get('fix', '').strip()

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
