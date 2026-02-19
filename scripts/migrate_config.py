#!/usr/bin/env python3
"""Configuration migration script.

Migrates old glab_config.yaml format to new multi-platform config format.

Usage:
    python scripts/migrate_config.py glab_config.yaml config.yaml
"""

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install PyYAML")
    sys.exit(1)


def transform_issue_template(old_template: dict) -> dict:
    """Transform old issue_template schema to new format.

    Old format:
        issue_template:
          sections:
            - name: "Description"
              required: true
            - name: "Acceptance Criteria"
              required: false

    New format:
        common:
          issue_template:
            required_sections:
              - "Description"

    Args:
        old_template: Old issue template configuration.

    Returns:
        New issue template configuration.
    """
    if not old_template:
        return {'required_sections': ['Description']}

    sections = old_template.get('sections', [])
    required_sections = [
        section['name']
        for section in sections
        if section.get('required', False)
    ]

    # Ensure at least "Description" is required
    if not required_sections:
        required_sections = ['Description']

    return {'required_sections': required_sections}


def migrate_config(old_config: dict) -> dict:
    """Migrate old config format to new format.

    Args:
        old_config: Configuration in old format.

    Returns:
        Configuration in new format.
    """
    old_template = old_config.get('issue_template', {})
    new_template = transform_issue_template(old_template)

    return {
        'platform': 'gitlab',
        'gitlab': {
            'default_group': old_config.get('gitlab', {}).get('default_group'),
            'labels': {
                'default': old_config.get('labels', {}).get('default', []),
                'default_epic': old_config.get('labels', {}).get('default_epic', []),
                'allowed': old_config.get('labels', {}).get('allowed_labels', [])
            }
        },
        'common': {
            'issue_template': new_template
        }
    }


def main():
    """Execute config migration."""
    parser = argparse.ArgumentParser(
        description='Migrate glab_config.yaml to new multi-platform format'
    )
    parser.add_argument(
        'input',
        type=Path,
        help='Input config file (old format)'
    )
    parser.add_argument(
        'output',
        type=Path,
        help='Output config file (new format)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print output without writing file'
    )

    args = parser.parse_args()

    # Load old config
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    with open(args.input, 'r') as f:
        old_config = yaml.safe_load(f)

    # Migrate
    new_config = migrate_config(old_config)

    if args.dry_run:
        print("# Migrated configuration (dry-run):")
        print(yaml.dump(new_config, default_flow_style=False))
        return 0

    # Write new config
    with open(args.output, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)

    print(f"✓ Migrated {args.input} → {args.output}")
    print(f"  Platform: {new_config['platform']}")
    print(f"  Required sections: {new_config['common']['issue_template']['required_sections']}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
