"""Configuration migration utilities.

Handles transformation from legacy glab_config.yaml format to new multi-platform format.
"""

from typing import Any, Dict


def transform_issue_template(old_template: Dict[str, Any]) -> Dict[str, Any]:
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
