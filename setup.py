"""Setup configuration for CI Platform Manager."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text() if readme_file.exists() else ''

setup(
    name='ci-platform-manager',
    version='2.0.0',
    author='GenAI Automations',
    description='Multi-platform CI automation tool for GitLab and GitHub',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/genai-automations/ci-platform-manager',
    packages=find_packages(exclude=['tests', 'tests.*']),
    python_requires='>=3.7',
    install_requires=[
        'PyYAML>=5.4',
    ],
    entry_points={
        'console_scripts': [
            'ci-platform-manager=ci_platform_manager.cli:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='gitlab github ci automation issues epics milestones',
)
