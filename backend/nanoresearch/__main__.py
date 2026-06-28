"""
Entry point for running nanoresearch as a module: python -m nanoresearch
"""

from nanoresearch.utils.env_compat import apply_legacy_env_compat

apply_legacy_env_compat()

from nanoresearch.cli.commands import app  # noqa: E402

if __name__ == "__main__":
    app()
