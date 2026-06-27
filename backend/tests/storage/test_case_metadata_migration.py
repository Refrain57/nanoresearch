"""Verify AgentTestCase model exposes the new B4 metadata fields after migration."""
from sqlalchemy import inspect

from nanobot.storage.models import AgentTestCase


def test_agent_test_case_has_b4_metadata_columns():
    columns = {c.name for c in inspect(AgentTestCase).columns}
    required = {
        "origin_badcase_id",
        "target_dimension",
        "added_at",
        "added_by",
        "coverage_tags",
    }
    missing = required - columns
    assert not missing, f"AgentTestCase missing B4 metadata columns: {missing}"


def test_target_dimension_is_indexed():
    indexes = {idx.name for idx in AgentTestCase.__table__.indexes}
    assert "idx_agent_test_cases_target_dimension" in indexes
