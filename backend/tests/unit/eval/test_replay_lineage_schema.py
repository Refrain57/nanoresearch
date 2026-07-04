"""AgentRunSnapshot must carry replay-lineage columns (route B persistence)."""
from nanoresearch.storage.models import AgentRunSnapshot


def test_snapshot_has_lineage_columns():
    cols = AgentRunSnapshot.__table__.columns
    assert "origin" in cols
    assert "parent_snapshot_id" in cols
    assert "root_snapshot_id" in cols
    assert "replay_config" in cols


def test_origin_defaults_to_live():
    assert AgentRunSnapshot.__table__.columns["origin"].default.arg == "live"
