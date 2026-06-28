"""Verify TunableObjectVersion ORM class exists with the expected schema."""

from __future__ import annotations


def test_tunable_object_version_importable():
    """Class must be importable from nanoresearch.storage.models."""
    from nanoresearch.storage.models import TunableObjectVersion

    assert TunableObjectVersion.__tablename__ == "tunable_object_versions"


def test_tunable_object_version_columns():
    """Class must declare id/kind/target_id/content/active/created_at/created_by columns."""
    from nanoresearch.storage.models import TunableObjectVersion

    columns = {c.name for c in TunableObjectVersion.__table__.columns}
    assert columns == {
        "id",
        "kind",
        "target_id",
        "content",
        "active",
        "created_at",
        "created_by",
    }


def test_tunable_object_version_id_is_uuid_pk():
    from nanoresearch.storage.models import TunableObjectVersion

    pk = [c for c in TunableObjectVersion.__table__.primary_key.columns]
    assert len(pk) == 1
    assert pk[0].name == "id"


def test_agent_eval_repo_imports_cleanly():
    """The whole reason this class exists — repo must import without ImportError."""
    from nanoresearch.storage.repositories import agent_eval_repo  # noqa: F401
