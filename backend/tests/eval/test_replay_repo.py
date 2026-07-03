"""save_snapshot must persist replay lineage; list_replays_for_root must return them."""
import pytest

from nanoresearch.eval.snapshot import RunSnapshotData


def _mk(run_id: str) -> RunSnapshotData:
    return RunSnapshotData(
        run_id=run_id, user_input="q", tool_call_chain=[], llm_calls=[],
        final_response="a", run_status="success", total_input_tokens=0,
        total_output_tokens=0, ttft_ms=None, total_duration_ms=1.0,
        tool_call_count=0, llm_call_count=0, retry_count=0,
    )


@pytest.mark.asyncio
async def test_save_and_list_replays(eval_repo):
    root_id = await eval_repo.save_snapshot(_mk("root"), uid="u1")
    r1 = await eval_repo.save_snapshot(
        _mk("rep1"), uid="u1", origin="replay",
        parent_snapshot_id=root_id, root_snapshot_id=root_id,
        replay_config={"note": "baseline re-run"},
    )
    replays = await eval_repo.list_replays_for_root(root_id)
    assert [r.id for r in replays] == [r1]
    assert replays[0].origin == "replay"
    assert replays[0].replay_config == {"note": "baseline re-run"}
