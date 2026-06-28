"""CI gate: fail if a PR adds an agent_test_cases fixture row that lacks B4 metadata.

Scans the PR diff (via `git diff origin/main...HEAD`) for changes touching
fixture files under backend/tests/ or backend/nanoresearch/tests/ that insert into
agent_test_cases. For each added row, verifies origin_badcase_id and
target_dimension are populated and target_dimension is NOT the 'legacy_pre_b4'
sentinel (sentinel is for backfilled rows only — new rows must have a real
dimension).

Exit 0 on pass, 1 on fail.
"""
import re
import subprocess
import sys

LEGACY_SENTINEL = "legacy_pre_b4"


def _added_lines() -> list[str]:
    # Paths are relative to cwd (backend/) when called via `cd backend && uv run python ...`
    # but git diff pathspecs are also checked against the git root prefix.
    # We pass both forms to work whether called from repo root or backend/.
    result = subprocess.run(
        ["git", "diff", "origin/main...HEAD", "--", "tests/", "nanoresearch/tests/", "backend/tests/", "backend/nanoresearch/tests/"],
        capture_output=True, text=True, check=True,
        encoding="utf-8", errors="replace",
    )
    return [ln[1:] for ln in result.stdout.splitlines() if ln.startswith("+") and not ln.startswith("+++")]


def main() -> int:
    added = _added_lines()
    failures: list[str] = []
    # Heuristic: look for added Python literals that look like AgentTestCase construction or insert dicts.
    # Pattern matches "AgentTestCase(" or '"dataset_type":' to find candidate blocks; check the surrounding
    # lines for origin_badcase_id and target_dimension.
    joined = "\n".join(added)
    case_blocks = re.findall(r"(AgentTestCase\([^)]*\)|\{[^}]*?dataset_type[^}]*?\})", joined, re.DOTALL)
    for block in case_blocks:
        if "target_dimension" not in block:
            failures.append(f"Added case lacks target_dimension:\n{block[:200]}")
            continue
        if "origin_badcase_id" not in block:
            failures.append(f"Added case lacks origin_badcase_id:\n{block[:200]}")
            continue
        if LEGACY_SENTINEL in block:
            failures.append(f"Added case uses legacy sentinel '{LEGACY_SENTINEL}':\n{block[:200]}")
    if failures:
        print("CI gate failed: new test cases must carry B4 metadata.\n")
        for f in failures:
            print(f"  - {f}\n")
        return 1
    print("OK: all added test cases carry B4 metadata.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
