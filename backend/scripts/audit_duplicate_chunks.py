"""
Diagnostic: scan ChromaDB for chunks with identical content but different chunk_ids.
Run from backend/ dir:  python scripts/audit_duplicate_chunks.py
"""
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import chromadb

CHROMA_PATH = Path.home() / ".nanoresearch" / "rag" / "chroma"
if not CHROMA_PATH.exists():
    sys.exit(f"ChromaDB not found at {CHROMA_PATH}")

client = chromadb.PersistentClient(path=str(CHROMA_PATH))
all_cols = client.list_collections()
print(f"ChromaDB: {CHROMA_PATH}")
print(f"Collections: {[(c.name, c.count()) for c in all_cols]}\n")

# Audit the two large collections separately
TARGET_COLS = [c.name for c in all_cols if c.count() >= 300]
print(f"Auditing: {TARGET_COLS}\n")


def norm_hash(text: str) -> str:
    if not text:
        return ""
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def audit_collection(col_name: str):
    col = client.get_collection(col_name)
    print(f"\n{'=' * 60}")
    print(f"COLLECTION: {col_name}  ({col.count()} chunks)")
    print("=" * 60)

    result = col.get(include=["metadatas", "documents"])
    ids   = result["ids"]
    docs  = result["documents"]
    metas = result["metadatas"]
    total = len(ids)

    hash_to_chunks: dict[str, list[dict]] = defaultdict(list)
    for chunk_id, doc, meta in zip(ids, docs, metas):
        text = doc if doc else ((meta or {}).get("text"))
        h = norm_hash(text)
        if not h:
            continue
        source = (meta or {}).get("source_path") or (meta or {}).get("source") or "unknown"
        hash_to_chunks[h].append({"id": chunk_id, "source": source, "text": text})

    dup_groups = {h: chunks for h, chunks in hash_to_chunks.items() if len(chunks) > 1}
    dup_chunk_count = sum(len(v) for v in dup_groups.values())
    dup_group_count = len(dup_groups)
    dup_ratio = dup_chunk_count / total if total else 0

    print(f"\nSUMMARY")
    print(f"  Total chunks            : {total}")
    print(f"  Chunks involved in dups : {dup_chunk_count}")
    print(f"  Duplicate groups        : {dup_group_count}")
    print(f"  Dup chunk ratio         : {dup_ratio:.2%}")

    # Distribution by source
    source_dup_count: dict[str, int] = defaultdict(int)
    for chunks in dup_groups.values():
        for c in chunks:
            source_dup_count[c["source"]] += 1

    print(f"\nTOP 5 DOCUMENTS BY DUPLICATE CHUNK COUNT")
    print("-" * 60)
    top5 = sorted(source_dup_count.items(), key=lambda x: x[1], reverse=True)[:5]
    for rank, (src, cnt) in enumerate(top5, 1):
        print(f"  {rank}. [{cnt} dup chunks] {Path(src).name}")
        print(f"       {src}")

    # Sample up to 5 dup groups
    print(f"\nSAMPLE DUPLICATE GROUPS (up to 5)")
    print("-" * 60)
    for i, (_, group) in enumerate(list(dup_groups.items())[:5], 1):
        print(f"\n[Group {i}]  {len(group)} copies")
        for c in group:
            print(f"  id     : {c['id']}")
            print(f"  source : {Path(c['source']).name}")
        snippet = (group[0]["text"] or "")[:400].replace("\n", " ")
        print(f"  text   : {snippet!r}")


for name in TARGET_COLS:
    audit_collection(name)
