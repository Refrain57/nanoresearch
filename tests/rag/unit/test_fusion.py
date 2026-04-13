"""Unit tests for RRF Fusion algorithm.

Tests the Reciprocal Rank Fusion implementation including:
- Basic fusion of multiple ranking lists
- Weighted fusion
- Edge cases (empty lists, single lists, duplicates)
- RRF score calculation correctness
"""

import pytest
from nanobot.rag.core.query_engine.fusion import RRFFusion, rrf_score
from nanobot.rag.core.types import RetrievalResult


def create_mock_retrieval_result(chunk_id: str, score: float, text: str = "") -> RetrievalResult:
    """Helper to create mock retrieval results."""
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text or f"Mock text for {chunk_id}",
        metadata={"source_path": f"{chunk_id}.pdf"},
    )


class TestRRFFusionInit:
    """Tests for RRFFusion initialization."""

    def test_init_default_k(self):
        """Test initialization with default k value."""
        fusion = RRFFusion()
        assert fusion.k == RRFFusion.DEFAULT_K
        assert fusion.k == 60

    def test_init_custom_k(self):
        """Test initialization with custom k value."""
        fusion = RRFFusion(k=20)
        assert fusion.k == 20

    def test_init_invalid_k_negative(self):
        """Test that negative k raises ValueError."""
        with pytest.raises(ValueError, match="k must be a positive integer"):
            RRFFusion(k=-1)

    def test_init_invalid_k_zero(self):
        """Test that zero k raises ValueError."""
        with pytest.raises(ValueError, match="k must be a positive integer"):
            RRFFusion(k=0)

    def test_init_invalid_k_type(self):
        """Test that non-integer k raises ValueError."""
        with pytest.raises(ValueError, match="k must be a positive integer"):
            RRFFusion(k="60")  # type: ignore


class TestRRFScoreCalculation:
    """Tests for RRF score calculation."""

    def test_rrf_score_rank_1(self):
        """Test RRF score for top-ranked document."""
        score = rrf_score(1, k=60)
        assert score == 1.0 / (60 + 1)
        assert score == pytest.approx(0.01639344, rel=1e-5)

    def test_rrf_score_rank_10(self):
        """Test RRF score for 10th-ranked document."""
        score = rrf_score(10, k=60)
        assert score == 1.0 / (60 + 10)
        assert score == pytest.approx(0.01428571, rel=1e-5)

    def test_rrf_score_different_k(self):
        """Test RRF score with different k values."""
        score_k60 = rrf_score(1, k=60)
        score_k20 = rrf_score(1, k=20)
        # Lower k gives higher scores for same rank
        assert score_k20 > score_k60

    def test_rrf_score_invalid_rank(self):
        """Test that invalid rank raises ValueError."""
        with pytest.raises(ValueError, match="rank must be a positive integer"):
            rrf_score(0, k=60)

    def test_rrf_score_invalid_k(self):
        """Test that invalid k raises ValueError."""
        with pytest.raises(ValueError, match="k must be a positive integer"):
            rrf_score(1, k=0)


class TestRRFFusionBasic:
    """Tests for basic fusion operations."""

    def test_fuse_single_list(self):
        """Test fusion with single ranking list."""
        fusion = RRFFusion(k=60)
        results = [
            create_mock_retrieval_result("a", 0.9),
            create_mock_retrieval_result("b", 0.8),
            create_mock_retrieval_result("c", 0.7),
        ]

        fused = fusion.fuse([results])
        assert len(fused) == 3
        # Single list - results should preserve original order
        assert fused[0].chunk_id == "a"

    def test_fuse_two_lists(self):
        """Test fusion of two ranking lists."""
        fusion = RRFFusion(k=60)
        dense = [
            create_mock_retrieval_result("a", 0.9),
            create_mock_retrieval_result("b", 0.8),
        ]
        sparse = [
            create_mock_retrieval_result("b", 5.0),
            create_mock_retrieval_result("c", 4.0),
        ]

        fused = fusion.fuse([dense, sparse])

        # "b" appears in both lists, should have highest RRF score
        assert fused[0].chunk_id == "b"
        assert len(fused) == 3

    def test_fuse_empty_lists(self):
        """Test fusion with all empty lists."""
        fusion = RRFFusion(k=60)

        fused = fusion.fuse([[], []])
        assert fused == []

    def test_fuse_no_lists(self):
        """Test fusion with no ranking lists provided."""
        fusion = RRFFusion(k=60)

        with pytest.raises(ValueError, match="ranking_lists cannot be empty"):
            fusion.fuse([])

    def test_fuse_with_top_k(self):
        """Test fusion respects top_k limit."""
        fusion = RRFFusion(k=60)
        results = [
            create_mock_retrieval_result(f"chunk_{i}", 0.9 - i * 0.1)
            for i in range(20)
        ]

        fused = fusion.fuse([results], top_k=5)
        assert len(fused) == 5

    def test_fuse_preserves_text_and_metadata(self):
        """Test that fusion preserves text and metadata from first occurrence."""
        fusion = RRFFusion(k=60)
        dense = [
            RetrievalResult(
                chunk_id="shared",
                score=0.9,
                text="Text from dense",
                metadata={"source": "dense.pdf", "extra": "dense_meta"},
            ),
        ]
        sparse = [
            RetrievalResult(
                chunk_id="shared",
                score=5.0,
                text="Text from sparse",
                metadata={"source": "sparse.pdf"},
            ),
        ]

        fused = fusion.fuse([dense, sparse])

        # Should preserve dense's text and metadata (first occurrence)
        assert fused[0].text == "Text from dense"
        assert fused[0].metadata["source"] == "dense.pdf"
        assert fused[0].metadata["extra"] == "dense_meta"


class TestRRFFusionWeights:
    """Tests for weighted fusion."""

    def test_fuse_with_weights_uniform(self):
        """Test weighted fusion with uniform weights."""
        fusion = RRFFusion(k=60)
        dense = [create_mock_retrieval_result("a", 0.9)]
        sparse = [create_mock_retrieval_result("b", 5.0)]

        fused = fusion.fuse_with_weights([dense, sparse], weights=[1.0, 1.0])
        assert len(fused) == 2

    def test_fuse_with_weights_dense_heavier(self):
        """Test weighted fusion with higher weight for dense."""
        fusion = RRFFusion(k=60)
        dense = [
            create_mock_retrieval_result("a", 0.9),
            create_mock_retrieval_result("b", 0.8),
        ]
        sparse = [
            create_mock_retrieval_result("b", 5.0),
            create_mock_retrieval_result("c", 4.0),
        ]

        # Give 2x weight to dense
        fused = fusion.fuse_with_weights([dense, sparse], weights=[2.0, 1.0])

        # "a" only in dense (rank 1, weight 2), "b" in both, "c" only in sparse
        # With weight 2 on dense, "a" should rank higher than "c"
        assert fused[0].chunk_id == "b"  # Still highest due to appearing in both

    def test_fuse_with_weights_zero(self):
        """Test weighted fusion with zero weight for one list."""
        fusion = RRFFusion(k=60)
        dense = [create_mock_retrieval_result("a", 0.9)]
        sparse = [create_mock_retrieval_result("b", 5.0)]

        # Zero weight on sparse means sparse contributes 0 RRF score
        # but the chunk still appears in results (just with score from dense only)
        fused = fusion.fuse_with_weights([dense, sparse], weights=[1.0, 0.0])

        # Both chunks should appear, but 'a' should have higher score (from dense)
        # 'b' should have score 0 from sparse but still be included
        assert len(fused) == 2
        # 'a' should be first (has score from dense with weight 1.0)
        assert fused[0].chunk_id == "a"

    def test_fuse_with_weights_invalid_count(self):
        """Test that mismatched weights count raises ValueError."""
        fusion = RRFFusion(k=60)
        dense = [create_mock_retrieval_result("a", 0.9)]
        sparse = [create_mock_retrieval_result("b", 5.0)]

        with pytest.raises(ValueError, match="weights length"):
            fusion.fuse_with_weights([dense, sparse], weights=[1.0])  # Missing weight

    def test_fuse_with_weights_negative(self):
        """Test that negative weight raises ValueError."""
        fusion = RRFFusion(k=60)
        dense = [create_mock_retrieval_result("a", 0.9)]
        sparse = [create_mock_retrieval_result("b", 5.0)]

        with pytest.raises(ValueError, match="Weight.*must be non-negative"):
            fusion.fuse_with_weights([dense, sparse], weights=[-1.0, 1.0])


class TestRRFFusionDeterminism:
    """Tests for deterministic behavior."""

    def test_fusion_is_deterministic(self):
        """Test that same inputs produce same outputs."""
        fusion = RRFFusion(k=60)
        dense = [
            create_mock_retrieval_result("a", 0.9),
            create_mock_retrieval_result("b", 0.8),
        ]
        sparse = [
            create_mock_retrieval_result("b", 5.0),
            create_mock_retrieval_result("a", 4.0),
        ]

        fused1 = fusion.fuse([dense, sparse])
        fused2 = fusion.fuse([dense, sparse])

        # Results should be identical
        assert len(fused1) == len(fused2)
        for r1, r2 in zip(fused1, fused2):
            assert r1.chunk_id == r2.chunk_id
            assert r1.score == pytest.approx(r2.score, rel=1e-10)

    def test_fusion_tie_breaking(self):
        """Test that ties are broken by chunk_id for stability."""
        fusion = RRFFusion(k=60)
        # Create results where two chunks have same RRF contribution
        results = [
            create_mock_retrieval_result("z", 0.9),
            create_mock_retrieval_result("a", 0.8),
            # Both at rank 1 in separate lists would have same RRF score
        ]

        # Use same rank positions to create tie
        fused = fusion.fuse([
            [create_mock_retrieval_result("z", 0.9)],
            [create_mock_retrieval_result("a", 0.9)],
        ])

        # Tie should be broken by chunk_id alphabetically
        assert fused[0].chunk_id == "a"
        assert fused[1].chunk_id == "z"


class TestRRFFusionEdgeCases:
    """Tests for edge cases."""

    def test_fuse_with_duplicate_chunk_ids_in_same_list(self):
        """Test handling of duplicate chunk_ids within same list."""
        fusion = RRFFusion(k=60)
        results = [
            create_mock_retrieval_result("a", 0.9),
            create_mock_retrieval_result("a", 0.8),  # Duplicate
        ]

        fused = fusion.fuse([results])

        # Should deduplicate - only one "a" in result
        assert len(fused) == 1
        assert fused[0].chunk_id == "a"
        # Score should be sum of both contributions
        expected_score = rrf_score(1) + rrf_score(2)
        assert fused[0].score == pytest.approx(expected_score, rel=1e-5)

    def test_fuse_mixed_empty_and_non_empty(self):
        """Test fusion with some empty lists mixed with non-empty."""
        fusion = RRFFusion(k=60)
        dense = [create_mock_retrieval_result("a", 0.9)]
        sparse = []  # Empty

        fused = fusion.fuse([dense, sparse])
        assert len(fused) == 1
        assert fused[0].chunk_id == "a"

    def test_fuse_large_ranking_lists(self):
        """Test fusion with large ranking lists."""
        fusion = RRFFusion(k=60)
        dense = [create_mock_retrieval_result(f"d_{i}", 0.9) for i in range(100)]
        sparse = [create_mock_retrieval_result(f"s_{i}", 5.0) for i in range(100)]

        fused = fusion.fuse([dense, sparse], top_k=10)
        assert len(fused) == 10

    def test_fuse_preserves_metadata_copy(self):
        """Test that metadata is copied, not referenced."""
        fusion = RRFFusion(k=60)
        results = [
            RetrievalResult(
                chunk_id="a",
                score=0.9,
                text="text",
                metadata={"key": "value"},
            ),
        ]

        fused = fusion.fuse([results])

        # Modify original metadata
        results[0].metadata["key"] = "modified"

        # Fused metadata should be unchanged (copy)
        assert fused[0].metadata["key"] == "value"