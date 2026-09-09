"""
Unit tests for the retrieve module.

Tests chunking logic (no embedding model needed) and retrieval contracts.
The vector store tests that need ChromaDB + embeddings are marked with
a comment — run them after `build_vector_store()` on your machine.
"""

import json
from pathlib import Path

import pytest

from src.retrieve import (
    RetrievalResult,
    RetrievedChunk,
    _chunk_document,
)

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

SAMPLE_DOC = {
    "doc_id": "DOC-TEST-001",
    "title": "Test document for unit tests",
    "category": "testing",
    "applies_to": "All plans",
    "content": (
        "# Test document for unit tests\n\n"
        "**Applies to:** All plans\n\n"
        "## Symptoms\n\n"
        "- The test fails when run on Tuesdays\n"
        "- The mock returns None instead of a value\n\n"
        "## Common causes\n\n"
        "- The fixture was not initialized\n"
        "- The test depends on execution order\n\n"
        "## Resolution\n\n"
        "1. Initialize all fixtures in setUp.\n"
        "2. Make each test independent.\n\n"
        "## Notes\n\n"
        "Flaky tests should be quarantined until fixed."
    ),
    "related_docs": ["DOC-TEST-002"],
    "last_reviewed_days_ago": 5,
}

MINIMAL_DOC = {
    "doc_id": "DOC-MIN-001",
    "title": "Minimal doc",
    "category": "minimal",
    "content": "Just some plain text without any headers or sections.",
}


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------


class TestChunkDocument:
    """Tests for _chunk_document — the section-based chunking logic."""

    def test_standard_doc_produces_four_sections(self):
        """A standard doc with Symptoms/Causes/Resolution/Notes → 4 chunks."""
        chunks = _chunk_document(SAMPLE_DOC)
        assert len(chunks) == 4
        section_names = [c["section_name"] for c in chunks]
        assert section_names == ["Symptoms", "Common causes", "Resolution", "Notes"]

    def test_each_chunk_has_required_metadata(self):
        """Every chunk carries doc_id, title, category, section_name, content."""
        chunks = _chunk_document(SAMPLE_DOC)
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "doc_id" in chunk
            assert "title" in chunk
            assert "category" in chunk
            assert "section_name" in chunk
            assert "content" in chunk
            assert chunk["doc_id"] == "DOC-TEST-001"
            assert chunk["title"] == "Test document for unit tests"
            assert chunk["category"] == "testing"

    def test_chunk_ids_are_unique(self):
        """Each chunk has a unique ID derived from doc_id + section_name."""
        chunks = _chunk_document(SAMPLE_DOC)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), f"Duplicate chunk IDs: {ids}"

    def test_chunk_content_excludes_header_line(self):
        """The ## header line itself is stripped from chunk content."""
        chunks = _chunk_document(SAMPLE_DOC)
        for chunk in chunks:
            assert not chunk["content"].startswith("## ")

    def test_preamble_is_skipped(self):
        """The title and 'Applies to' preamble don't become a chunk."""
        chunks = _chunk_document(SAMPLE_DOC)
        section_names = [c["section_name"] for c in chunks]
        assert "Overview" not in section_names  # No preamble chunk

    def test_minimal_doc_gets_single_chunk(self):
        """A doc with no ## headers produces a single 'Overview' chunk."""
        chunks = _chunk_document(MINIMAL_DOC)
        assert len(chunks) == 1
        assert chunks[0]["section_name"] == "Overview"
        assert "plain text" in chunks[0]["content"]

    def test_applies_to_is_preserved(self):
        """The applies_to metadata is carried through to chunks."""
        chunks = _chunk_document(SAMPLE_DOC)
        for chunk in chunks:
            assert chunk["applies_to"] == "All plans"

    def test_all_29_docs_produce_chunks(self):
        """Integration: all 29 real docs produce exactly 4 chunks each."""
        docs_path = Path("data/documentation.json")
        if not docs_path.exists():
            pytest.skip("documentation.json not available")

        with open(docs_path) as f:
            docs = json.load(f)

        assert len(docs) == 29
        total_chunks = 0
        for doc in docs:
            chunks = _chunk_document(doc)
            assert len(chunks) == 4, (
                f"{doc['doc_id']} produced {len(chunks)} chunks, expected 4"
            )
            total_chunks += len(chunks)

        assert total_chunks == 116


# ---------------------------------------------------------------------------
# RetrievedChunk model tests
# ---------------------------------------------------------------------------


class TestRetrievedChunk:
    """Tests for the RetrievedChunk Pydantic model."""

    def test_relevance_score_perfect_match(self):
        """Distance 0 → relevance 1.0."""
        chunk = RetrievedChunk(
            doc_id="DOC-TEST-001",
            title="Test",
            category="test",
            section_name="Symptoms",
            content="test content",
            similarity_score=0.0,
        )
        assert chunk.relevance_score == 1.0

    def test_relevance_score_moderate_match(self):
        """Distance 1.0 → relevance 0.5."""
        chunk = RetrievedChunk(
            doc_id="DOC-TEST-001",
            title="Test",
            category="test",
            section_name="Symptoms",
            content="test content",
            similarity_score=1.0,
        )
        assert chunk.relevance_score == 0.5

    def test_relevance_score_weak_match(self):
        """Distance 2.0 → relevance ~0.33."""
        chunk = RetrievedChunk(
            doc_id="DOC-TEST-001",
            title="Test",
            category="test",
            section_name="Symptoms",
            content="test content",
            similarity_score=2.0,
        )
        assert abs(chunk.relevance_score - 0.333) < 0.01


# ---------------------------------------------------------------------------
# RetrievalResult model tests
# ---------------------------------------------------------------------------


class TestRetrievalResult:
    """Tests for the RetrievalResult Pydantic model."""

    def test_empty_result(self):
        result = RetrievalResult(query="test", chunks=[])
        assert result.num_results == 0
        assert result.unique_doc_ids == []

    def test_deduplicates_doc_ids(self):
        """unique_doc_ids should deduplicate while preserving order."""
        chunks = [
            RetrievedChunk(
                doc_id="DOC-A",
                title="A",
                category="test",
                section_name="Symptoms",
                content="a",
                similarity_score=0.1,
            ),
            RetrievedChunk(
                doc_id="DOC-A",
                title="A",
                category="test",
                section_name="Resolution",
                content="b",
                similarity_score=0.2,
            ),
            RetrievedChunk(
                doc_id="DOC-B",
                title="B",
                category="test",
                section_name="Symptoms",
                content="c",
                similarity_score=0.3,
            ),
        ]
        result = RetrievalResult(query="test", chunks=chunks)
        assert result.num_results == 3
        assert result.unique_doc_ids == ["DOC-A", "DOC-B"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
