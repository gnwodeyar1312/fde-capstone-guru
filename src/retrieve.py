"""
Retrieve module for CloudServe Support System.

This is the THIRD stage of the pipeline: Ingest → Classify → **Retrieve** → Route → Generate → Validate

Purpose:
    Given a support ticket, find the most relevant documentation passages
    that could help answer the customer's question.

Design decisions:
    1. We embed SECTIONS of documents, not whole documents.
       Why? Each doc is ~200 words (~270 tokens). all-MiniLM-L6-v2 works
       best under 256 tokens. Splitting by section (Symptoms, Common causes,
       Resolution, Notes) gives ~50-80 word chunks that are well within the
       sweet spot AND more precise for matching.
    2. We use ChromaDB as the vector store.
       Why? It's embedded (no server to run), persistent (survives restarts),
       and has a simple API. For 29 docs with ~4 sections each (~116 chunks),
       it's the right tool. We don't need Pinecone or Weaviate at this scale.
    3. The embedding model (all-MiniLM-L6-v2) runs locally via sentence-transformers.
       Why not an API-based embedding? Free, fast, no rate limits, no API key
       needed. The model is 80MB — tiny. And it runs offline, which matters
       for the unattended evaluation run (THE GATE).
    4. We store rich metadata with each chunk: doc_id, title, category,
       section_name, applies_to. This lets us trace back from a retrieved
       chunk to the full source article, which is critical for citations
       in the generation stage.
    5. The build step is IDEMPOTENT. If the collection already exists, we
       delete it and rebuild from scratch. This is fine for 29 docs — it
       takes under 5 seconds. In production, you'd do incremental updates.

Interview context:
    "Why not just do keyword search?"
    → Keyword search fails on vocabulary mismatch. A customer writes
      "my deployment keeps dying" — keyword search looks for "dying" in
      the docs and finds nothing. Semantic search matches it to
      "Container deployments failing during the health check phase"
      because the MEANING is similar, even though no words overlap.
    "Why not use a larger embedding model?"
    → Diminishing returns at this scale. 29 docs, ~116 chunks. A larger
      model won't improve retrieval meaningfully but will slow down
      embedding and increase memory usage. Start simple, measure, upgrade
      if needed.
    "What about hybrid search (semantic + keyword)?"
    → Good idea for v2. BM25 + semantic reranking catches cases where
      exact terms matter (error codes, product names). Not needed for MVP
      with 29 docs, but would be for 2,900 docs.
"""

import json
import logging
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pydantic import BaseModel, Field

from src.config import CHROMA_PATH, EMBEDDING_MODEL, RETRIEVAL_TOP_K

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for retrieval output — the CONTRACT
# ---------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    """A single chunk retrieved from the vector store."""

    doc_id: str
    title: str
    category: str
    section_name: str
    content: str
    similarity_score: float = Field(ge=0.0, le=2.0)  # ChromaDB L2 distance
    applies_to: str = ""

    @property
    def relevance_score(self) -> float:
        """
        Convert L2 distance to a 0-1 relevance score.

        ChromaDB returns L2 (Euclidean) distance by default.
        Lower distance = more similar. We convert to a score where
        higher = more relevant, for easier reasoning downstream.

        The formula: score = 1 / (1 + distance)
        - distance 0.0 → score 1.0 (identical)
        - distance 1.0 → score 0.5 (moderate match)
        - distance 2.0 → score 0.33 (weak match)
        """
        return 1.0 / (1.0 + self.similarity_score)


class RetrievalResult(BaseModel):
    """The output of the retrieve stage."""

    query: str
    chunks: list[RetrievedChunk]
    num_results: int = 0
    unique_doc_ids: list[str] = []

    def model_post_init(self, __context, /):
        self.num_results = len(self.chunks)
        self.unique_doc_ids = list(dict.fromkeys(c.doc_id for c in self.chunks))


# ---------------------------------------------------------------------------
# Document chunking — split docs into sections
# ---------------------------------------------------------------------------


def _chunk_document(doc: dict) -> list[dict]:
    """
    Split a documentation article into section-level chunks.

    Each chunk gets the doc metadata (doc_id, title, category) plus
    the section name and content. This gives us precise matching
    while preserving traceability to the source.

    Chunking strategy:
        - Split on markdown ## headers
        - Each section becomes one chunk
        - If a doc has no sections, the whole content is one chunk
        - The title line and "Applies to" line are stripped (they're in metadata)

    Args:
        doc: A documentation article dict with doc_id, title, category, content, etc.

    Returns:
        List of chunk dicts ready for embedding and storage
    """
    content = doc["content"]
    doc_id = doc["doc_id"]
    title = doc["title"]
    category = doc["category"]
    applies_to = doc.get("applies_to", "")

    # Split on ## headers
    sections = re.split(r"\n(?=## )", content)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract section name from ## header
        header_match = re.match(r"^## (.+)", section)
        if header_match:
            section_name = header_match.group(1).strip()
            # Remove the header line from content
            section_content = section[header_match.end() :].strip()
        else:
            # This is the preamble (title + applies_to) — skip or use as "overview"
            # Skip lines that are just the title or "Applies to:" metadata
            lines = section.split("\n")
            meaningful_lines = [
                line
                for line in lines
                if line.strip()
                and not line.startswith("# ")
                and not line.startswith("**Applies to:")
            ]
            if not meaningful_lines:
                continue
            section_name = "Overview"
            section_content = "\n".join(meaningful_lines).strip()

        if not section_content:
            continue

        chunk_id = f"{doc_id}_{section_name.lower().replace(' ', '_')}"

        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "title": title,
                "category": category,
                "applies_to": applies_to,
                "section_name": section_name,
                "content": section_content,
            }
        )

    # Fallback: if no sections were extracted, use the whole content
    if not chunks:
        chunks.append(
            {
                "chunk_id": f"{doc_id}_full",
                "doc_id": doc_id,
                "title": title,
                "category": category,
                "applies_to": applies_to,
                "section_name": "Full document",
                "content": content,
            }
        )

    return chunks


# ---------------------------------------------------------------------------
# Vector store management
# ---------------------------------------------------------------------------

COLLECTION_NAME = "cloudserve_docs"


def _get_chroma_client() -> chromadb.ClientAPI:
    """Create a persistent ChromaDB client."""
    persist_dir = str(Path(CHROMA_PATH).resolve())
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def _get_embedding_function():
    """
    Get the sentence-transformer embedding function for ChromaDB.

    Uses all-MiniLM-L6-v2 by default (configurable via EMBEDDING_MODEL).
    This runs LOCALLY — no API call, no rate limit, no cost.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def build_vector_store(docs_path: str = "data/documentation.json") -> int:
    """
    Load documentation, chunk it, embed it, and store in ChromaDB.

    This is the B-03 build step. It is IDEMPOTENT — safe to run
    multiple times. Each run rebuilds the collection from scratch.

    Args:
        docs_path: Path to the documentation JSON file

    Returns:
        Number of chunks stored

    Raises:
        FileNotFoundError: If the docs file doesn't exist
    """
    # Load documentation
    docs_file = Path(docs_path)
    if not docs_file.exists():
        raise FileNotFoundError(f"Documentation file not found: {docs_path}")

    with open(docs_file) as f:
        docs = json.load(f)

    logger.info("Loaded %d documentation articles from %s", len(docs), docs_path)

    # Chunk all documents
    all_chunks = []
    for doc in docs:
        chunks = _chunk_document(doc)
        all_chunks.extend(chunks)
        logger.debug("  %s → %d chunks", doc["doc_id"], len(chunks))

    logger.info("Created %d chunks from %d documents", len(all_chunks), len(docs))

    # Initialize ChromaDB
    client = _get_chroma_client()
    ef = _get_embedding_function()

    # Delete existing collection if it exists (idempotent rebuild)
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted existing collection '%s'", COLLECTION_NAME)
    except Exception:  # noqa: BLE001, S110
        pass  # Collection didn't exist — ValueError or NotFoundError depending on version

    # Create fresh collection
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "l2"},  # L2 distance (Euclidean)
    )

    # Add chunks to collection
    collection.add(
        ids=[c["chunk_id"] for c in all_chunks],
        documents=[c["content"] for c in all_chunks],
        metadatas=[
            {
                "doc_id": c["doc_id"],
                "title": c["title"],
                "category": c["category"],
                "section_name": c["section_name"],
                "applies_to": c["applies_to"],
            }
            for c in all_chunks
        ],
    )

    logger.info(
        "Stored %d chunks in ChromaDB collection '%s' at %s",
        len(all_chunks),
        COLLECTION_NAME,
        CHROMA_PATH,
    )

    return len(all_chunks)


# ---------------------------------------------------------------------------
# Core retrieval function
# ---------------------------------------------------------------------------


def retrieve_context(
    query: str,
    top_k: int | None = None,
    category_filter: str | None = None,
) -> RetrievalResult:
    """
    Retrieve the most relevant documentation chunks for a query.

    This is the core function called by the pipeline. It:
    1. Embeds the query using the same model as the documents
    2. Finds the top-k nearest chunks in ChromaDB
    3. Returns them as validated RetrievedChunk objects with metadata

    Args:
        query: The text to search for (typically ticket.combined_text())
        top_k: Number of chunks to retrieve (default: RETRIEVAL_TOP_K from config)
        category_filter: Optional — only retrieve from this doc category

    Returns:
        RetrievalResult with ranked chunks and metadata

    Raises:
        ValueError: If the vector store hasn't been built yet
    """
    if top_k is None:
        top_k = RETRIEVAL_TOP_K

    client = _get_chroma_client()
    ef = _get_embedding_function()

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
        )
    except ValueError:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' not found. "
            "Run build_vector_store() first to index the documentation."
        )

    # Build query parameters
    query_params = {
        "query_texts": [query],
        "n_results": top_k,
    }

    # Optional category filter
    if category_filter:
        query_params["where"] = {"category": category_filter}

    # Query the collection
    results = collection.query(**query_params)

    # Parse results into RetrievedChunk objects
    chunks = []
    if results and results["ids"] and results["ids"][0]:
        for i, chunk_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            chunks.append(
                RetrievedChunk(
                    doc_id=meta["doc_id"],
                    title=meta["title"],
                    category=meta["category"],
                    section_name=meta["section_name"],
                    content=results["documents"][0][i],
                    similarity_score=results["distances"][0][i],
                    applies_to=meta.get("applies_to", ""),
                )
            )

    result = RetrievalResult(query=query, chunks=chunks)

    logger.info(
        "Retrieved %d chunks for query (%.50s...), from %d unique docs: %s",
        result.num_results,
        query,
        len(result.unique_doc_ids),
        result.unique_doc_ids,
    )

    return result


# ---------------------------------------------------------------------------
# Convenience: retrieve for a ticket
# ---------------------------------------------------------------------------


def retrieve_for_ticket(ticket, top_k: int | None = None) -> RetrievalResult:
    """
    Retrieve relevant documentation for a StandardTicket.

    Convenience wrapper that extracts the combined text from a ticket
    and calls retrieve_context().

    Args:
        ticket: A StandardTicket from the ingest stage
        top_k: Number of chunks to retrieve

    Returns:
        RetrievalResult with ranked chunks
    """
    query = ticket.combined_text()
    return retrieve_context(query, top_k=top_k)
