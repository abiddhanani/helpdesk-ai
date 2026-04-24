"""Retrieval quality evaluation: NDCG@K, Recall@K, MRR.

Compares BM25-only vs hybrid (BM25 + pgvector RRF) retrieval on the knowledge base.
Labels are auto-generated from the seed data: each article's title and tags are used
as natural queries, with the source article and tag-similar articles as relevant.

Usage (from exercise-4/):
    uv run python -m src.retrieval_eval

Requires:
    - PostgreSQL running with seed data loaded (exercise-1/seed_data.py)
    - EMBEDDING_API_KEY set for hybrid search (optional — BM25 comparison still runs)
    - DATABASE_URL set
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mcp-server"))

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://helpdesk:helpdesk@localhost:5432/helpdesk"
)


def ndcg_at_k(relevant_ids: list[int], retrieved_ids: list[int], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, doc_id in enumerate(retrieved_ids[:k])
        if doc_id in relevant_ids
    )
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(relevant_ids: list[int], retrieved_ids: list[int], k: int) -> float:
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def mrr(relevant_ids: list[int], retrieved_ids: list[int]) -> float:
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def generate_labels(db) -> list[dict]:
    """Auto-generate (query, relevant_doc_ids) pairs from knowledge articles.

    Strategy: use article title as the query; mark the article itself + any article
    sharing at least one tag as relevant. This tests that semantically similar articles
    surface together.
    """
    from src.models import KnowledgeArticle

    articles = db.query(KnowledgeArticle).all()
    tag_to_ids: dict[str, list[int]] = {}
    for article in articles:
        for tag in [t.strip().lower() for t in article.tags.split(",") if t.strip()]:
            tag_to_ids.setdefault(tag, []).append(article.id)

    labels = []
    for article in articles:
        tags = [t.strip().lower() for t in article.tags.split(",") if t.strip()]
        relevant: set[int] = {article.id}
        for tag in tags:
            relevant.update(tag_to_ids.get(tag, []))
        labels.append({
            "query": article.title,
            "relevant_ids": sorted(relevant),
            "source_id": article.id,
        })
    return labels


def bm25_search(db, query: str, limit: int = 20) -> list[int]:
    from sqlalchemy import func
    from src.models import KnowledgeArticle

    results = (
        db.query(KnowledgeArticle)
        .filter(KnowledgeArticle.search_vector.op("@@")(func.plainto_tsquery("english", query)))
        .order_by(func.ts_rank_cd(KnowledgeArticle.search_vector, func.plainto_tsquery("english", query)).desc())
        .limit(limit)
        .all()
    )
    return [r.id for r in results]


def hybrid_search(db, query: str, limit: int = 20) -> list[int]:
    from sqlalchemy import text
    from src.tools.embeddings import article_embed_text, embedding_to_pg

    embedding = article_embed_text(query, "", "")
    if embedding is None:
        return bm25_search(db, query, limit)

    embedding_str = embedding_to_pg(embedding)
    sql = text("""
        WITH bm25 AS (
            SELECT id,
                ROW_NUMBER() OVER (
                    ORDER BY ts_rank_cd(search_vector, plainto_tsquery('english', :query)) DESC
                ) AS bm25_rank
            FROM knowledge_articles
            WHERE search_vector @@ plainto_tsquery('english', :query)
            LIMIT 50
        ),
        vector_search AS (
            SELECT id,
                ROW_NUMBER() OVER (
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                ) AS vector_rank
            FROM knowledge_articles
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT 50
        ),
        rrf AS (
            SELECT COALESCE(b.id, v.id) AS id,
                COALESCE(0.5 / (60 + COALESCE(b.bm25_rank, 100)::float), 0)
                + COALESCE(0.5 / (60 + COALESCE(v.vector_rank, 100)::float), 0) AS rrf_score
            FROM bm25 b FULL OUTER JOIN vector_search v ON b.id = v.id
        )
        SELECT id FROM rrf ORDER BY rrf_score DESC LIMIT :limit
    """)
    rows = db.execute(sql, {"query": query, "embedding": embedding_str, "limit": limit}).fetchall()
    return [r.id for r in rows]


def evaluate(db, labels: list[dict], k5: int = 5, k10: int = 10) -> dict:
    bm25_ndcg5, bm25_recall10, bm25_mrr_vals = [], [], []
    hybrid_ndcg5, hybrid_recall10, hybrid_mrr_vals = [], [], []
    hybrid_available = True

    for label in labels:
        query = label["query"]
        relevant = label["relevant_ids"]

        bm25_ids = bm25_search(db, query, limit=k10)
        hybrid_ids = hybrid_search(db, query, limit=k10)
        if hybrid_ids == bm25_ids or not any(i is not None for i in hybrid_ids):
            hybrid_available = False

        bm25_ndcg5.append(ndcg_at_k(relevant, bm25_ids, k5))
        bm25_recall10.append(recall_at_k(relevant, bm25_ids, k10))
        bm25_mrr_vals.append(mrr(relevant, bm25_ids))

        hybrid_ndcg5.append(ndcg_at_k(relevant, hybrid_ids, k5))
        hybrid_recall10.append(recall_at_k(relevant, hybrid_ids, k10))
        hybrid_mrr_vals.append(mrr(relevant, hybrid_ids))

    def avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    bm25_ndcg = avg(bm25_ndcg5)
    hybrid_ndcg = avg(hybrid_ndcg5)
    improvement = ((hybrid_ndcg - bm25_ndcg) / bm25_ndcg * 100) if bm25_ndcg > 0 else 0.0

    return {
        "queries_evaluated": len(labels),
        "hybrid_embeddings_available": hybrid_available,
        "bm25_ndcg_at_5": round(bm25_ndcg, 4),
        "hybrid_ndcg_at_5": round(hybrid_ndcg, 4),
        "bm25_recall_at_10": round(avg(bm25_recall10), 4),
        "hybrid_recall_at_10": round(avg(hybrid_recall10), 4),
        "bm25_mrr": round(avg(bm25_mrr_vals), 4),
        "hybrid_mrr": round(avg(hybrid_mrr_vals), 4),
        "ndcg_improvement_pct": round(improvement, 1),
    }


def main() -> None:
    from src.database import SessionLocal

    with SessionLocal() as db:
        print("Generating eval labels from seed data...", file=sys.stderr)
        labels = generate_labels(db)
        print(f"Generated {len(labels)} query-label pairs.", file=sys.stderr)

        print("Running retrieval eval...", file=sys.stderr)
        results = evaluate(db, labels)

    print(json.dumps(results, indent=2))

    if not results["hybrid_embeddings_available"]:
        print(
            "\nNote: hybrid search fell back to BM25 (EMBEDDING_API_KEY not set "
            "or no embeddings in DB). Run seed_data.py with embeddings enabled first.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
