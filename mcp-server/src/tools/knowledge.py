"""Knowledge base search tool — hybrid BM25 + vector similarity via RRF.

Search strategy:
  1. BM25 via existing tsvector GIN index (plainto_tsquery) — exact keyword match
  2. Cosine similarity via pgvector HNSW index — semantic match
  3. Reciprocal Rank Fusion (RRF, k=60) to merge both ranked lists
  4. Articles with no embedding fall back to BM25 rank only (backward-compatible)

If EMBEDDING_API_KEY is not set, falls back to BM25-only (original behaviour).
RRF score is included in results for eval/monitoring purposes.
"""
from sqlalchemy import func, text

from ..database import get_db
from ..errors import InvalidInputError
from ..models import KnowledgeArticle
from .embeddings import article_embed_text, embedding_to_pg

VALID_CATEGORIES = {"hardware", "software", "network", "access", "other"}

_RRF_K = 60
_RRF_BM25_WEIGHT = 0.5
_RRF_VECTOR_WEIGHT = 0.5


def _article_to_dict(article: KnowledgeArticle, bm25_rank: int | None = None, vector_rank: int | None = None, rrf_score: float | None = None) -> dict:
    d = {
        "id": article.id,
        "title": article.title,
        "content": article.content,
        "category": article.category,
        "tags": article.tags,
        "created_at": article.created_at.isoformat(),
        "updated_at": article.updated_at.isoformat(),
    }
    if rrf_score is not None:
        d["bm25_rank"] = bm25_rank
        d["vector_rank"] = vector_rank
        d["rrf_score"] = round(rrf_score, 6)
    return d


def _hybrid_search(db, keyword: str, category: str | None, tags: str | None) -> list[dict]:
    """Hybrid BM25 + cosine similarity search with RRF merge."""
    embedding = article_embed_text(keyword, "", "")
    if embedding is None:
        return _bm25_search(db, keyword, category, tags)

    embedding_str = embedding_to_pg(embedding)
    category_filter = "AND category = :category" if category else ""
    tag_filters = ""
    tag_params: dict = {}
    if tags:
        tag_list = [t.strip().lower() for t in tags.split(",") if t.strip()]
        for i, tag in enumerate(tag_list):
            tag_filters += f" AND LOWER(tags) LIKE :tag_{i}"
            tag_params[f"tag_{i}"] = f"%{tag}%"

    sql = text(f"""
        WITH bm25 AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    ORDER BY ts_rank_cd(search_vector, plainto_tsquery('english', :keyword)) DESC
                ) AS bm25_rank
            FROM knowledge_articles
            WHERE search_vector @@ plainto_tsquery('english', :keyword)
            {category_filter} {tag_filters}
            LIMIT 50
        ),
        vector_search AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                ) AS vector_rank
            FROM knowledge_articles
            WHERE embedding IS NOT NULL
            {category_filter} {tag_filters}
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT 50
        ),
        rrf AS (
            SELECT
                COALESCE(b.id, v.id) AS id,
                b.bm25_rank,
                v.vector_rank,
                COALESCE(:bm25w / (:k + b.bm25_rank::float), 0)
                + COALESCE(:vecw / (:k + v.vector_rank::float), 0) AS rrf_score
            FROM bm25 b
            FULL OUTER JOIN vector_search v ON b.id = v.id
        )
        SELECT
            ka.id, ka.title, ka.content, ka.category, ka.tags,
            ka.created_at, ka.updated_at,
            r.bm25_rank, r.vector_rank, r.rrf_score
        FROM rrf r
        JOIN knowledge_articles ka ON ka.id = r.id
        ORDER BY r.rrf_score DESC
        LIMIT 20
    """)

    params = {
        "keyword": keyword,
        "embedding": embedding_str,
        "k": _RRF_K,
        "bm25w": _RRF_BM25_WEIGHT,
        "vecw": _RRF_VECTOR_WEIGHT,
        "category": category,
        **tag_params,
    }
    rows = db.execute(sql, params).fetchall()
    return [
        {
            "id": row.id,
            "title": row.title,
            "content": row.content,
            "category": row.category,
            "tags": row.tags,
            "created_at": row.created_at.isoformat(),
            "updated_at": row.updated_at.isoformat(),
            "bm25_rank": row.bm25_rank,
            "vector_rank": row.vector_rank,
            "rrf_score": round(float(row.rrf_score), 6),
        }
        for row in rows
    ]


def _bm25_search(db, keyword: str, category: str | None, tags: str | None) -> list[dict]:
    """BM25-only fallback when embeddings are unavailable."""
    query = db.query(KnowledgeArticle)
    query = query.filter(
        KnowledgeArticle.search_vector.op("@@")(
            func.plainto_tsquery("english", keyword)
        )
    )
    if category:
        query = query.filter(KnowledgeArticle.category == category)
    if tags:
        for tag in [t.strip().lower() for t in tags.split(",") if t.strip()]:
            query = query.filter(func.lower(KnowledgeArticle.tags).contains(tag))
    articles = query.order_by(KnowledgeArticle.updated_at.desc()).limit(20).all()
    return [_article_to_dict(a) for a in articles]


def register_tools(mcp) -> None:

    @mcp.tool()
    def search_knowledge_base(
        keyword: str | None = None,
        category: str | None = None,
        tags: str | None = None,
    ) -> dict:
        """Search the knowledge base for articles relevant to an IT issue.

        Uses hybrid BM25 + semantic vector search (RRF merge) when embeddings are
        available, falling back to keyword-only search otherwise.

        - keyword: search across article title, content, and tags
        - category: filter by category (hardware/software/network/access/other)
        - tags: comma-separated tag terms

        Returns articles with optional rrf_score for retrieval quality monitoring.
        """
        if category and category not in VALID_CATEGORIES:
            raise InvalidInputError(
                f"INVALID_INPUT: category must be one of {sorted(VALID_CATEGORIES)}"
            )

        with get_db() as db:
            if keyword:
                articles = _hybrid_search(db, keyword, category, tags)
            else:
                # No keyword: list by category/tags only (no ranking needed)
                query = db.query(KnowledgeArticle)
                if category:
                    query = query.filter(KnowledgeArticle.category == category)
                if tags:
                    for tag in [t.strip().lower() for t in tags.split(",") if t.strip()]:
                        query = query.filter(func.lower(KnowledgeArticle.tags).contains(tag))
                rows = query.order_by(KnowledgeArticle.updated_at.desc()).all()
                articles = [_article_to_dict(a) for a in rows]

            return {"articles": articles, "count": len(articles)}
