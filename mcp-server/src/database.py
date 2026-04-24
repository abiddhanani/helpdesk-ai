import os
import sys
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import Index, create_engine, text
from sqlalchemy.orm import sessionmaker

from .models import Base, CustomerMemory, KnowledgeArticle, Ticket

load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://helpdesk:helpdesk@localhost:5432/helpdesk",
)

engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=5,
    pool_timeout=30,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Create tables and apply full-text search DDL."""
    Base.metadata.create_all(engine)

    with engine.connect() as conn:
        # Generated tsvector column for tickets
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='tickets' AND column_name='search_vector'
                ) THEN
                    ALTER TABLE tickets
                    ADD COLUMN search_vector tsvector
                    GENERATED ALWAYS AS (
                        to_tsvector('english',
                            coalesce(title, '') || ' ' || coalesce(description, ''))
                    ) STORED;
                END IF;
            END $$;
        """))

        # Generated tsvector column for knowledge_articles
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='knowledge_articles' AND column_name='search_vector'
                ) THEN
                    ALTER TABLE knowledge_articles
                    ADD COLUMN search_vector tsvector
                    GENERATED ALWAYS AS (
                        to_tsvector('english',
                            coalesce(title, '') || ' ' ||
                            coalesce(content, '') || ' ' ||
                            coalesce(tags, ''))
                    ) STORED;
                END IF;
            END $$;
        """))

        # GIN indexes for full-text search
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_tickets_search ON tickets USING GIN(search_vector);"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_kb_search ON knowledge_articles USING GIN(search_vector);"
        ))

        # Composite indexes for common filter patterns
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tickets_priority ON tickets(priority);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tickets_agent ON tickets(assigned_agent_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tickets_customer ON tickets(customer_id);"))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_tickets_sla ON tickets(sla_deadline) "
            "WHERE status NOT IN ('resolved', 'closed');"
        ))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_comments_ticket ON ticket_comments(ticket_id);"))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_memories_customer ON customer_memories(customer_id, created_at DESC);"
        ))

        # pgvector: semantic embedding columns and HNSW indexes
        # HNSW chosen over IVFFlat: better recall for dynamic insert workloads, no training step.
        # m=16 (connectivity), ef_construction=64 (build quality), cosine metric.
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.execute(text("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='tickets' AND column_name='embedding'
                ) THEN
                    ALTER TABLE tickets ADD COLUMN embedding vector(1536);
                END IF;
            END $$;
        """))
        conn.execute(text("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='knowledge_articles' AND column_name='embedding'
                ) THEN
                    ALTER TABLE knowledge_articles ADD COLUMN embedding vector(1536);
                END IF;
            END $$;
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tickets_embedding_hnsw
            ON tickets USING hnsw (embedding vector_cosine_ops)
            WITH (m=16, ef_construction=64);
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_kb_embedding_hnsw
            ON knowledge_articles USING hnsw (embedding vector_cosine_ops)
            WITH (m=16, ef_construction=64);
        """))

        conn.commit()
    print("Database initialised.", file=sys.stderr)
