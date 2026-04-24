import threading
from datetime import datetime, timedelta
from typing import Literal
from sqlalchemy import func, text

from ..database import SessionLocal, get_db
from ..errors import BusinessRuleError, InvalidInputError, NotFoundError
from ..models import Agent, Customer, Ticket, TicketComment
from .embeddings import embedding_to_pg, ticket_embed_text

SLA_HOURS: dict[str, timedelta] = {
    "critical": timedelta(hours=4),
    "high": timedelta(hours=8),
    "medium": timedelta(hours=24),
    "low": timedelta(hours=72),
}

VALID_STATUSES = {"open", "in_progress", "waiting_on_customer", "resolved", "closed"}
VALID_PRIORITIES = {"low", "medium", "high", "critical"}
VALID_CATEGORIES = {"hardware", "software", "network", "access", "other"}
VALID_AUTHOR_TYPES = {"agent", "customer", "system"}


def _ticket_to_dict(ticket: Ticket) -> dict:
    return {
        "id": ticket.id,
        "title": ticket.title,
        "description": ticket.description,
        "status": ticket.status,
        "priority": ticket.priority,
        "category": ticket.category,
        "customer_id": ticket.customer_id,
        "assigned_agent_id": ticket.assigned_agent_id,
        "sla_deadline": ticket.sla_deadline.isoformat() if ticket.sla_deadline else None,
        "created_at": ticket.created_at.isoformat(),
        "updated_at": ticket.updated_at.isoformat(),
        "version": ticket.version,
    }


def _comment_to_dict(comment: TicketComment) -> dict:
    return {
        "id": comment.id,
        "ticket_id": comment.ticket_id,
        "author_type": comment.author_type,
        "author_id": comment.author_id,
        "content": comment.content,
        "created_at": comment.created_at.isoformat(),
    }


def _embed_ticket_async(ticket_id: int, title: str, description: str) -> None:
    """Fire-and-forget: generate embedding and persist it in a background thread.

    The ticket is returned to the caller immediately. The embedding column is
    updated seconds later. Hybrid search falls back to BM25 for un-embedded rows.
    """
    def _work() -> None:
        embedding = ticket_embed_text(title, description)
        if embedding is None:
            return
        try:
            db = SessionLocal()
            db.execute(
                text("UPDATE tickets SET embedding = CAST(:emb AS vector) WHERE id = :id"),
                {"emb": embedding_to_pg(embedding), "id": ticket_id},
            )
            db.commit()
            db.close()
        except Exception:
            pass

    threading.Thread(target=_work, daemon=True).start()


_RRF_K = 60


def _hybrid_ticket_search(
    db,
    keyword: str,
    status: str | None,
    priority: str | None,
    category: str | None,
    assigned_agent_id: int | None,
    customer_id: int | None,
    sla_overdue: bool | None,
) -> list[dict]:
    """Hybrid BM25 + cosine similarity ticket search with RRF merge.

    Falls back to BM25-only if embeddings are unavailable.
    """
    embedding = ticket_embed_text(keyword, "")
    filters: list[str] = []
    params: dict = {"keyword": keyword, "k": _RRF_K, "now": datetime.utcnow()}

    if status:
        filters.append("status = :status")
        params["status"] = status
    if priority:
        filters.append("priority = :priority")
        params["priority"] = priority
    if category:
        filters.append("category = :category")
        params["category"] = category
    if assigned_agent_id is not None:
        filters.append("assigned_agent_id = :agent_id")
        params["agent_id"] = assigned_agent_id
    if customer_id is not None:
        filters.append("customer_id = :customer_id")
        params["customer_id"] = customer_id
    if sla_overdue:
        filters.append("sla_deadline < :now AND status NOT IN ('resolved', 'closed')")

    where_clause = ("AND " + " AND ".join(filters)) if filters else ""

    if embedding is None:
        # BM25-only fallback
        sql = text(f"""
            SELECT id FROM tickets
            WHERE search_vector @@ plainto_tsquery('english', :keyword)
            {where_clause}
            ORDER BY ts_rank_cd(search_vector, plainto_tsquery('english', :keyword)) DESC
            LIMIT 20
        """)
        rows = db.execute(sql, params).fetchall()
        ticket_ids = [r.id for r in rows]
    else:
        params["embedding"] = embedding_to_pg(embedding)
        sql = text(f"""
            WITH bm25 AS (
                SELECT id,
                    ROW_NUMBER() OVER (
                        ORDER BY ts_rank_cd(search_vector, plainto_tsquery('english', :keyword)) DESC
                    ) AS bm25_rank
                FROM tickets
                WHERE search_vector @@ plainto_tsquery('english', :keyword) {where_clause}
                LIMIT 50
            ),
            vector_search AS (
                SELECT id,
                    ROW_NUMBER() OVER (
                        ORDER BY embedding <=> CAST(:embedding AS vector)
                    ) AS vector_rank
                FROM tickets
                WHERE embedding IS NOT NULL {where_clause}
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT 50
            ),
            rrf AS (
                SELECT COALESCE(b.id, v.id) AS id,
                    COALESCE(0.5 / (:k + COALESCE(b.bm25_rank, 100)::float), 0)
                    + COALESCE(0.5 / (:k + COALESCE(v.vector_rank, 100)::float), 0) AS rrf_score
                FROM bm25 b FULL OUTER JOIN vector_search v ON b.id = v.id
            )
            SELECT id FROM rrf ORDER BY rrf_score DESC LIMIT 20
        """)
        rows = db.execute(sql, params).fetchall()
        ticket_ids = [r.id for r in rows]

    if not ticket_ids:
        return []

    tickets = db.query(Ticket).filter(Ticket.id.in_(ticket_ids)).all()
    id_to_ticket = {t.id: t for t in tickets}
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    sorted_tickets = sorted(
        [id_to_ticket[i] for i in ticket_ids if i in id_to_ticket],
        key=lambda t: (priority_order.get(t.priority, 9), t.sla_deadline or datetime.max),
    )
    return [_ticket_to_dict(t) for t in sorted_tickets]


def register_tools(mcp) -> None:

    @mcp.tool()
    def search_tickets(
        status: str | None = None,
        priority: str | None = None,
        category: str | None = None,
        assigned_agent_id: int | None = None,
        customer_id: int | None = None,
        keyword: str | None = None,
        sla_overdue: bool | None = None,
    ) -> dict:
        """Search help desk tickets with optional filters.

        All parameters are optional and combinable. Returns a list of matching tickets
        sorted by priority (critical first) then by SLA deadline.

        - keyword: full-text search across ticket title and description
        - sla_overdue: when True, returns only open/in-progress tickets past their SLA deadline
        """
        if status and status not in VALID_STATUSES:
            raise InvalidInputError(f"INVALID_INPUT: status must be one of {sorted(VALID_STATUSES)}")
        if priority and priority not in VALID_PRIORITIES:
            raise InvalidInputError(f"INVALID_INPUT: priority must be one of {sorted(VALID_PRIORITIES)}")
        if category and category not in VALID_CATEGORIES:
            raise InvalidInputError(f"INVALID_INPUT: category must be one of {sorted(VALID_CATEGORIES)}")

        with get_db() as db:
            if keyword:
                tickets = _hybrid_ticket_search(
                    db, keyword, status, priority, category, assigned_agent_id, customer_id, sla_overdue
                )
                return {"tickets": tickets, "count": len(tickets)}

            query = db.query(Ticket)
            if status:
                query = query.filter(Ticket.status == status)
            if priority:
                query = query.filter(Ticket.priority == priority)
            if category:
                query = query.filter(Ticket.category == category)
            if assigned_agent_id is not None:
                query = query.filter(Ticket.assigned_agent_id == assigned_agent_id)
            if customer_id is not None:
                query = query.filter(Ticket.customer_id == customer_id)
            if sla_overdue:
                query = query.filter(
                    Ticket.sla_deadline < datetime.utcnow(),
                    Ticket.status.notin_(["resolved", "closed"]),
                )

            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            tickets = query.all()
            tickets.sort(key=lambda t: (priority_order.get(t.priority, 9), t.sla_deadline or datetime.max))
            return {"tickets": [_ticket_to_dict(t) for t in tickets], "count": len(tickets)}

    @mcp.tool()
    def get_ticket_details(ticket_id: int) -> dict:
        """Get full details for a ticket including all comments.

        Returns the ticket fields plus a list of comments in chronological order.
        """
        with get_db() as db:
            ticket = db.get(Ticket, ticket_id)
            if not ticket:
                raise NotFoundError(f"NOT_FOUND: Ticket {ticket_id} does not exist.")

            ticket_dict = _ticket_to_dict(ticket)
            ticket_dict["comments"] = [_comment_to_dict(c) for c in ticket.comments]
            return ticket_dict

    @mcp.tool()
    def create_ticket(
        title: str,
        description: str,
        priority: Literal["low", "medium", "high", "critical"],
        category: Literal["hardware", "software", "network", "access", "other"],
        customer_id: int,
    ) -> dict:
        """Create a new support ticket.

        SLA deadline is automatically calculated from priority:
        critical=4h, high=8h, medium=24h, low=72h from now.
        Returns the created ticket.
        """
        if not title.strip():
            raise InvalidInputError("INVALID_INPUT: title cannot be empty.")
        if not description.strip():
            raise InvalidInputError("INVALID_INPUT: description cannot be empty.")

        with get_db() as db:
            customer = db.get(Customer, customer_id)
            if not customer:
                raise NotFoundError(f"NOT_FOUND: Customer {customer_id} does not exist.")

            now = datetime.utcnow()
            ticket = Ticket(
                title=title.strip(),
                description=description.strip(),
                status="open",
                priority=priority,
                category=category,
                customer_id=customer_id,
                sla_deadline=now + SLA_HOURS[priority],
                created_at=now,
                updated_at=now,
            )
            db.add(ticket)
            db.flush()  # get the id before commit
            result = _ticket_to_dict(ticket)

        # Schedule background embedding after commit (non-blocking)
        _embed_ticket_async(result["id"], title.strip(), description.strip())
        return result

    @mcp.tool()
    def update_ticket(
        ticket_id: int,
        title: str | None = None,
        description: str | None = None,
        status: Literal["open", "in_progress", "waiting_on_customer", "resolved", "closed"] | None = None,
        priority: Literal["low", "medium", "high", "critical"] | None = None,
        category: Literal["hardware", "software", "network", "access", "other"] | None = None,
        sla_deadline: str | None = None,
    ) -> dict:
        """Update one or more fields on an existing ticket.

        Only provide fields you want to change. sla_deadline must be an ISO 8601
        datetime string if provided (e.g. '2025-12-31T23:59:00').
        Returns the updated ticket.
        """
        content_changed = title is not None or description is not None
        with get_db() as db:
            ticket = db.get(Ticket, ticket_id)
            if not ticket:
                raise NotFoundError(f"NOT_FOUND: Ticket {ticket_id} does not exist.")

            if title is not None:
                if not title.strip():
                    raise InvalidInputError("INVALID_INPUT: title cannot be empty.")
                ticket.title = title.strip()
            if description is not None:
                ticket.description = description.strip()
            if status is not None:
                ticket.status = status
            if priority is not None:
                ticket.priority = priority
            if category is not None:
                ticket.category = category
            if sla_deadline is not None:
                try:
                    ticket.sla_deadline = datetime.fromisoformat(sla_deadline)
                except ValueError:
                    raise InvalidInputError(
                        "INVALID_INPUT: sla_deadline must be ISO 8601 format, e.g. '2025-12-31T23:59:00'."
                    )

            ticket.updated_at = datetime.utcnow()
            ticket.version += 1
            result = _ticket_to_dict(ticket)

        # Re-embed in background only when searchable text changed
        if content_changed:
            _embed_ticket_async(
                result["id"],
                result["title"],
                result["description"],
            )
        return result

    @mcp.tool()
    def assign_ticket(ticket_id: int, agent_id: int) -> dict:
        """Assign or reassign a ticket to a support agent.

        Business rules enforced:
        - Agent must exist
        - Agent must be available (is_available=True)
        - Agent must not have reached their max_tickets limit (counting open/in-progress tickets)
        Returns the updated ticket.
        """
        with get_db() as db:
            ticket = db.get(Ticket, ticket_id)
            if not ticket:
                raise NotFoundError(f"NOT_FOUND: Ticket {ticket_id} does not exist.")

            agent = db.get(Agent, agent_id)
            if not agent:
                raise NotFoundError(f"NOT_FOUND: Agent {agent_id} does not exist.")

            if not agent.is_available:
                raise BusinessRuleError(
                    f"BUSINESS_RULE: Agent {agent.name} is not available for new assignments."
                )

            active_count = (
                db.query(func.count(Ticket.id))
                .filter(
                    Ticket.assigned_agent_id == agent_id,
                    Ticket.status.notin_(["resolved", "closed"]),
                )
                .scalar()
            )
            if active_count >= agent.max_tickets:
                raise BusinessRuleError(
                    f"BUSINESS_RULE: Agent {agent.name} is at capacity "
                    f"({active_count}/{agent.max_tickets} tickets)."
                )

            ticket.assigned_agent_id = agent_id
            ticket.updated_at = datetime.utcnow()
            ticket.version += 1

            if ticket.status == "open":
                ticket.status = "in_progress"

            return _ticket_to_dict(ticket)

    @mcp.tool()
    def add_comment(
        ticket_id: int,
        author_type: Literal["agent", "customer", "system"],
        author_id: int,
        content: str,
    ) -> dict:
        """Add a comment to an existing ticket.

        author_type must be 'agent', 'customer', or 'system'.
        author_id should be the id of the agent or customer; use 0 for system comments.
        Returns the created comment.
        """
        if not content.strip():
            raise InvalidInputError("INVALID_INPUT: comment content cannot be empty.")

        with get_db() as db:
            ticket = db.get(Ticket, ticket_id)
            if not ticket:
                raise NotFoundError(f"NOT_FOUND: Ticket {ticket_id} does not exist.")

            comment = TicketComment(
                ticket_id=ticket_id,
                author_type=author_type,
                author_id=author_id,
                content=content.strip(),
                created_at=datetime.utcnow(),
            )
            db.add(comment)
            db.flush()
            return _comment_to_dict(comment)
