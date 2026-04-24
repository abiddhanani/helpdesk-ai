from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    Computed,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Agent(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, unique=True)
    specialty = Column(String(50), nullable=False)  # hardware/software/network/security
    is_available = Column(Boolean, nullable=False, default=True)
    max_tickets = Column(Integer, nullable=False, default=10)

    tickets = relationship("Ticket", back_populates="assigned_agent", foreign_keys="Ticket.assigned_agent_id")

    def __repr__(self) -> str:
        return f"<Agent id={self.id} name={self.name!r}>"


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, unique=True)
    department = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    tickets = relationship("Ticket", back_populates="customer")

    def __repr__(self) -> str:
        return f"<Customer id={self.id} name={self.name!r}>"


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String(50), nullable=False, default="open")
    # open / in_progress / waiting_on_customer / resolved / closed
    priority = Column(String(20), nullable=False)
    # low / medium / high / critical
    category = Column(String(50), nullable=False)
    # hardware / software / network / access / other
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    assigned_agent_id = Column(Integer, ForeignKey("agents.id", ondelete="SET NULL"), nullable=True)
    sla_deadline = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    version = Column(Integer, nullable=False, default=1)

    search_vector = Column(TSVECTOR, Computed(
        "to_tsvector('english', coalesce(title, '') || ' ' || coalesce(description, ''))",
        persisted=True,
    ))

    customer = relationship("Customer", back_populates="tickets")
    assigned_agent = relationship("Agent", back_populates="tickets", foreign_keys=[assigned_agent_id])
    comments = relationship("TicketComment", back_populates="ticket", order_by="TicketComment.created_at")

    def __repr__(self) -> str:
        return f"<Ticket id={self.id} priority={self.priority!r} status={self.status!r}>"


class KnowledgeArticle(Base):
    __tablename__ = "knowledge_articles"

    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(50), nullable=False)
    tags = Column(String(500), nullable=False, default="")  # comma-separated
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    search_vector = Column(TSVECTOR, Computed(
        "to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, '') || ' ' || coalesce(tags, ''))",
        persisted=True,
    ))

    def __repr__(self) -> str:
        return f"<KnowledgeArticle id={self.id} title={self.title!r}>"


class CustomerMemory(Base):
    """Long-term memory facts persisted across requests, keyed by customer_id.

    Agents write here to remember important context (workarounds, preferences,
    recurring issues). Triage reads these at the start of each new request.
    """
    __tablename__ = "customer_memories"

    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey("customers.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    source_agent = Column(String(50), nullable=False)
    token_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class TicketComment(Base):
    __tablename__ = "ticket_comments"

    id = Column(Integer, primary_key=True)
    ticket_id = Column(Integer, ForeignKey("tickets.id", ondelete="CASCADE"), nullable=False)
    author_type = Column(String(20), nullable=False)  # agent / customer / system
    author_id = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    ticket = relationship("Ticket", back_populates="comments")

    def __repr__(self) -> str:
        return f"<TicketComment id={self.id} ticket_id={self.ticket_id}>"
