"""Customer memory tools — cross-request long-term memory for agents.

Agents write facts here during a request; triage reads them at the start of
subsequent requests for the same customer. This gives agents continuity across
sessions without embedding the entire history in the system prompt.

Triage calls get_customer_memories after identifying the customer.
Any agent can call save_customer_memory to persist an important fact.
"""
from ..database import get_db
from ..errors import NotFoundError
from ..models import Customer, CustomerMemory

_MAX_MEMORIES_PER_CUSTOMER = 20  # oldest are evicted when limit is reached


def register_tools(mcp) -> None:

    @mcp.tool()
    def get_customer_memories(customer_id: int, limit: int = 10) -> dict:
        """Retrieve long-term memory facts for a customer from prior support sessions.

        Call this after identifying the customer to load prior context: known
        workarounds, recurring issues, escalation history, preferences.

        Returns facts in reverse-chronological order (most recent first).
        """
        with get_db() as db:
            customer = db.get(Customer, customer_id)
            if not customer:
                raise NotFoundError(f"NOT_FOUND: Customer {customer_id} does not exist.")

            memories = (
                db.query(CustomerMemory)
                .filter(CustomerMemory.customer_id == customer_id)
                .order_by(CustomerMemory.created_at.desc())
                .limit(min(limit, _MAX_MEMORIES_PER_CUSTOMER))
                .all()
            )
            return {
                "customer_id": customer_id,
                "memories": [
                    {
                        "id": m.id,
                        "content": m.content,
                        "source_agent": m.source_agent,
                        "created_at": m.created_at.isoformat(),
                    }
                    for m in memories
                ],
                "count": len(memories),
            }

    @mcp.tool()
    def save_customer_memory(
        customer_id: int,
        content: str,
        source_agent: str,
    ) -> dict:
        """Save a long-term memory fact about a customer for future sessions.

        Use this to persist facts that should inform future agents:
        - Known workarounds that worked for this customer
        - Recurring issue patterns
        - Escalation decisions and their outcomes
        - Customer preferences or constraints

        Keep content concise (1-2 sentences). Avoid storing ticket IDs as facts
        — those are ephemeral. Store patterns and outcomes instead.
        """
        if not content.strip():
            return {"saved": False, "reason": "empty content"}

        with get_db() as db:
            customer = db.get(Customer, customer_id)
            if not customer:
                raise NotFoundError(f"NOT_FOUND: Customer {customer_id} does not exist.")

            # Evict oldest memories if at limit
            count = db.query(CustomerMemory).filter(
                CustomerMemory.customer_id == customer_id
            ).count()
            if count >= _MAX_MEMORIES_PER_CUSTOMER:
                oldest = (
                    db.query(CustomerMemory)
                    .filter(CustomerMemory.customer_id == customer_id)
                    .order_by(CustomerMemory.created_at.asc())
                    .first()
                )
                if oldest:
                    db.delete(oldest)

            memory = CustomerMemory(
                customer_id=customer_id,
                content=content.strip(),
                source_agent=source_agent,
                token_count=len(content.split()),  # rough estimate
            )
            db.add(memory)
            db.flush()
            return {"saved": True, "memory_id": memory.id}
