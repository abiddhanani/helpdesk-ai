from datetime import datetime

from sqlalchemy import func

from ..database import get_db
from ..errors import NotFoundError
from ..models import Agent, Ticket

VALID_SPECIALTIES = {"hardware", "software", "network", "security"}


def _agent_to_dict(agent: Agent) -> dict:
    return {
        "id": agent.id,
        "name": agent.name,
        "email": agent.email,
        "specialty": agent.specialty,
        "is_available": agent.is_available,
        "max_tickets": agent.max_tickets,
    }


def register_tools(mcp) -> None:

    @mcp.tool()
    def get_agent_workload(agent_id: int) -> dict:
        """Get an agent's current active ticket count, capacity, and availability.

        Returns the agent details plus a breakdown of ticket counts by status.
        """
        with get_db() as db:
            agent = db.get(Agent, agent_id)
            if not agent:
                raise NotFoundError(f"NOT_FOUND: Agent {agent_id} does not exist.")

            rows = (
                db.query(Ticket.status, func.count(Ticket.id))
                .filter(Ticket.assigned_agent_id == agent_id)
                .group_by(Ticket.status)
                .all()
            )
            by_status = {row[0]: row[1] for row in rows}
            active_count = sum(
                v for k, v in by_status.items() if k not in ("resolved", "closed")
            )

            return {
                **_agent_to_dict(agent),
                "active_tickets": active_count,
                "tickets_by_status": by_status,
                "capacity_remaining": max(0, agent.max_tickets - active_count),
            }

    @mcp.tool()
    def list_agents(
        specialty: str | None = None,
        is_available: bool | None = None,
    ) -> dict:
        """List all support agents with optional filters.

        - specialty: filter by specialty (hardware/software/network/security)
        - is_available: when True, return only agents currently accepting new tickets

        Returns agents with their current active ticket count included.
        """
        with get_db() as db:
            query = db.query(Agent)

            if specialty:
                query = query.filter(Agent.specialty == specialty)
            if is_available is not None:
                query = query.filter(Agent.is_available == is_available)

            agents = query.order_by(Agent.name).all()

            # Attach live active ticket counts
            agent_ids = [a.id for a in agents]
            counts = (
                db.query(Ticket.assigned_agent_id, func.count(Ticket.id))
                .filter(
                    Ticket.assigned_agent_id.in_(agent_ids),
                    Ticket.status.notin_(["resolved", "closed"]),
                )
                .group_by(Ticket.assigned_agent_id)
                .all()
            )
            active_by_agent = {row[0]: row[1] for row in counts}

            result = []
            for agent in agents:
                active = active_by_agent.get(agent.id, 0)
                d = _agent_to_dict(agent)
                d["active_tickets"] = active
                d["capacity_remaining"] = max(0, agent.max_tickets - active)
                result.append(d)

            return {"agents": result, "count": len(result)}

    @mcp.tool()
    def get_sla_report(
        category: str | None = None,
        priority: str | None = None,
    ) -> dict:
        """Get SLA compliance summary across all non-closed tickets.

        Optionally filter by category or priority to scope the report.
        Returns overall breach rate plus breakdowns by category and priority.
        A ticket is considered breached if its sla_deadline is in the past
        and its status is not resolved or closed.
        """
        now = datetime.utcnow()

        with get_db() as db:
            base_query = db.query(Ticket).filter(
                Ticket.status.notin_(["closed"]),
                Ticket.sla_deadline.isnot(None),
            )
            if category:
                base_query = base_query.filter(Ticket.category == category)
            if priority:
                base_query = base_query.filter(Ticket.priority == priority)

            all_tickets = base_query.all()
            total = len(all_tickets)
            breached = [
                t for t in all_tickets
                if t.sla_deadline < now and t.status not in ("resolved", "closed")
            ]
            breach_count = len(breached)

            def _breakdown(field: str) -> dict:
                groups: dict[str, dict] = {}
                for t in all_tickets:
                    key = getattr(t, field)
                    if key not in groups:
                        groups[key] = {"total": 0, "breached": 0}
                    groups[key]["total"] += 1
                    if t.sla_deadline < now and t.status not in ("resolved", "closed"):
                        groups[key]["breached"] += 1
                for key, vals in groups.items():
                    vals["breach_rate"] = (
                        round(vals["breached"] / vals["total"], 2) if vals["total"] else 0.0
                    )
                return groups

            return {
                "total_tickets": total,
                "breached": breach_count,
                "breach_rate": round(breach_count / total, 2) if total else 0.0,
                "by_category": _breakdown("category"),
                "by_priority": _breakdown("priority"),
                "filters_applied": {"category": category, "priority": priority},
            }
