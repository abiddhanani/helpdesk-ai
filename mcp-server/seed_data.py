"""Seed the helpdesk database with realistic sample data.

Usage (from exercise-1/ directory):
    uv run seed_data.py

Truncates all tables and reloads fresh data on every run.
"""
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Allow running as a top-level script outside the package
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from src.database import SessionLocal, engine, init_db
from src.models import Agent, Customer, KnowledgeArticle, Ticket, TicketComment

SLA_HOURS = {
    "critical": timedelta(hours=4),
    "high": timedelta(hours=8),
    "medium": timedelta(hours=24),
    "low": timedelta(hours=72),
}


def wait_for_db(retries: int = 30, delay: float = 1.0) -> None:
    for attempt in range(1, retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("Database is ready.", file=sys.stderr)
            return
        except OperationalError:
            print(f"Waiting for database... attempt {attempt}/{retries}", file=sys.stderr)
            time.sleep(delay)
    raise RuntimeError("Database not ready after maximum retries.")


def truncate_all(conn) -> None:
    conn.execute(text(
        "TRUNCATE TABLE ticket_comments, tickets, knowledge_articles, customers, agents "
        "RESTART IDENTITY CASCADE"
    ))
    conn.commit()
    print("Tables truncated.", file=sys.stderr)


def main() -> None:
    wait_for_db()
    init_db()

    with engine.connect() as conn:
        truncate_all(conn)

    now = datetime.utcnow()

    db = SessionLocal()
    try:
        # ------------------------------------------------------------------ #
        # Agents (8)
        # ------------------------------------------------------------------ #
        agents = [
            Agent(name="Alice Chen", email="alice.chen@helpdesk.local", specialty="network", is_available=True, max_tickets=8),
            Agent(name="Bob Martinez", email="bob.martinez@helpdesk.local", specialty="software", is_available=True, max_tickets=10),
            Agent(name="Carol Singh", email="carol.singh@helpdesk.local", specialty="hardware", is_available=True, max_tickets=10),
            Agent(name="David Kim", email="david.kim@helpdesk.local", specialty="security", is_available=True, max_tickets=6),
            Agent(name="Eva Johansson", email="eva.johansson@helpdesk.local", specialty="network", is_available=False, max_tickets=8),
            Agent(name="Frank Okafor", email="frank.okafor@helpdesk.local", specialty="software", is_available=True, max_tickets=10),
            Agent(name="Grace Lee", email="grace.lee@helpdesk.local", specialty="hardware", is_available=True, max_tickets=10),
            Agent(name="Henry Patel", email="henry.patel@helpdesk.local", specialty="security", is_available=True, max_tickets=6),
        ]
        db.add_all(agents)
        db.flush()

        a = {agent.name: agent for agent in agents}

        # ------------------------------------------------------------------ #
        # Customers (15)
        # ------------------------------------------------------------------ #
        customers = [
            Customer(name="James Wilson", email="james.wilson@corp.local", department="Engineering", created_at=now - timedelta(days=300)),
            Customer(name="Sarah Thompson", email="sarah.thompson@corp.local", department="Marketing", created_at=now - timedelta(days=280)),
            Customer(name="Michael Brown", email="michael.brown@corp.local", department="Finance", created_at=now - timedelta(days=260)),
            Customer(name="Emily Davis", email="emily.davis@corp.local", department="HR", created_at=now - timedelta(days=240)),
            Customer(name="Robert Garcia", email="robert.garcia@corp.local", department="Sales", created_at=now - timedelta(days=220)),
            Customer(name="Jennifer Martinez", email="jennifer.martinez@corp.local", department="Engineering", created_at=now - timedelta(days=200)),
            Customer(name="William Johnson", email="william.johnson@corp.local", department="Operations", created_at=now - timedelta(days=180)),
            Customer(name="Amanda White", email="amanda.white@corp.local", department="Legal", created_at=now - timedelta(days=160)),
            Customer(name="Christopher Taylor", email="christopher.taylor@corp.local", department="Finance", created_at=now - timedelta(days=140)),
            Customer(name="Jessica Anderson", email="jessica.anderson@corp.local", department="Marketing", created_at=now - timedelta(days=120)),
            Customer(name="Daniel Jackson", email="daniel.jackson@corp.local", department="Engineering", created_at=now - timedelta(days=100)),
            Customer(name="Ashley Harris", email="ashley.harris@corp.local", department="HR", created_at=now - timedelta(days=80)),
            Customer(name="Matthew Lewis", email="matthew.lewis@corp.local", department="Sales", created_at=now - timedelta(days=60)),
            Customer(name="Stephanie Robinson", email="stephanie.robinson@corp.local", department="Operations", created_at=now - timedelta(days=40)),
            Customer(name="Ryan Walker", email="ryan.walker@corp.local", department="Legal", created_at=now - timedelta(days=20)),
        ]
        db.add_all(customers)
        db.flush()

        c = customers  # index by position for convenience

        # ------------------------------------------------------------------ #
        # Tickets (35) — varied statuses, priorities, categories; ~10 overdue
        # ------------------------------------------------------------------ #
        def make_ticket(title, desc, priority, category, customer, agent, status, created_ago_hours, sla_offset_hours=None):
            created = now - timedelta(hours=created_ago_hours)
            if sla_offset_hours is not None:
                # Explicit override: positive = future, negative = past (overdue)
                sla = now + timedelta(hours=sla_offset_hours)
            else:
                sla = created + SLA_HOURS[priority]
            return Ticket(
                title=title,
                description=desc,
                status=status,
                priority=priority,
                category=category,
                customer_id=customer.id,
                assigned_agent_id=agent.id if agent else None,
                sla_deadline=sla,
                created_at=created,
                updated_at=created,
            )

        tickets = [
            # --- Critical / overdue (5) ---
            make_ticket("Complete network outage in Building A", "All workstations in Building A cannot reach the internet or internal network. Affecting 50+ users.", "critical", "network", c[0], a["Alice Chen"], "in_progress", 10, -2),
            make_ticket("Production database server unreachable", "The primary database server is not responding. Multiple applications are down. Engineers cannot deploy.", "critical", "software", c[5], a["Bob Martinez"], "in_progress", 8, -1),
            make_ticket("Security breach detected on finance servers", "Unusual login activity detected on finance file servers. Possible unauthorized access to payroll data.", "critical", "security", c[2], a["David Kim"], "open", 6, -3),
            make_ticket("VPN gateway failure", "VPN gateway is refusing all connections. Remote workers cannot access internal systems.", "critical", "network", c[6], a["Alice Chen"], "in_progress", 12, -4),
            make_ticket("Active Directory domain controller down", "Domain controller is unreachable. Users cannot authenticate. Login failures across the organization.", "critical", "software", c[10], a["Bob Martinez"], "in_progress", 9, -5),

            # --- High / some overdue (8) ---
            make_ticket("Email server performance degradation", "Exchange server is extremely slow. Emails taking 10-15 minutes to deliver. Marketing team unable to send campaigns.", "high", "software", c[1], a["Frank Okafor"], "in_progress", 20, -2),
            make_ticket("Printer fleet offline in Finance dept", "All 6 printers in the Finance department are offline after a firmware update pushed last night.", "high", "hardware", c[2], a["Carol Singh"], "in_progress", 18, -1),
            make_ticket("SSL certificate expired on customer portal", "Customer-facing portal showing certificate error. Sales team unable to demo the product to prospects.", "high", "software", c[4], a["Frank Okafor"], "open", 16, 3),
            make_ticket("New hire workstation setup — batch of 5", "5 new engineers starting Monday. Need workstations configured with dev tools, VPN, and repo access.", "high", "hardware", c[5], a["Grace Lee"], "open", 14, 6),
            make_ticket("Ransomware alert on HR shared drive", "Antivirus flagged suspicious encryption activity on HR shared drive. Drive has been quarantined pending investigation.", "high", "security", c[3], a["Henry Patel"], "in_progress", 22, -3),
            make_ticket("Wi-Fi dead zones in conference rooms 3-6", "Conference rooms 3 through 6 have no Wi-Fi coverage. Executive board meetings scheduled this week.", "high", "network", c[6], a["Alice Chen"], "open", 30, 5),
            make_ticket("Laptop screen broken — executive device", "CFO's laptop screen is cracked. Device unusable. Needs immediate replacement or loaner.", "high", "hardware", c[7], a["Carol Singh"], "open", 5, 7),
            make_ticket("CRM application crashing on save", "Salesforce integration crashing when reps save opportunity records. Affecting all 20 sales reps.", "high", "software", c[4], a["Bob Martinez"], "in_progress", 40, -2),

            # --- Medium (12) ---
            make_ticket("Password reset for locked account", "User locked out of account after multiple failed attempts. Needs password reset and MFA re-enrollment.", "medium", "access", c[11], a["Henry Patel"], "resolved", 72, 20),
            make_ticket("Slow VPN speeds for remote employees", "Remote workers reporting VPN speeds below 5 Mbps. Affecting video calls and file transfers.", "medium", "network", c[0], a["Alice Chen"], "in_progress", 48, 10),
            make_ticket("Software license renewal — Adobe CC", "Adobe Creative Cloud licenses expiring in 2 weeks for 12 design team members.", "medium", "software", c[1], a["Frank Okafor"], "open", 60, 36),
            make_ticket("Monitor flickering on workstation", "Secondary monitor on user's workstation flickers intermittently. Tried different cables — same issue.", "medium", "hardware", c[8], a["Grace Lee"], "in_progress", 36, 15),
            make_ticket("Request access to financial reporting system", "Business analyst needs read access to the financial reporting dashboard for Q4 audit preparation.", "medium", "access", c[2], None, "open", 24, 20),
            make_ticket("macOS update breaking dev tools", "After macOS 14.3 update, Homebrew and Docker Desktop are not working. Affecting 8 engineers.", "medium", "software", c[10], a["Bob Martinez"], "in_progress", 72, 5),
            make_ticket("Conference room AV system not working", "Projector in main conference room not connecting via HDMI. Important client presentation tomorrow.", "medium", "hardware", c[6], a["Grace Lee"], "resolved", 48, 24),
            make_ticket("Guest Wi-Fi network down", "Guest Wi-Fi is not broadcasting. Visitors and contractors cannot access the internet.", "medium", "network", c[13], a["Alice Chen"], "in_progress", 96, 8),
            make_ticket("Onboarding — set up email for new hire", "New marketing coordinator starting next week. Needs email account, Slack, and Google Workspace access.", "medium", "access", c[1], a["Henry Patel"], "open", 12, 18),
            make_ticket("Laptop battery draining too fast", "Engineering manager's laptop battery lasts only 2 hours. Under warranty — needs diagnosis.", "medium", "hardware", c[5], a["Carol Singh"], "open", 8, 22),
            make_ticket("Backup job failing nightly", "Nightly backup job has been failing for 3 days with error 'insufficient disk space'. Storage at 94%.", "medium", "software", c[10], a["Frank Okafor"], "in_progress", 84, -5),
            make_ticket("Request SharePoint site for project team", "Project manager requesting a new SharePoint site for the Q1 infrastructure project.", "medium", "access", c[13], None, "open", 18, 20),

            # --- Low (10) ---
            make_ticket("Keyboard feels sticky after liquid spill", "User spilled coffee on keyboard. Keys are sticky but still functional. Requesting replacement.", "low", "hardware", c[11], a["Grace Lee"], "open", 120, 50),
            make_ticket("Request ergonomic mouse", "Developer experiencing wrist pain. Requesting an ergonomic mouse as a reasonable adjustment.", "low", "hardware", c[0], None, "open", 96, 60),
            make_ticket("Old laptop disposal — 3 units", "3 laptops from departed employees need secure data wiping and disposal per company policy.", "low", "hardware", c[7], a["Carol Singh"], "resolved", 168, 48),
            make_ticket("Browser homepage reset after update", "Edge browser homepage reverts to Bing after every Windows update. Minor annoyance but persistent.", "low", "software", c[12], a["Frank Okafor"], "open", 72, 60),
            make_ticket("Outlook signature not applying correctly", "Email signature not appearing on replies, only on new emails. User has tried reconfiguring.", "low", "software", c[3], a["Bob Martinez"], "open", 48, 65),
            make_ticket("Desk phone voicemail setup", "New employee needs voicemail set up on their desk phone. No urgency — they use mobile primarily.", "low", "other", c[14], None, "open", 36, 70),
            make_ticket("Software recommendation for screen recording", "Marketing team requesting a software recommendation for screen recording training videos.", "low", "software", c[1], None, "open", 24, 68),
            make_ticket("USB hub not recognized", "USB hub connected to docking station not recognized. Workaround: plugging devices directly into dock.", "low", "hardware", c[8], a["Grace Lee"], "closed", 240, 48),
            make_ticket("Reconfigure dual-monitor setup", "User moved to a new desk. Needs monitors repositioned and display settings reconfigured.", "low", "other", c[9], None, "waiting_on_customer", 60, 70),
            make_ticket("IT asset inventory audit support", "IT team needs help tagging and cataloguing laptops in the Engineering department for annual audit.", "low", "other", c[5], a["Henry Patel"], "in_progress", 168, 30),
        ]
        db.add_all(tickets)
        db.flush()

        # ------------------------------------------------------------------ #
        # Knowledge Articles (10)
        # ------------------------------------------------------------------ #
        articles = [
            KnowledgeArticle(
                title="Troubleshooting VPN Connection Issues",
                content=(
                    "This guide covers common VPN connection problems and their solutions. "
                    "Step 1: Verify your internet connection is active. Step 2: Check VPN client version — "
                    "ensure you are running version 5.x or later. Step 3: Clear the VPN client cache by "
                    "navigating to Settings > Advanced > Clear Cache. Step 4: If the gateway is unreachable, "
                    "contact the network team to check gateway status. Step 5: For persistent issues, "
                    "reinstall the VPN client using the IT portal software center."
                ),
                category="network",
                tags="vpn,remote-access,connectivity,gateway",
                created_at=now - timedelta(days=90),
                updated_at=now - timedelta(days=10),
            ),
            KnowledgeArticle(
                title="Password Reset and MFA Re-enrollment Procedure",
                content=(
                    "To reset a user password: 1. Go to the IT self-service portal at helpdesk.corp.local. "
                    "2. Click 'Forgot Password' and enter your employee email. 3. Check your recovery email "
                    "for a reset link (valid for 30 minutes). 4. Set a new password meeting complexity requirements: "
                    "minimum 12 characters, one uppercase, one number, one special character. "
                    "5. To re-enroll MFA: open the Authenticator app, add a new account, and scan the QR code "
                    "shown in the self-service portal under Security Settings."
                ),
                category="access",
                tags="password,mfa,authentication,account,locked",
                created_at=now - timedelta(days=120),
                updated_at=now - timedelta(days=5),
            ),
            KnowledgeArticle(
                title="How to Set Up a New Employee Workstation",
                content=(
                    "Standard workstation setup checklist for new hires: "
                    "1. Unbox hardware and connect monitors, keyboard, mouse, and network cable. "
                    "2. Boot to the IT provisioning USB and run the automated setup script. "
                    "3. The script will: join the domain, install standard software (Office 365, VPN client, "
                    "antivirus, Chrome), configure printers, and set up the IT management agent. "
                    "4. Log in with the new hire's credentials and complete Windows setup wizard. "
                    "5. Install role-specific software from the IT software portal. "
                    "6. Test: email, VPN, printer access, and network shares."
                ),
                category="hardware",
                tags="onboarding,workstation,setup,new-hire,provisioning",
                created_at=now - timedelta(days=200),
                updated_at=now - timedelta(days=30),
            ),
            KnowledgeArticle(
                title="Diagnosing and Fixing Printer Offline Issues",
                content=(
                    "When printers show as offline: "
                    "1. Check the printer's physical power and network cable connections. "
                    "2. Print a configuration page directly from the printer control panel to verify it has an IP. "
                    "3. On Windows: go to Devices and Printers, right-click the printer, select 'See what's printing', "
                    "then click Printer > Use Printer Online. "
                    "4. If the printer IP has changed, update the port in printer properties. "
                    "5. For fleet-wide offline issues after a firmware push, roll back firmware via the printer admin portal."
                ),
                category="hardware",
                tags="printer,offline,hardware,printing,firmware",
                created_at=now - timedelta(days=150),
                updated_at=now - timedelta(days=20),
            ),
            KnowledgeArticle(
                title="Responding to Ransomware and Malware Incidents",
                content=(
                    "Immediate steps when ransomware or malware is suspected: "
                    "1. Isolate the affected machine — disconnect from network (unplug cable, disable Wi-Fi). "
                    "2. Do NOT reboot — forensic evidence may be lost. "
                    "3. Contact the security team immediately via the emergency Slack channel #security-incidents. "
                    "4. Do not attempt to decrypt or remove the malware yourself. "
                    "5. Preserve logs: take screenshots of any ransom notes or error messages. "
                    "6. The security team will forensically image the drive and begin investigation. "
                    "7. Restoration from last clean backup will be coordinated after forensics are complete."
                ),
                category="security",
                tags="ransomware,malware,security,incident,breach",
                created_at=now - timedelta(days=180),
                updated_at=now - timedelta(days=15),
            ),
            KnowledgeArticle(
                title="Troubleshooting Slow or Dropped Wi-Fi",
                content=(
                    "For Wi-Fi performance issues: "
                    "1. Check if the issue is isolated to one device or affects multiple — if multiple, escalate to network team. "
                    "2. On the affected device: forget the Wi-Fi network and reconnect. "
                    "3. Check for channel congestion: use a Wi-Fi analyzer app to see if the AP is on a crowded channel. "
                    "4. If dead zones are reported in a specific area, the network team will audit AP coverage and add APs if needed. "
                    "5. For guest Wi-Fi issues, check the guest VLAN configuration on the wireless controller. "
                    "6. Reboot the access point as a last resort — coordinate with network team to avoid disrupting others."
                ),
                category="network",
                tags="wifi,wireless,network,connectivity,slow,dead-zone",
                created_at=now - timedelta(days=60),
                updated_at=now - timedelta(days=3),
            ),
            KnowledgeArticle(
                title="SSL Certificate Renewal Process",
                content=(
                    "To renew an SSL/TLS certificate: "
                    "1. Generate a new Certificate Signing Request (CSR) on the server: "
                    "`openssl req -new -newkey rsa:2048 -nodes -keyout server.key -out server.csr`. "
                    "2. Submit the CSR to the certificate authority (CA) via the IT procurement portal. "
                    "3. Once issued (typically 1-2 business days), install the certificate: "
                    "copy cert.pem to /etc/ssl/certs/ and update the web server config. "
                    "4. Reload the web server: `systemctl reload nginx` or `systemctl reload apache2`. "
                    "5. Verify with: `openssl s_client -connect yourdomain.com:443`. "
                    "6. Set a calendar reminder 60 days before the next expiry."
                ),
                category="software",
                tags="ssl,certificate,tls,https,renewal,security",
                created_at=now - timedelta(days=45),
                updated_at=now - timedelta(days=2),
            ),
            KnowledgeArticle(
                title="macOS Developer Environment Recovery After OS Update",
                content=(
                    "After a macOS update breaks developer tools: "
                    "1. Reinstall Xcode Command Line Tools: `xcode-select --install`. "
                    "2. For Homebrew issues: run `brew doctor` and follow the output instructions. "
                    "3. If Homebrew is missing: reinstall via `/bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"`. "
                    "4. For Docker Desktop: download the latest version from the Docker website and reinstall. "
                    "5. Python/Node version managers (pyenv, nvm): run `pyenv rehash` or `nvm install --reinstall-packages-from=current`. "
                    "6. If issues persist, the IT team maintains a dev environment setup script at git.corp.local/it/dev-setup."
                ),
                category="software",
                tags="macos,developer,homebrew,docker,update,dev-tools",
                created_at=now - timedelta(days=30),
                updated_at=now - timedelta(days=1),
            ),
            KnowledgeArticle(
                title="Storage Management and Disk Cleanup Guide",
                content=(
                    "When disk space is critically low: "
                    "1. Identify large files: `du -sh /* 2>/dev/null | sort -rh | head -20`. "
                    "2. Check and clear log files: logs older than 30 days in /var/log can typically be archived. "
                    "3. Docker cleanup: `docker system prune -a --volumes` (confirm with the dev team first). "
                    "4. For backup job failures due to space: expand the backup volume via the storage admin portal, "
                    "or delete the oldest backup set after confirming a recent successful backup exists. "
                    "5. Request a storage expansion ticket if recurring — storage quotas are managed by the infrastructure team."
                ),
                category="software",
                tags="storage,disk,backup,space,cleanup,logs",
                created_at=now - timedelta(days=75),
                updated_at=now - timedelta(days=7),
            ),
            KnowledgeArticle(
                title="Access Request and Provisioning Workflow",
                content=(
                    "To request access to systems or shared resources: "
                    "1. Submit an access request ticket with: system name, access level needed (read/write/admin), "
                    "business justification, and manager approval (attach email or tag manager in ticket). "
                    "2. IT will verify manager approval before provisioning. "
                    "3. Standard provisioning SLA: 24 hours for standard access, 4 hours for critical business need. "
                    "4. For SharePoint sites: IT creates the site and sets the requesting manager as site owner. "
                    "5. For financial systems: Finance director sign-off is required in addition to direct manager approval. "
                    "6. Access is reviewed quarterly — unused access may be revoked automatically."
                ),
                category="access",
                tags="access,provisioning,sharepoint,permissions,onboarding",
                created_at=now - timedelta(days=100),
                updated_at=now - timedelta(days=14),
            ),
        ]
        db.add_all(articles)
        db.flush()

        # ------------------------------------------------------------------ #
        # Comments (25)
        # ------------------------------------------------------------------ #
        t = tickets  # shorthand

        comments = [
            # Ticket 0 — network outage
            TicketComment(ticket_id=t[0].id, author_type="agent", author_id=a["Alice Chen"].id, content="Confirmed: core switch in Building A is unresponsive. Escalating to network vendor.", created_at=t[0].created_at + timedelta(minutes=20)),
            TicketComment(ticket_id=t[0].id, author_type="system", author_id=0, content="SLA breach detected. Ticket escalated to priority queue.", created_at=now - timedelta(hours=1)),

            # Ticket 1 — database down
            TicketComment(ticket_id=t[1].id, author_type="agent", author_id=a["Bob Martinez"].id, content="Investigating. DB server shows disk I/O saturation. May need emergency storage expansion.", created_at=t[1].created_at + timedelta(minutes=15)),
            TicketComment(ticket_id=t[1].id, author_type="customer", author_id=c[5].id, content="This is blocking all our deploys. Please prioritize!", created_at=t[1].created_at + timedelta(hours=1)),

            # Ticket 2 — security breach
            TicketComment(ticket_id=t[2].id, author_type="agent", author_id=a["David Kim"].id, content="Isolated the affected servers. Running forensic scan. Do not reboot any finance machines.", created_at=t[2].created_at + timedelta(minutes=30)),

            # Ticket 4 — AD down
            TicketComment(ticket_id=t[4].id, author_type="agent", author_id=a["Bob Martinez"].id, content="Domain controller service crashed. Attempting restore from last snapshot.", created_at=t[4].created_at + timedelta(minutes=25)),
            TicketComment(ticket_id=t[4].id, author_type="system", author_id=0, content="Automated health check failed 3 consecutive times. Alert sent to on-call.", created_at=t[4].created_at + timedelta(minutes=5)),

            # Ticket 5 — email slow
            TicketComment(ticket_id=t[5].id, author_type="agent", author_id=a["Frank Okafor"].id, content="Exchange message queue has 4,200 messages backed up. Investigating mail flow rules.", created_at=t[5].created_at + timedelta(hours=2)),
            TicketComment(ticket_id=t[5].id, author_type="customer", author_id=c[1].id, content="We have a campaign going out today — this is critical for us!", created_at=t[5].created_at + timedelta(hours=3)),

            # Ticket 9 — ransomware HR
            TicketComment(ticket_id=t[9].id, author_type="agent", author_id=a["Henry Patel"].id, content="Quarantine confirmed. Running Defender offline scan. Notifying CISO per incident response procedure.", created_at=t[9].created_at + timedelta(minutes=45)),
            TicketComment(ticket_id=t[9].id, author_type="system", author_id=0, content="Incident response process initiated. Ticket tagged SECURITY-INCIDENT.", created_at=t[9].created_at + timedelta(minutes=10)),

            # Ticket 13 — password reset (resolved)
            TicketComment(ticket_id=t[13].id, author_type="agent", author_id=a["Henry Patel"].id, content="Password reset completed. MFA re-enrollment link sent to recovery email.", created_at=t[13].created_at + timedelta(hours=1)),
            TicketComment(ticket_id=t[13].id, author_type="customer", author_id=c[11].id, content="Got the link, setting up now. Thank you!", created_at=t[13].created_at + timedelta(hours=2)),

            # Ticket 19 — macOS
            TicketComment(ticket_id=t[19].id, author_type="agent", author_id=a["Bob Martinez"].id, content="Ran `xcode-select --install` and `brew doctor` on 3 of the 8 machines. Issue resolved on those. Continuing rollout.", created_at=t[19].created_at + timedelta(hours=4)),

            # Ticket 20 — AV system (resolved)
            TicketComment(ticket_id=t[20].id, author_type="agent", author_id=a["Grace Lee"].id, content="HDMI cable was faulty. Replaced with new cable from stock. Projector working.", created_at=t[20].created_at + timedelta(hours=2)),
            TicketComment(ticket_id=t[20].id, author_type="customer", author_id=c[6].id, content="Perfect, presentation went great. Thanks for the quick fix!", created_at=t[20].created_at + timedelta(hours=3)),

            # Ticket 23 — backup failing
            TicketComment(ticket_id=t[23].id, author_type="agent", author_id=a["Frank Okafor"].id, content="Storage at 94%. Identified 200GB of old Docker images. Requesting approval to clean up.", created_at=t[23].created_at + timedelta(hours=6)),
            TicketComment(ticket_id=t[23].id, author_type="customer", author_id=c[10].id, content="Approved — clean up Docker images older than 30 days.", created_at=t[23].created_at + timedelta(hours=8)),

            # Ticket 27 — laptop disposal (resolved)
            TicketComment(ticket_id=t[27].id, author_type="agent", author_id=a["Carol Singh"].id, content="All 3 units wiped with DBAN. Certificates of destruction generated and filed.", created_at=t[27].created_at + timedelta(days=3)),

            # Ticket 33 — asset inventory
            TicketComment(ticket_id=t[33].id, author_type="agent", author_id=a["Henry Patel"].id, content="Started audit in Engineering wing. 40 of 80 laptops tagged so far.", created_at=t[33].created_at + timedelta(days=2)),
            TicketComment(ticket_id=t[33].id, author_type="customer", author_id=c[5].id, content="Can you come by Tuesday morning? The team will all be in.", created_at=t[33].created_at + timedelta(days=3)),
            TicketComment(ticket_id=t[33].id, author_type="agent", author_id=a["Henry Patel"].id, content="Confirmed for Tuesday 9am. Will bring the asset scanner.", created_at=t[33].created_at + timedelta(days=3, hours=1)),

            # Extra comments to reach 25
            TicketComment(ticket_id=t[3].id, author_type="agent", author_id=a["Alice Chen"].id, content="VPN gateway is back online after restart. Monitoring for stability.", created_at=t[3].created_at + timedelta(hours=3)),
            TicketComment(ticket_id=t[7].id, author_type="customer", author_id=c[4].id, content="The demo is in 4 hours — is there any workaround we can use?", created_at=t[7].created_at + timedelta(hours=2)),
            TicketComment(ticket_id=t[7].id, author_type="agent", author_id=a["Frank Okafor"].id, content="Workaround: use HTTP (port 80) temporarily. SSL cert renewal request submitted.", created_at=t[7].created_at + timedelta(hours=3)),
            TicketComment(ticket_id=t[14].id, author_type="agent", author_id=a["Alice Chen"].id, content="Split tunneling enabled for affected remote workers as interim fix. Investigating root cause of speed issue.", created_at=t[14].created_at + timedelta(hours=5)),
        ]
        db.add_all(comments)
        db.commit()

        print(f"Seed complete: {len(agents)} agents, {len(customers)} customers, "
              f"{len(tickets)} tickets, {len(articles)} KB articles, {len(comments)} comments.",
              file=sys.stderr)

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
