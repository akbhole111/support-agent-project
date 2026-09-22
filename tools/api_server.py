from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
import uuid
from datetime import datetime

app = FastAPI(title="Mock Customer Support API")

# Load orders data once at startup
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "orders.json")
with open(DATA_PATH, "r") as f:
    ORDERS = json.load(f)

ORDERS_BY_ID = {o["order_id"]: o for o in ORDERS}

# In-memory ticket store (resets each time the server restarts — fine for a demo)
TICKETS = {}


@app.get("/order-status/{order_id}")
def get_order_status(order_id: str):
    order = ORDERS_BY_ID.get(order_id)
    if not order:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    return order


@app.get("/orders/sample")
def get_sample_orders(n: int = 5):
    """Handy endpoint for testing — returns n random order_ids to try."""
    return [o["order_id"] for o in ORDERS[:n]]


class TicketRequest(BaseModel):
    customer_id: str
    subject: str
    description: str
    priority: str = "Medium"


@app.post("/create-ticket")
def create_ticket(ticket: TicketRequest):
    ticket_id = str(uuid.uuid4())[:8]
    record = {
        "ticket_id": ticket_id,
        "customer_id": ticket.customer_id,
        "subject": ticket.subject,
        "description": ticket.description,
        "priority": ticket.priority,
        "status": "Open",
        "created_at": datetime.now().isoformat()
    }
    TICKETS[ticket_id] = record
    return record


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str):
    ticket = TICKETS.get(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
    return ticket