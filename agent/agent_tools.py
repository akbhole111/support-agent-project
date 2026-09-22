import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import requests
from langchain_core.tools import tool
from tools.knowledge_base import query_knowledge_base

API_BASE = "http://localhost:8000"


@tool
def get_order_status(order_id: str) -> str:
    """Look up the status of a customer's order by order ID.
    Returns order status, item category, price, and delivery dates.
    Use this when the customer asks about an order, delivery, or shipment."""
    try:
        resp = requests.get(f"{API_BASE}/order-status/{order_id}", timeout=5)
        if resp.status_code == 404:
            return f"No order found with ID {order_id}. Please check the order ID and try again."
        resp.raise_for_status()
        return str(resp.json())
    except requests.exceptions.ConnectionError:
        return "Error: order lookup service is unavailable right now."


@tool
def search_knowledge_base(query: str) -> str:
    """Search the company's support knowledge base for policy answers
    (refunds, cancellations, shipping, billing, account issues, etc.).
    Use this when the customer asks a general policy/how-to question
    that isn't about a specific order."""
    results = query_knowledge_base(query, n_results=2)
    if not results:
        return "No relevant knowledge base entry found."
    formatted = "\n\n".join(
        f"[{r['category']} - {r['intent']}]: {r['answer']}" for r in results
    )
    return formatted


@tool
def create_support_ticket(customer_id: str, subject: str, description: str, priority: str = "Medium") -> str:
    """Create a support ticket to escalate an issue to a human agent.
    Use this only when the order lookup and knowledge base cannot resolve
    the customer's issue, or the customer explicitly asks to speak to a human."""
    try:
        resp = requests.post(
            f"{API_BASE}/create-ticket",
            json={
                "customer_id": customer_id,
                "subject": subject,
                "description": description,
                "priority": priority
            },
            timeout=5
        )
        resp.raise_for_status()
        return str(resp.json())
    except requests.exceptions.ConnectionError:
        return "Error: ticketing service is unavailable right now."


ALL_TOOLS = [get_order_status, search_knowledge_base, create_support_ticket]