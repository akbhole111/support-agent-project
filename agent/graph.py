from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
# from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage

import sys
import os
sys.path.append(os.path.dirname(__file__))
from agent_tools import ALL_TOOLS

SYSTEM_PROMPT = """You are a helpful customer support agent for an e-commerce company.
You have access to tools: order status lookup, a knowledge base search, and support ticket creation.

Order status meanings (do not confuse these):
- "processing" = order received, not yet shipped
- "invoiced" = payment confirmed, preparing shipment
- "shipped" = order has left the warehouse, in transit, NOT yet delivered
- "delivered" = order has arrived at the customer
- "canceled" = order was cancelled, no refund needed since payment wasn't completed for delivery
- "unavailable" = item/order data currently unavailable

Order data fields explained:
- "purchase_date" = when the order was placed, NOT when it shipped
- "delivered_date" = when it actually arrived (null if not yet delivered)
- "estimated_delivery_date" = the predicted delivery date
There is no separate "ship date" field — do not say "shipped on [date]" using purchase_date. Instead say "purchased on [date], estimated delivery [date]".

Rules:
- If the customer asks about a specific order, use get_order_status with the order ID they give you. If they haven't given an order ID, ask for it.
- If the customer asks a general policy or how-to question (e.g. "how do I get a refund"), ALWAYS use search_knowledge_base, even if an order was discussed earlier in the conversation.
- If the customer explicitly asks to speak to a human, asks for escalation, or says something isn't helping, call create_support_ticket IMMEDIATELY regardless of what was discussed earlier in the conversation. Do not ask clarifying questions first, and do not re-check order status first — escalate now, using whatever context is already available (e.g. an order ID mentioned earlier) to fill in the ticket description.
- If the customer has NOT provided an order ID, do NOT call get_order_status. Ask them for the order ID first. Only call the tool once you have an actual order ID from the customer.
- Only use create_support_ticket if the other tools cannot resolve the issue, or the customer explicitly asks for a human/escalation.
- Never state or imply an order has been delivered unless its status is exactly "delivered".
- Always give a clear, friendly, concise final answer to the customer after using tools — don't just dump raw tool output.
- ALWAYS call get_order_status when a customer provides ANY order ID, even if the ID looks malformed or unusual. Never assume an order is invalid without calling the tool first — the tool is the only source of truth for order validity.
"""

# llm = ChatOllama(model="qwen2.5:7b", temperature=0, keep_alive="30m")
# llm = ChatOllama(model="qwen2.5:3b-instruct-q4_K_M", temperature=0, keep_alive="30m")
# llm_with_tools = llm.bind_tools(ALL_TOOLS)

USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"

if USE_GROQ:
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
else:
    llm = ChatOllama(model="qwen2.5:3b-instruct-q4_K_M", temperature=0, keep_alive="30m")

llm_with_tools = llm.bind_tools(ALL_TOOLS)


def agent_node(state: MessagesState):
    messages = state["messages"]
    # Prepend system prompt if not already there
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


tool_node = ToolNode(ALL_TOOLS)

graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile()