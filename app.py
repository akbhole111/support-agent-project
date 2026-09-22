import streamlit as st
import sys, os
import threading
import time
import uvicorn

sys.path.append(os.path.join(os.path.dirname(__file__), "agent"))
from graph import app as agent_app
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

st.set_page_config(page_title="Customer Support Agent", page_icon="🤖", layout="centered")

def run_api_server():
    from tools.api_server import app as fastapi_app
    uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, log_level="warning")

# Near the top, replace session_state-based guards with a global flag
_api_started = False

def start_api_once():
    global _api_started
    if not _api_started:
        thread = threading.Thread(target=run_api_server, daemon=True)
        thread.start()
        _api_started = True
        time.sleep(2)

start_api_once()

if "kb_built" not in st.session_state:
    from tools.knowledge_base import build_knowledge_base
    build_knowledge_base()
    st.session_state.kb_built = True

st.title("🤖 Customer Support Agent")

USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"
model_label = "openai/gpt-oss-20b via Groq" if USE_GROQ else "qwen2.5:3b-instruct-q4_K_M via Ollama"
st.caption(f"Agentic support assistant — order lookups, policy Q&A, and escalation, powered by {model_label}")

# Initialize conversation state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {"messages": []}

# Sidebar: sample data to try
with st.sidebar:
    st.header("Try it out")
    st.markdown("**Sample order IDs:**")
    st.code("2c75c33f103e365438cc19a05d444b7f", language=None)
    st.code("ab76f54a321a0431ef243b3b6865078b", language=None)
    st.markdown("**Sample questions:**")
    st.markdown("- How do I request a refund?\n- What's the status of order [ID]?\n- I want to speak to a human")

    if st.button("Reset conversation"):
        st.session_state.messages = []
        st.session_state.agent_state = {"messages": []}
        st.rerun()

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("tool_calls"):
            with st.expander("🔧 Tools used"):
                for tc in msg["tool_calls"]:
                    st.code(tc, language=None)

# Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.agent_state["messages"].append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent_app.invoke(st.session_state.agent_state)
            st.session_state.agent_state = result

            # Extract tool calls made this turn for display
            tool_calls_used = []
            for m in result["messages"]:
                if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                    for tc in m.tool_calls:
                        tool_calls_used.append(f"{tc['name']}({tc['args']})")

            final_answer = result["messages"][-1].content
            st.write(final_answer)

            if tool_calls_used:
                with st.expander("🔧 Tools used"):
                    for tc in tool_calls_used:
                        st.code(tc, language=None)

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer,
        "tool_calls": tool_calls_used
    })