# 🤖 Customer Support Agent — Agentic Workflows Portfolio Project

**Live demo:** https://anujbhole-support-agent-project.streamlit.app

An automated customer support agent that goes beyond a basic chatbot: it queries real backend systems, calls the right tool for the job, and keeps track of multi-turn conversation state — built entirely with free, open tools.

---

## What it does

The agent handles three kinds of customer requests:

- **Order status lookups** — queries a mock order database (real e-commerce data) for status, delivery dates, and item info
- **Policy / FAQ questions** — semantic search over a real customer-support knowledge base (refunds, cancellations, shipping, billing, etc.)
- **Escalation** — creates a support ticket when a request can't be resolved or the customer explicitly asks for a human

It decides which tool to use (or whether to ask a clarifying question instead) based on the conversation so far — including context from earlier turns.

---

## Architecture

```
User message
     │
     ▼
┌──────────────┐      ┌──────────────────────┐
│  LangGraph   │◄────►│  LLM (reasoning)     │
│  agent loop  │      │  qwen2.5:3b (local)  │
│              │      │  gpt-oss-20b (Groq)  │
└──────┬───────┘      └──────────────────────┘
       │
       ▼ (tool calls)
┌──────────────┬─────────────────┬──────────────────┐
│ get_order_   │ search_         │ create_support_  │
│ status       │ knowledge_base  │ ticket           │
│              │                 │                  │
│ FastAPI mock │ ChromaDB +      │ FastAPI mock     │
│ backend      │ sentence-       │ backend          │
│              │ transformers    │                  │
└──────────────┴─────────────────┴──────────────────┘
```

**Reasoning engine:** runs locally via [Ollama](https://ollama.com) (`qwen2.5:3b-instruct-q4_K_M`) for development, and via [Groq](https://groq.com)'s free API (`openai/gpt-oss-20b`) for the deployed version — same agent code, swapped via an environment variable.

**Agent framework:** [LangGraph](https://langchain-ai.github.io/langgraph/) — handles the tool-call loop and multi-turn conversation state.

**Tools:**
- A FastAPI mock backend simulating a real order-management and ticketing system
- A ChromaDB vector store for semantic FAQ retrieval, embedded locally with `sentence-transformers` (no paid embedding API)

**UI:** Streamlit, deployed on Streamlit Community Cloud.

---

## Datasets (real, not synthetic)

| Dataset | Source | Used for |
|---|---|---|
| Brazilian E-Commerce (Olist) | [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) | Order status tool — ~600 real orders across 6 statuses |
| Bitext Customer Support LLM Training Dataset | [Hugging Face](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | Knowledge base — 270 real Q&A pairs across 27 support intents |

An earlier Kaggle ticket dataset was tried first for the knowledge base but its "resolution" field turned out to be auto-generated filler text, not real answers — caught during evaluation and swapped for the Bitext dataset, which has genuine, well-formed support responses. (See [Known limitations](#known-limitations) below.)

---

## Evaluation

Rather than eyeballing single conversations, the agent is tested against a 10-case structured suite (`eval/run_eval.py`) covering:
- Basic tool use (order lookup, policy question)
- Missing-information handling (agent should ask, not guess)
- Multi-turn context retention
- Escalation (explicit request, and after a failed resolution attempt)
- Edge cases: invalid order IDs, ambiguous requests, topic switches

**Result:** 10/10 passing, on both the local model (qwen2.5:3b) and the deployed model (gpt-oss-20b via Groq).

### Bugs found and fixed during evaluation

This is the part of the project I'd point to as evidence of real debugging, not just "it worked first try":

1. **Hallucinated tool verification** — the agent once claimed an order ID "didn't exist" without actually calling the lookup tool, just because the ID looked malformed. Fixed by explicitly instructing the agent to always verify via the tool, never assume.
2. **Prompt-instruction interference** — fixing bug #1 caused a *different* regression: the agent started asking for an order ID even on unrelated escalation requests, because the new instruction shifted its general priorities. Fixed by making the escalation rule explicit enough to not get crowded out.
3. **Context-dependent escalation failure** — the automated test suite passed an "I want to speak to a human" scenario in isolation, but the same request *failed* in a live multi-turn session where an order had just been discussed — the agent second-guessed itself back into asking for order details. This was only caught through manual, exploratory testing, not the automated suite — a genuine limitation of scripted eval sets.
4. **Model-size tradeoff** — swapping from `qwen2.5:7b` to a smaller `qwen2.5:3b` quantized model (to fit a 4GB GPU and cut response latency ~3-4x) reintroduced a version of the "asks for ID unnecessarily" bug, requiring an explicit prompt constraint to fully resolve.

### Known limitations

- **Thin knowledge-base coverage for some intents.** With only ~10 examples per intent, less common categories (e.g. subscription cancellation) can occasionally retrieve a topically-adjacent but incorrect-category answer (e.g. returning an "order cancellation" answer for a "subscription cancellation" question). A larger or better-balanced sample would reduce this.
- **Smaller/quantized models are more prone to speculative tool calls** — attempting a tool call without sufficient information, rather than asking a clarifying question first. Mitigated via explicit prompt constraints; not fully eliminated at the 3B model size.
- **No persistent conversation storage** — history resets on page reload; this is a single-session demo, not a production system with user accounts.

---

## Tech stack

Python · LangGraph · LangChain · Ollama / Groq · FastAPI · ChromaDB · sentence-transformers · pandas · Streamlit

All tools used are free — no paid APIs required to run this locally (Ollama). The deployed version uses Groq's free tier for inference.

---

## Running it locally

```bash
git clone https://github.com/akbhole111/support-agent-project.git
cd support-agent-project
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Pull a local model via Ollama (https://ollama.com)
ollama pull qwen2.5:3b-instruct-q4_K_M

# Prepare data and build the knowledge base
python data/prepare_data.py
python tools/knowledge_base.py

# Run the mock backend
uvicorn tools.api_server:app --reload --port 8000

# In a separate terminal, run the app
streamlit run app.py
```

To use Groq instead of a local model, set:
```bash
$env:USE_GROQ="true"
$env:GROQ_API_KEY="your_key_here"
```

Run the eval suite:
```bash
python eval/run_eval.py
```
