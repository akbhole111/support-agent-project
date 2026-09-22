import chromadb
from chromadb.utils import embedding_functions
import json
import os
import uuid

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faqs.json")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_store")

# Use a free, local sentence-transformers model for embeddings (no API calls, no cost)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=CHROMA_DIR)


def build_knowledge_base():
    """Run once to create/populate the ChromaDB collection from faqs.json."""
    with open(DATA_PATH, "r") as f:
        faqs = json.load(f)

    collection = client.get_or_create_collection(
        name="support_faqs",
        embedding_function=embedding_fn
    )

    # Skip rebuilding if already populated
    if collection.count() > 0:
        print(f"Collection already has {collection.count()} documents. Skipping rebuild.")
        return collection

    documents = []
    metadatas = []
    ids = []

    for faq in faqs:
        doc_text = f"Intent: {faq['intent']}\nQuestion: {faq['question']}\nAnswer: {faq['answer']}"
        documents.append(doc_text)
        metadatas.append({
            "category": faq["category"] or "Unknown",
            "intent": faq["intent"] or "",
            "answer": faq["answer"] or ""
        })
        ids.append(str(uuid.uuid4()))  # no ticket_id anymore, generate one

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Added {len(documents)} FAQ documents to ChromaDB collection 'support_faqs'.")
    return collection


def query_knowledge_base(query: str, n_results: int = 3):
    """The actual tool function the agent will call."""
    collection = client.get_or_create_collection(
        name="support_faqs",
        embedding_function=embedding_fn
    )
    results = collection.query(query_texts=[query], n_results=n_results)

    output = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        output.append({
            "answer": meta["answer"],
            "category": meta["category"],
            "intent": meta["intent"],
            "relevance_score": round(1 - dist, 3)
        })
    return output


if __name__ == "__main__":
    build_knowledge_base()

    # Quick manual test
    test_query = "How do I get a refund for a delayed order?"
    print(f"\nTest query: {test_query}\n")
    for r in query_knowledge_base(test_query):
        print(f"- [{r['category']}] {r['intent']} (score: {r['relevance_score']})")
        print(f"  {r['answer']}\n")