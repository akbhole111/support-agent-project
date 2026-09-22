import sys, os
sys.path.append(os.path.dirname(__file__))
from graph import app
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def chat():
    state = {"messages": []}
    print("Support agent ready. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        state["messages"].append(HumanMessage(content=user_input))
        result = app.invoke(state)

        # Show which tools were called this turn, in order
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    print(f"  [tool call] {tc['name']}({tc['args']})")
            if isinstance(msg, ToolMessage):
                print(f"  [tool result] {msg.content[:150]}...")

        state = result
        print("Agent:", state["messages"][-1].content, "\n")

if __name__ == "__main__":
    chat()