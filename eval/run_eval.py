import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "agent"))
from graph import app
from langchain_core.messages import HumanMessage, AIMessage
from test_cases import TEST_CASES
import json

def get_tool_calls_from_result(result):
    """Extract the list of tool names called anywhere in this turn's response."""
    calls = []
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                calls.append(tc["name"])
    return calls

def run_test_case(test_case):
    state = {"messages": []}
    all_turn_calls = []

    for turn_text in test_case["turns"]:
        state["messages"].append(HumanMessage(content=turn_text))
        result = app.invoke(state)
        state = result
        all_turn_calls.append(get_tool_calls_from_result(result))

    final_answer = state["messages"][-1].content
    last_turn_calls = all_turn_calls[-1]
    expected = test_case["expected_tool"]

    if expected is None:
        passed = len(last_turn_calls) == 0
    else:
        passed = expected in last_turn_calls

    return {
        "id": test_case["id"],
        "passed": passed,
        "expected_tool": expected,
        "actual_tools_called": last_turn_calls,
        "final_answer": final_answer,
        "notes": test_case["notes"]
    }

def main():
    results = []
    for tc in TEST_CASES:
        print(f"Running {tc['id']}...")
        result = run_test_case(tc)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] expected={result['expected_tool']} actual={result['actual_tools_called']}\n")

    passed_count = sum(r["passed"] for r in results)
    total = len(results)

    print("=" * 60)
    print(f"RESULTS: {passed_count}/{total} passed ({passed_count/total*100:.0f}%)")
    print("=" * 60)

    for r in results:
        status = "✅" if r["passed"] else "❌"
        print(f"{status} {r['id']}: {r['notes']}")
        if not r["passed"]:
            print(f"    → Expected: {r['expected_tool']}, Got: {r['actual_tools_called']}")
            print(f"    → Answer given: {r['final_answer'][:200]}")

    # Save full results to a JSON file for the writeup
    with open(os.path.join(os.path.dirname(__file__), "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to eval/eval_results.json")

if __name__ == "__main__":
    main()