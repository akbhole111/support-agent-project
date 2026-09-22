"""
Structured test scenarios for the support agent.
Each test case is a list of turns (multi-turn conversation),
with the expected tool call(s) for the LAST turn (the one we're grading).
Earlier turns just build context.
"""

TEST_CASES = [
    {
        "id": "T1_order_lookup_basic",
        "turns": ["What's the status of order 2c75c33f103e365438cc19a05d444b7f?"],
        "expected_tool": "get_order_status",
        "notes": "Basic single-turn order lookup"
    },
    {
        "id": "T2_order_missing_id",
        "turns": ["What's the status of my order?"],
        "expected_tool": None,  # should ask for the order ID, not call a tool blindly
        "notes": "Agent should ask for order ID rather than guessing or hallucinating one"
    },
    {
        "id": "T3_policy_question_refund",
        "turns": ["How do I request a refund?"],
        "expected_tool": "search_knowledge_base",
        "notes": "General policy question, well-covered intent"
    },
    {
        "id": "T4_policy_question_thin_coverage",
        "turns": ["How do I cancel a subscription?"],
        "expected_tool": "search_knowledge_base",
        "notes": "Known thin-coverage intent — check if retrieved category matches 'subscription' vs 'order'"
    },
    {
        "id": "T5_multiturn_context_retention",
        "turns": [
            "What's the status of order ab76f54a321a0431ef243b3b6865078b?",
            "Can I get a refund for it?"
        ],
        "expected_tool": "get_order_status",  
        "notes": "Order was canceled — agent may re-check order status or reason from prior context; either is acceptable"
    },
    {
        "id": "T6_escalation_explicit",
        "turns": ["I want to speak to a human, this isn't helping."],
        "expected_tool": "create_support_ticket",
        "notes": "Explicit escalation request with no prior context"
    },
    {
        "id": "T7_escalation_after_failed_help",
        "turns": [
            "How do I cancel a subscription?",
            "That didn't answer my question, I want a human."
        ],
        "expected_tool": "create_support_ticket",
        "notes": "Escalation after knowledge base failed to help — tests whether agent recognizes failure"
    },
    {
        "id": "T8_invalid_order_id",
        "turns": ["What's the status of order XXXX999INVALID?"],
        "expected_tool": "get_order_status",
        "notes": "Should call the tool, get a 'not found', and communicate that clearly rather than making up a status"
    },
    {
        "id": "T9_ambiguous_request",
        "turns": ["Something's wrong with my stuff."],
        "expected_tool": None,
        "notes": "Too vague — agent should ask a clarifying question, not guess a tool"
    },
    {
        "id": "T10_context_switch",
        "turns": [
            "What's the status of order 2c75c33f103e365438cc19a05d444b7f?",
            "Actually never mind, how do I update my shipping address?"
        ],
        "expected_tool": "search_knowledge_base",
        "notes": "Tests whether agent correctly switches topics rather than getting stuck on the earlier order"
    },
]