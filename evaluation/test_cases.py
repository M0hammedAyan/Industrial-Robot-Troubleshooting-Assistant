"""Manual test-case checklist (TC01–TC10).

Run interactively in Streamlit after index build, or use notes below.
Record actual outcomes in evaluation/test_case_results.md after demo.
"""

TEST_CASES = [
    {
        "id": "TC01",
        "description": "Known error-code question",
        "action": "Ask: What does error E-123 mean?",
        "expected": "Answer grounded in troubleshooting manual; sources shown",
    },
    {
        "id": "TC02",
        "description": "Maintenance question",
        "action": "Ask: How often should I replace the encoder backup battery?",
        "expected": "Mentions 24 months / BATT1 from maintenance docs",
    },
    {
        "id": "TC03",
        "description": "Safety question",
        "action": "Ask: What is the lockout tagout procedure?",
        "expected": "LOTO steps + safety caution; safety_manual source",
    },
    {
        "id": "TC04",
        "description": "Troubleshooting question",
        "action": "Ask: Why is the robot servo not turning on?",
        "expected": "Structured diagnosis/checks from troubleshooting manual",
    },
    {
        "id": "TC05",
        "description": "Multi-page / multi-doc question",
        "action": "Ask: Relate A-305 to cooling system maintenance checks",
        "expected": "Retrieves troubleshooting + maintenance cooling content",
    },
    {
        "id": "TC06",
        "description": "Unknown question",
        "action": "Ask: What is the maximum payload of Robot XYZ?",
        "expected": "States information not found; no fabricated payload",
    },
    {
        "id": "TC07",
        "description": "Ambiguous question",
        "action": "Ask: The robot has a problem",
        "expected": "Requests clarification (error code / symptoms)",
    },
    {
        "id": "TC08",
        "description": "Upload second manual",
        "action": "Upload an additional PDF via sidebar and ask a question from it",
        "expected": "New content becomes searchable without app restart",
    },
    {
        "id": "TC09",
        "description": "Restart loads FAISS",
        "action": "Stop Streamlit, restart, ask a known question without rebuild",
        "expected": "Existing vector_store loads; answers still work",
    },
    {
        "id": "TC10",
        "description": "Sources/pages displayed",
        "action": "Ask any in-document question and inspect Retrieved Documentation",
        "expected": "Document name, page, relevance, expandable excerpt",
    },
]
