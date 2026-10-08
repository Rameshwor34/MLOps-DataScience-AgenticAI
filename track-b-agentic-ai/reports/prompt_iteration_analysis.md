# Prompt Iteration Analysis

## Version 1 (v1)
**Observed Failure:**
The v1 prompt lacked explicit evidence-sufficiency checking. In real traces, particularly for multi-step queries like "Can I return ORD-1003, and what does the return policy say?", the agent would terminate early (calling `final_answer`) before it gathered evidence for both parts of the question.

## Version 2 (v2)
**Change made:**
Added a "CRITICAL EVIDENCE SUFFICIENCY RULES" section, explicitly requiring the agent to gather evidence for each distinct part before generating a final answer. 

**Observed Failure:**
While v2 fixed the early termination, it led to over-retrieval. The agent would sometimes call the same tool type redundantly or repeatedly query information that was already present in the evidence state, leading to wasted iterations and sometimes exceeding the 6-iteration budget.

## Version 3 (v3)
**Change made:**
Added a strict "EVIDENCE SUFFICIENCY PROTOCOL" requiring the agent to mentally check if evidence already covers the need before calling a tool. Added rules against repeating tool calls for the same entity and explicitly preserving iterations.

**Final Comparison:**
v3 successfully balances evidence sufficiency with bounded tool use, ensuring multi-part questions are fully answered without exceeding the iteration budget or causing redundant API calls.
