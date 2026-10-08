import pytest
from backend.agents.state import AgentState

def test_agent_state_initialization():
    state = AgentState(user_query="Hello")
    assert state.user_query == "Hello"
    assert state.iteration == 0
    assert state.status == "running"

def test_agent_marks_completed():
    state = AgentState(user_query="Test")
    state.mark_completed("Done")
    assert state.status == "completed"
    assert state.final_answer == "Done"

def test_agent_marks_max_iterations():
    state = AgentState(user_query="Test")
    state.iteration = 6
    state.mark_max_iterations()
    assert state.status == "max_iterations"
