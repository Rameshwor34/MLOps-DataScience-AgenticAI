import argparse
import json
import sys
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.agent import AgentDecisionEngine
from backend.agents.executor import AgentActionExecutor
from backend.services.agentic_service import AgenticService

try:
    from backend.llm.gemini_provider import GeminiProvider
except ImportError:
    GeminiProvider = None

from backend.tests.mock_provider import MockAgentProvider

def run_trace(query, prompt_text, use_real_gemini=False):
    if use_real_gemini and GeminiProvider:
        provider = GeminiProvider()
    else:
        print("Using MockAgentProvider for traces as fallback")
        provider = MockAgentProvider(decisions=[], final_answer="Mocked trace answer")
        
    engine = AgentDecisionEngine(provider=provider, system_prompt=prompt_text)
    service = AgenticService(
        decision_engine=engine,
        executor=AgentActionExecutor(),
        provider=provider,
    )
    return service.process(query)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-version", required=True, help="v1, v2, or v3")
    parser.add_argument("--mock", action="store_true", help="Force mock traces")
    args = parser.parse_args()
    
    prompt_file = ROOT / "prompts" / f"prompt_{args.prompt_version}.txt"
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    queries = {
        "success_trace": "What is the status of order ORD-1003?",
        "multistep_trace": "Can I return ORD-1003, and what does the return policy say?",
        "edge_trace": "Can you check the status of my order?"
    }

    traces_dir = ROOT / "traces" / args.prompt_version
    traces_dir.mkdir(parents=True, exist_ok=True)
    
    use_real = not args.mock

    mlflow.set_experiment("shopassist-agentic-evaluation")
    
    with mlflow.start_run(run_name=f"prompt-{args.prompt_version}-traces"):
        for trace_name, query in queries.items():
            try:
                print(f"Running trace: {trace_name}")
                result = run_trace(query, prompt_text, use_real_gemini=use_real)
                
                trace_data = {
                    "query": query,
                    "prompt_version": args.prompt_version,
                    "trajectory": result.get("trajectory", []),
                    "final_answer": result.get("answer", ""),
                    "termination_reason": result.get("status", "unknown"),
                    "generated_by": "real_gemini" if (use_real and GeminiProvider) else "deterministic_mock"
                }
                
                trace_file = traces_dir / f"{trace_name}.json"
                with open(trace_file, "w", encoding="utf-8") as f:
                    json.dump(trace_data, f, indent=2)
                    
                mlflow.log_artifact(str(trace_file))
                print(f"Saved trace {trace_name} to {trace_file}")
            except Exception as e:
                print(f"Failed trace {trace_name}: {e}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    main()
