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
from backend.tests.mock_provider import MockAgentProvider

from eval.metrics import aggregate_metrics, evaluate_case

DATASET_PATH = ROOT / "eval" / "dataset.json"

def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def run_case(case, prompt_text):
    provider = MockAgentProvider(
        decisions=case["decisions"],
        final_answer=case.get("final_answer", "Mock answer.")
    )
    
    # Inject prompt here
    engine = AgentDecisionEngine(provider=provider, system_prompt=prompt_text)
    
    service = AgenticService(
        decision_engine=engine,
        executor=AgentActionExecutor(),
        provider=provider,
    )
    
    result = service.process(case["query"])
    evaluation = evaluate_case(case, result)
    evaluation["trajectory"] = result.get("trajectory", [])
    evaluation["answer"] = result.get("answer")
    
    return evaluation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-version", required=True, help="v1, v2, or v3")
    args = parser.parse_args()
    
    prompt_file = ROOT / "prompts" / f"prompt_{args.prompt_version}.txt"
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    dataset = load_dataset()
    evaluations = []
    
    mlflow.set_experiment("shopassist-agentic-evaluation")
    
    with mlflow.start_run(run_name=f"prompt-{args.prompt_version}-eval"):
        mlflow.log_params({
            "prompt_version": args.prompt_version,
            "model": "gemini-2.5-flash",
            "temperature": 0.2,
            "top_p": 0.9,
            "max_iterations": 6,
            "chunk_size": 3,
            "n_eval_cases": len(dataset)
        })
        
        for case in dataset:
            try:
                evaluation = run_case(case, prompt_text)
                evaluations.append(evaluation)
            except Exception as exc:
                evaluations.append({
                    "id": case["id"],
                    "category": case["category"],
                    "completion": False,
                    "status_correct": False,
                    "tool_calls_correct": False,
                    "arguments_correct": False,
                    "hard_failure": True,
                    "soft_failure": False,
                    "error": str(exc),
                })

        aggregate = aggregate_metrics(evaluations)
        
        # Log metrics to MLFlow
        mlflow.log_metrics({
            "task_completion_rate": aggregate.get("task_completion_rate", 0),
            "tool_call_correctness": aggregate.get("tool_call_correctness", 0),
            "argument_correctness": aggregate.get("argument_correctness", 0),
            "average_trajectory_length": aggregate.get("average_trajectory_length", 0),
            "average_tool_calls": aggregate.get("average_tool_calls", 0),
            "hard_failures": aggregate.get("hard_failures", 0),
            "soft_failures": aggregate.get("soft_failures", 0),
        })
        
        results_file = ROOT / "eval" / f"results_{args.prompt_version}.json"
        output = {
            "evaluation_type": "deterministic_controller_regression",
            "provider": "MockAgentProvider",
            "metrics": aggregate,
            "cases": evaluations,
        }
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
            
        mlflow.log_artifact(str(results_file))
        mlflow.log_artifact(str(prompt_file))
        
        print(f"Results for {args.prompt_version}:")
        for k, v in aggregate.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
