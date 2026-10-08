# Track B: Agentic AI MLOps

## W15 → W16 → W17 Architecture Evolution
- **W15**: Basic FastAPI backend serving stateless LLM chat requests.
- **W16**: Introduced the AgenticService and AgentDecisionEngine. The architecture shifted from a simple completion generation to a bounded reasoning loop, using `MockAgentProvider` to validate agent trajectory logically and deterministically.
- **W17**: Added extensive MLflow tracking, structured dataset testing for regressions, and an Evidently-like "LLM-as-a-judge" flow to continuously monitor the health of prompt iterations and regressions.

## Why `uv` Solves Dependency Problems
This project integrates complex ML and data tools (FastAPI, MLflow, Google GenAI, Chromadb, Sentence-Transformers, Torch). `uv` acts as an extremely fast, deterministic Python package manager written in Rust, resolving native ABI issues (like Chroma/Torch conflicts on Windows) seamlessly, and allowing rapid iteration across different package boundaries without polluting the system-wide Python.

## Reproduction Commands
```bash
uv sync
uv run python scripts/run_evaluation.py --prompt-version v1
uv run python scripts/run_evaluation.py --prompt-version v2
uv run python scripts/run_evaluation.py --prompt-version v3
uv run python scripts/run_real_traces.py --prompt-version v1
uv run python scripts/run_regression.py --prompt-version v1
uv run pytest tests/
```

## MLflow Experiment Strategy
We tracked evaluation runs under the `shopassist-agentic-evaluation` MLflow experiment. We systematically varied the `system_prompt` across 3 versions. The mock-driven evaluations validated baseline deterministic paths, and the regression tests validated if the generated answers degraded against our reference dataset.

## 3 Prompt Version Iteration Story
1. **v1**: Baseline prompt. Lacked proper multi-step verification. Agent tended to terminate prematurely before fully gathering all required context.
2. **v2**: Added a critical Evidence Sufficiency protocol. The agent now properly answered multi-step queries but exhibited over-retrieval (redundant tool calls).
3. **v3**: Introduced strict limits on repeating tool calls for identical entities. It maintained the depth of information from v2 while resolving the trajectory length bloat.

## Regression Testing Approach
- **Golden Dataset**: Hand-curated set of 5 distinct queries across categories with optimal reference responses.
- **Judge**: A Gemini-powered LLM-as-a-judge evaluated the new outputs against the reference text for contradictions and critical missing information.
- **Threshold**: `pct_tests_passed` must be >= 0.80 to promote the prompt version.

## Results
Please view `reports/mlflow_comparison.json` and `reports/prompt_iteration_analysis.md` for full results. Our tests validated that v3 met all correctness constraints and passed regression thresholds.
