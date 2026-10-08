# Track B — Week 17 Agentic AI MLOps Deliverables

This package contains the evidence for the Week 17 Track B Agentic AI MLOps implementation.

The complete source implementation is maintained in the parent repository.

## Contents

### 1. MLflow Tracking

`mlflow/mlflow_comparison.json`

Contains the recorded comparison of prompt/configuration versions v1, v2, and v3 for the `shopassist-agentic-evaluation` experiment.

The comparison records:

* task completion rate
* tool-call correctness
* argument correctness
* regression-test pass rate
* hard failures
* selected configuration and selection rationale

The recorded comparison is:

| Version | Task Completion | Tool-Call Correctness | Argument Correctness | Regression Tests Passed | Hard Failures |
| ------- | --------------: | --------------------: | -------------------: | ----------------------: | ------------: |
| v1      |            100% |                   80% |                  80% |                      0% |             0 |
| v2      |            100% |                   80% |                  80% |                     80% |             0 |
| v3      |            100% |                   80% |                  80% |                    100% |             0 |

The recorded selection identifies **v3** as the selected configuration because it maintains the evaluation performance while showing the strongest recorded regression-test behavior.

### 2. Prompt / Configuration Versions

`prompts/`

Contains:

* `prompt_v1.txt`
* `prompt_v2.txt`
* `prompt_v3.txt`

The versions progressively strengthen evidence-sufficiency behavior:

* v1 establishes the basic bounded agentic decision loop.
* v2 adds explicit multi-part evidence requirements.
* v3 adds a strict iteration bound, explicit evidence-sufficiency protocol, duplicate-tool prevention, and stopping criteria.

### 3. Agent Evaluation

`evaluation/`

Contains the complete evaluation evidence, including:

* aggregate evaluation results
* per-version results
* regression results
* failure cases
* W15/W16 comparison

The primary recorded evaluation results are:

* Total cases: 10
* Task completion rate: 100%
* Tool-call correctness: 80%
* Argument correctness: 80%
* Average trajectory length: 2.1
* Average tool calls: 1.1
* Average tokens: 1,976.3
* Average latency: 202.99 ms
* Hard failures: 0
* Soft failures: 0
* Cascading soft failures: 0

The 80% tool-call and argument correctness values are documented as resulting from the two missing-information cases, which intentionally terminate through clarification behavior and therefore do not require tool calls or tool arguments.

### 4. Representative Agent Traces

`traces/`

Contains representative traces for each prompt version:

* success trajectory
* multi-step trajectory
* edge-case trajectory

for v1, v2, and v3.

These traces provide concrete evidence of the agent's bounded decision loop and its selection of subsequent actions based on the evidence available at each step.

### 5. Evidently AI Reports

`evidently/`

Contains the generated Evidently AI HTML regression-monitoring reports for the prompt/configuration iterations.

These reports provide visual monitoring evidence for regression behavior across the prompt versions.

### 6. Evaluation Evidence Note

`EVALUATION-EVIDENCE-NOTE.md`

Documents an important limitation in the recorded regression evaluation.

The configured external LLM judge returned HTTP 404 errors because the configured `gemini-2.5-flash` model was no longer available to new users.

The raw regression JSON files are therefore preserved as audit evidence, but their judge verdict fields should not be interpreted as successful independent LLM-judge validation.

The recorded response outputs and MLflow comparison are preserved unchanged so that the evaluation history remains transparent and reproducible.

### 7. Reproducible Environment

The package includes:

* `pyproject.toml`
* `uv.lock`

These files define the Track B Python environment and dependency lock state.

The complete source repository contains the W15, W16, and Week 17 implementation.

## Repository

GitHub repository:

https://github.com/Rameshwor34/MLOps-DataScience-AgenticAI

The repository contains the complete implementation, source code, environment configuration, evaluation framework, prompts, traces, monitoring reports, and Week 17 documentation.
