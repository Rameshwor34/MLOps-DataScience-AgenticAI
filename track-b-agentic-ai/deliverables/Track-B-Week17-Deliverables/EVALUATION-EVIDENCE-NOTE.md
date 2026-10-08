# Week 17 Evaluation Evidence Note

## Regression Evaluation

The Track B regression artifacts preserve the raw results generated for prompt versions v1, v2, and v3.

During the recorded regression evaluation, the configured LLM judge returned HTTP 404 errors because the configured model `gemini-2.5-flash` was no longer available to new users.

Therefore:

- `regression_results_v1.json`
- `regression_results_v2.json`
- `regression_results_v3.json`

must be interpreted as raw evaluation evidence containing judge errors, rather than as fully independent LLM-judge validation.

The files are intentionally preserved unchanged so that the evaluation process remains reproducible and auditable.

The MLflow comparison additionally records the progression of the regression-test metric across v1, v2, and v3. The final prompt version v3 achieved the intended regression-response behavior in the recorded test outputs, while the external judge failure limits the interpretation of that metric.

The main evaluation results in `eval/RESULTS.md` report the agent's deterministic evaluation performance separately from this regression-judge limitation.

## Reproducibility

Prompt versions are preserved under `prompts/`.

Representative agent traces are preserved under `traces/`.

The exported MLflow comparison is preserved under `mlflow/`.

Evidently HTML regression reports are preserved under `evidently/`.

The source repository contains the complete implementation and environment configuration.
