# MLOps, Data Science, and Agentic AI

This repository contains the complete implementation for the Data Science and Agentic AI assignments (W15, W16, and W17). It demonstrates advanced integration of traditional Machine Learning operations (MLOps) with modern Large Language Model (LLM) Agentic workflows.

## Repository Structure

The project is divided into two main tracks:

### 1. `track-a-data-science/`
Focuses on the foundational Data Science and MLOps components.
- **Model Tracking:** Integration with MLflow to track parameters, metrics, and models.
- **Data Engineering:** Preprocessing, exploratory data analysis, and feature engineering.
- **Validation:** Automated evaluation of model drift and data quality.

### 2. `track-b-agentic-ai/`
Focuses on the Agentic AI implementation (ShopAssist AI).
- **Backend Architecture:** Built with FastAPI, integrating tools for checking orders, returning policies, and product details.
- **Agentic Loop:** Implements a bounded LLM reasoning loop (`AgentDecisionEngine`) with strict evidence sufficiency protocols.
- **Evaluation & Prompt Iteration:** Systematically tests multiple prompt versions (v1, v2, v3) against a golden dataset using deterministic mock providers and real `Gemini-3.8-Flash` traces.
- **LLM-as-a-Judge:** Uses Evidently-inspired custom LLM judges to evaluate regression and output quality continuously.
- **Tracking:** Full MLflow logging for metrics, LLM trajectory lengths, tool call correctness, and dataset validations.

## Setup & Installation

This repository uses [`uv`](https://github.com/astral-sh/uv), a lightning-fast Python package installer and resolver written in Rust.

1. **Install `uv`**:
   ```bash
   pip install uv
   ```

2. **Sync Dependencies**:
   Navigate to the respective track directory and run:
   ```bash
   cd track-b-agentic-ai
   uv sync
   ```

3. **Environment Configuration**:
   Copy the `.env.example` file to `.env` in the track folder and populate your keys:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   GEMINI_MODEL=gemini-3.8-flash
   MLFLOW_TRACKING_URI=./mlruns
   MLFLOW_ALLOW_FILE_STORE=true
   ```

## Running Evaluations (Track B)

To execute the LLM evaluation harness and view the prompt iteration traces:
```bash
cd track-b-agentic-ai
uv run python scripts/run_evaluation.py --prompt-version v3
uv run python scripts/run_real_traces.py --prompt-version v3
uv run python scripts/run_regression.py --prompt-version v3
```

For detailed agent iteration results, view `track-b-agentic-ai/reports/mlflow_comparison.json`.

---

## Week 17 Submission — MLOps Deliverables

This repository contains both required Week 17 MLOps tracks.

### Track A — Data Science MLOps

Complete classical ML MLOps pipeline covering:

- `uv` reproducible environment
- MLflow experiment tracking
- 3-model comparison
- model registration and Staging → Production promotion
- FastAPI serving
- Evidently data/target drift monitoring
- synthetic drift experiments
- custom monitoring metrics
- HTML monitoring reports

**[Download Track A Week 17 Deliverables](./track-a-data-science/deliverables/Track-A-Week17-Deliverables.zip)**

### Track B — Agentic AI MLOps

Complete Agentic AI MLOps pipeline covering:

- `uv` reproducible environment
- prompt/configuration versioning
- MLflow experiment tracking
- agent trajectory/trace evaluation
- configuration comparison
- regression evaluation
- Evidently AI monitoring/regression testing
- documentation

**[Download Track B Week 17 Deliverables](./track-b-agentic-ai/deliverables/Track-B-Week17-Deliverables.zip)**

### GitHub Repository

**[MLOps-DataScience-AgenticAI — GitHub Repository](https://github.com/Rameshwor34/MLOps-DataScience-AgenticAI)**

The repository contains the complete source code, `pyproject.toml`, `uv.lock`, implementation documentation, evaluation code, monitoring code, and generated evidence for both tracks.
