# SupportFlow AI

A local, product-style prototype for classifying digital-banking support messages, routing confident predictions to the appropriate team, escalating uncertain cases for human review, and retrieving semantically similar resolved examples.

## What the project demonstrates

- Pretrained `all-MiniLM-L6-v2` sentence embeddings.
- A supervised 77-class logistic-regression intent classifier.
- A TF-IDF baseline for an honest comparison.
- Confidence-based human escalation with a threshold chosen only on validation data.
- Semantic retrieval of similar training examples using normalized embeddings and cosine similarity.
- Held-out evaluation using the official BANKING77 test split.
- A Streamlit product demo with downloadable JSON routing reports.
- A persistent local support-operations dashboard with department queues, ticket statuses, human review, agent corrections, batch CSV intake, and queue exports.
- An explicit out-of-scope review outcome for non-banking requests rather than forcing an incorrect banking category.
- Duplicate batch-import protection using deterministic content fingerprints and exact-message checks.
- Operational analytics for automation, review and resolution rates, department workload, common issues, duplicate messages, and agent review history.
- Explainable Critical/High/Medium/Low priority assignment with first-response SLA deadlines, overdue tracking, queue filters, and SLA-health analytics.
- A versioned FastAPI interface for single and batch predictions, with health checks and interactive OpenAPI documentation.

Priority and SLA targets shown in the app are explicit demo business rules, not ML predictions or contractual commitments. The demo uses clock hours: Critical 1 hour, High 4 hours, Medium 8 hours, and Low 24 hours. The system does not generate banking advice, make account decisions, or contact customers.

## Data and leakage controls

The project uses [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77), an MIT-licensed English text-classification benchmark derived from the BANKING77 research dataset.

- Official training split: divided into stratified training and validation subsets.
- Validation subset: used to select the auto-route confidence threshold.
- Official test split: used only for final evaluation.
- Retrieval examples: sampled only from training data.

Downloaded data, pretrained weights, and trained artifacts are excluded from Git. Never add real customer messages or personal banking information to this repository.

The local operations database is written to `data/supportflow.db` and is also excluded from Git.
Use `samples/ticket_batch.csv` to try the batch workflow without supplying company data. The same file is available from the **Download sample CSV** button in the app.

## Quick start (Apple Silicon, Python 3.12)

```bash
cd /Users/hrushivasava/Documents/AI-ML-Projects/supportflow-ai
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m scripts.train_and_evaluate
streamlit run app.py
```

After the first setup and training run, start the complete interview demo with one command:

```bash
python -m scripts.start_demo
```

This starts both the Streamlit app and FastAPI service, verifies readiness, and opens the demo pages. See `docs/interview-guide.md` for the presentation flow and `docs/architecture.md` for the system design.

The first training run downloads BANKING77 and the pretrained sentence encoder. Later app launches use the saved local model bundle.

Run tests:

```bash
pytest -q
```

## Local API

Start the API from the project root:

```bash
source .venv/bin/activate
python -m scripts.run_api
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation. The main endpoints are:

- `GET /health`: checks whether the trained model artifacts are available.
- `POST /v1/predict`: classifies one message.
- `POST /v1/predict/batch`: classifies up to 50 messages.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"message":"I do not recognise this card payment","similar_count":3}'
```

Prediction endpoints are read-only: they do not add tickets to the local operations database. This API is a local demonstration without authentication; do not expose it directly to the public internet or send real customer data.

## Docker

Train the model locally first so `artifacts/model_bundle/` exists, then run:

```bash
docker compose up --build
```

The app is exposed at `http://127.0.0.1:8503` and the API documentation at `http://127.0.0.1:8000/docs`. Ticket data is stored in a named Docker volume. Docker packaging is provided for portability, but the ordinary Python launcher remains the simplest interview path.

## Outputs

- `artifacts/model_bundle/`: trained classifier, local sentence encoder, retrieval index, and split fingerprints. Git-ignored because it is generated and large.
- `reports/generated/metrics.json`: actual held-out evaluation metrics.
- `reports/generated/confusion_pairs.csv`: most common incorrect intent pairs.
- `reports/generated/test_predictions.csv`: locally generated per-test-example predictions and routing decisions (Git-ignored to keep the repository compact).

## Verified local benchmark

With the pinned environment and random seed included in this repository, the current run produced:

- 92.4% overall accuracy and 92.4% macro F1 on 3,076 official test examples.
- 98.3% top-3 accuracy.
- 97.0% accuracy among automatically routed test tickets at 88.3% coverage.
- 11.7% of test tickets escalated for human review.
- 87.7% test accuracy for the TF-IDF baseline.
- Approximately 2.5 ms mean single-ticket model inference time on the development Mac, excluding app startup.

These are benchmark results, not production banking-performance claims. See `reports/generated/metrics.json` and the per-example prediction file for the exact outputs.

## Responsible-use limitations

- BANKING77 is a research benchmark, not a representative sample of every bank, language, customer, or channel.
- A production model must be validated on the deploying company's consented historical tickets.
- Low confidence does not capture every possible error or unfamiliar request.
- Real deployments require privacy controls, authentication, access logging, monitoring, drift detection, and periodic retraining.
- Automated routing should retain agent correction and human-escalation paths.
- Do not load model artifact files obtained from untrusted sources; Python model serialization can execute code during loading.

## Accurate résumé wording

> Built a banking support-ticket routing prototype using pretrained sentence embeddings and a supervised 77-intent classifier; added validation-selected confidence escalation and semantic retrieval of similar resolved cases, measured routing quality and latency on the official held-out BANKING77 test split, and deployed the workflow in Streamlit.

## Primary references

- [BANKING77 paper](https://aclanthology.org/2020.nlp4convai-1.5/)
- [MTEB BANKING77 dataset card](https://huggingface.co/datasets/mteb/banking77)
- [Sentence Transformers documentation](https://www.sbert.net/docs/sentence_transformer/usage/usage.html)
- [scikit-learn top-k accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.top_k_accuracy_score.html)
