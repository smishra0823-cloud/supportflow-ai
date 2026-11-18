# SupportFlow AI interview guide

## 30-second explanation

SupportFlow AI is a banking support-ticket routing prototype. It uses MiniLM sentence embeddings and a supervised logistic-regression classifier trained on BANKING77 to identify 77 issue types. Confident predictions are routed automatically, uncertain cases go to human review, and agents can correct the result. The product layer adds similar-ticket retrieval, operational queues, duplicate protection, priority and SLA rules, analytics, batch intake, and a FastAPI interface.

## Start the complete demo

```bash
cd /Users/hrushivasava/Documents/AI-ML-Projects/supportflow-ai
source .venv/bin/activate
python -m scripts.start_demo
```

The command starts the Streamlit app and FastAPI service, checks that both are ready, and opens the app and API documentation. Press `Ctrl+C` in that terminal to stop services started by the command.

## Five-minute demonstration

1. Open the Streamlit app at `http://127.0.0.1:8503`.
2. Classify: “I do not recognise this card payment.”
3. Explain the predicted issue, confidence, department, High priority, four-hour SLA, top-three predictions, and similar cases.
4. Add the ticket to the queue and open the Operations dashboard.
5. Show status, department, priority and SLA filters, then explain human review for low-confidence cases.
6. Open Analytics and show automation, resolution, priority, and SLA-health metrics.
7. Open `http://127.0.0.1:8000/docs`, expand `POST /v1/predict`, and explain how another product could integrate the model.

## Key technical decisions

- **Why MiniLM?** It provides compact semantic embeddings and fast CPU inference suitable for a local prototype.
- **Why logistic regression?** It is fast, interpretable, and gives class probabilities needed for confidence-based routing.
- **Why a validation threshold?** Overall accuracy alone does not control automation risk. The threshold trades coverage for higher accuracy among automatically routed tickets.
- **Why human review?** Low confidence and out-of-scope inputs should not be forced into an automated department.
- **Why retrieval?** Similar examples provide context without generating unsupported banking advice.
- **Why rule-based priority and SLA?** The benchmark has intent labels but no reliable priority or SLA ground truth, so explicit auditable rules are more honest than pretending these are learned predictions.
- **Why SQLite?** It keeps the prototype self-contained. A production system would use a managed transactional database.

## Verified results to remember

- Held-out test accuracy: approximately 92.4%
- Macro F1: approximately 92.4%
- Top-three accuracy: approximately 98.3%
- Auto-route accuracy: approximately 97.0%
- Auto-route coverage: approximately 88.3%
- Mean model inference measured during evaluation: approximately 2.5 ms, excluding application startup

These are benchmark results, not claims about production banking performance.

## Honest limitations

- The dataset is English and research-oriented.
- It does not represent every bank, customer population, channel, or emerging issue.
- Confidence cannot detect every error or unknown request.
- The API is a local demo without authentication.
- A real bank would need privacy review, security controls, monitoring, production data validation, and periodic retraining.

## If the interviewer asks what you would build next

1. Authenticate users and add role-based access controls.
2. Integrate with a real ticket platform through consented APIs.
3. Validate and retrain with anonymized company-specific tickets.
4. Add monitoring for latency, drift, class distribution, automation accuracy, and SLA breaches.
5. Deploy behind HTTPS with a managed database and secrets management.
