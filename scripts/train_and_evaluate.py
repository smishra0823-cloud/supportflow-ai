from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from supportflow.confidence import choose_auto_route_threshold
from supportflow.evaluation import classification_metrics, most_common_confusions


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "model_bundle"
REPORT_DIR = ROOT / "reports" / "generated"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_SEED = 20261118


def _fingerprint(texts: list[str], labels: list[str]) -> str:
    digest = hashlib.sha256()
    for text, label in zip(texts, labels, strict=True):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _encode(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)


def main() -> None:
    dataset = load_dataset("mteb/banking77")
    official_train = pd.DataFrame(dataset["train"])
    official_test = pd.DataFrame(dataset["test"])
    train, validation = train_test_split(
        official_train,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=official_train["label_text"],
    )
    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = official_test.reset_index(drop=True)

    baseline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=1_500, C=4.0),
    )
    baseline.fit(train["text"], train["label_text"])
    baseline_predictions = baseline.predict(test["text"])

    encoder = SentenceTransformer(MODEL_NAME, device="cpu")
    train_embeddings = _encode(encoder, train["text"].tolist())
    validation_embeddings = _encode(encoder, validation["text"].tolist())
    embedding_start = perf_counter()
    test_embeddings = _encode(encoder, test["text"].tolist())
    batch_embedding_ms_per_ticket = (perf_counter() - embedding_start) * 1000 / len(test)

    classifier = LogisticRegression(max_iter=2_000, C=8.0)
    classifier.fit(train_embeddings, train["label_text"].to_numpy())
    validation_probabilities = classifier.predict_proba(validation_embeddings)
    threshold_info = choose_auto_route_threshold(
        validation["label_text"].to_numpy(),
        validation_probabilities,
        classes=classifier.classes_,
        target_accuracy=0.97,
        minimum_coverage=0.20,
    )

    start = perf_counter()
    test_probabilities = classifier.predict_proba(test_embeddings)
    classifier_ms_per_ticket = (perf_counter() - start) * 1000 / len(test)
    metrics = classification_metrics(
        test["label_text"].to_numpy(),
        test_probabilities,
        classifier.classes_,
        threshold_info["threshold"],
    )
    predicted = classifier.classes_[test_probabilities.argmax(axis=1)]
    single_ticket_latencies: list[float] = []
    for message in test["text"].iloc[:100]:
        latency_start = perf_counter()
        one_embedding = encoder.encode(
            [message], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        classifier.predict_proba(one_embedding)
        single_ticket_latencies.append((perf_counter() - latency_start) * 1000)
    metrics.update(
        {
            "dataset": "mteb/banking77",
            "dataset_license": "MIT",
            "embedding_model": MODEL_NAME,
            "classifier": "multinomial logistic regression",
            "train_examples": int(len(train)),
            "validation_examples": int(len(validation)),
            "baseline_tfidf_accuracy": float(accuracy_score(test["label_text"], baseline_predictions)),
            "baseline_tfidf_macro_f1": float(f1_score(test["label_text"], baseline_predictions, average="macro")),
            "classifier_only_ms_per_ticket": float(classifier_ms_per_ticket),
            "batch_embedding_ms_per_ticket": float(batch_embedding_ms_per_ticket),
            "single_ticket_mean_inference_ms": float(np.mean(single_ticket_latencies)),
            "single_ticket_p95_inference_ms": float(np.percentile(single_ticket_latencies, 95)),
            "validation_auto_route_accuracy": threshold_info["validation_auto_route_accuracy"],
            "validation_auto_route_coverage": threshold_info["validation_coverage"],
            "random_seed": RANDOM_SEED,
            "test_split_policy": "Official BANKING77 test split; untouched during training and threshold selection.",
        }
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, ARTIFACT_DIR / "classifier.joblib")
    encoder.save_pretrained(str(ARTIFACT_DIR / "embedding_model"))
    retrieval_count = min(4_000, len(train))
    retrieval_rows = train.sample(n=retrieval_count, random_state=RANDOM_SEED).sort_index()
    retrieval_embeddings = train_embeddings[retrieval_rows.index.to_numpy()]
    np.savez_compressed(
        ARTIFACT_DIR / "retrieval_embeddings.npz",
        embeddings=retrieval_embeddings.astype(np.float16),
    )
    retrieval_rows[["text", "label_text"]].reset_index(drop=True).to_csv(
        ARTIFACT_DIR / "resolved_tickets.csv", index=False
    )
    metadata = {
        "format_version": 1,
        "embedding_model": MODEL_NAME,
        "auto_route_threshold": threshold_info["threshold"],
        "train_fingerprint_sha256": _fingerprint(train["text"].tolist(), train["label_text"].tolist()),
        "validation_fingerprint_sha256": _fingerprint(
            validation["text"].tolist(), validation["label_text"].tolist()
        ),
        "test_fingerprint_sha256": _fingerprint(test["text"].tolist(), test["label_text"].tolist()),
        "dataset_license": "MIT",
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    confusions = most_common_confusions(test["label_text"].to_numpy(), predicted)
    pd.DataFrame(confusions).to_csv(REPORT_DIR / "confusion_pairs.csv", index=False)
    predictions = test[["text", "label_text"]].copy()
    predictions["predicted_intent"] = predicted
    predictions["confidence"] = test_probabilities.max(axis=1)
    predictions["auto_route"] = predictions["confidence"] >= threshold_info["threshold"]
    predictions.to_csv(REPORT_DIR / "test_predictions.csv", index=False)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
