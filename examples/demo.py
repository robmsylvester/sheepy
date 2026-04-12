#!/usr/bin/env python
"""
Sheepy End-to-End Demo
======================
Demonstrates every major capability of sheepy v1.0 in a single script:

  1. Transformer fine-tuning  (binary SMS spam classifier via Lightning + W&B)
  2. Rich W&B logging         (prediction tables, confusion matrices, dataset artifacts)
  3. KNN vector classification (FAISS index + sentence-transformer embeddings)
  4. RAG classification       (BM25 retrieval + majority-vote)
  5. LLM zero-shot            (via litellm — only runs if OPENAI_API_KEY is set)

Run:
    cd sheepy
    pip install -e ".[dev,vector,rag]"   # add [llm] if you want step 5
    wandb login                          # one-time, links your browser
    python examples/demo.py

After it finishes, open the W&B link printed in the console. You'll see:
  - Scalar charts: train/loss, val/loss, val/F1Score, val/Accuracy, etc.
  - Confusion matrices per epoch
  - A "predictions" table you can filter / sort in the W&B UI
  - A "dataset/summary" table with label distribution
  - A versioned dataset artifact under the Artifacts tab

The script uses a tiny 2-layer BERT (17 MB) so it runs in minutes even on CPU.
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# 0. Setup paths so imports work from the examples/ directory
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")
sys.path.insert(0, EXAMPLES_DIR)
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import torch
import wandb

from sheepy.common.run_progress import (
    configure_demo_logging,
    run_banner,
    run_context,
    run_note,
    run_phase_end,
    run_phase_start,
)
from sheepy.config.schema import SheepyConfig
from sheepy.experiment.base_experiment import Experiment
from sheepy.models import MODEL_MAPPING, TransformerClassifier
from sheepy.data_modules import DATA_MODULE_MAPPING
from sheepy.wandb_utils.experiment_logger import ExperimentLogger
from sheepy.wandb_utils.tables import build_prediction_table, build_dataset_summary_table
from sheepy.wandb_utils.artifacts import log_dataset_artifact

# Register the example data module
from sms_spam_data_module import SmsSpamDataModule
DATA_MODULE_MAPPING["sms_spam_data_module"] = SmsSpamDataModule

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(EXAMPLES_DIR, "config", "demo_config.yaml")
DATA_DIR    = os.path.join(EXAMPLES_DIR, "datasets", "spam")
OUTPUT_DIR  = os.path.join(REPO_ROOT, "output", "demo")


def separator(title: str) -> None:
    """Framed step title with timestamp (used for numbered demo steps)."""
    run_banner(title)


# ===================================================================
# STEP 1 — Train a transformer classifier (Lightning + W&B)
# ===================================================================
def step1_train_transformer():
    separator("STEP 1: Train SMS Spam Classifier (Transformer + Lightning + W&B)")

    run_phase_start("step1.load_config", CONFIG_PATH)
    cfg = SheepyConfig.from_file(CONFIG_PATH)
    config = cfg.to_flat_dict()
    run_phase_end("step1.load_config", "SheepyConfig parsed")

    run_phase_start("step1.apply_paths", "data_dir, output_dir, experiment_name")
    config["data_dir"]        = DATA_DIR
    config["output_dir"]      = OUTPUT_DIR
    config["evaluate"]        = False
    config["tune"]            = False
    config["fp16"]            = False
    config["experiment_name"] = "demo_sms_spam"
    run_phase_end("step1.apply_paths", f"output under {OUTPUT_DIR}")

    run_phase_start("step1.validate_experiment_config", "computing output_dir, precision, n_gpu")
    config = Experiment.validate_experiment_config(config)
    run_note(f"Resolved output_dir: {config['output_dir']}")
    run_phase_end("step1.validate_experiment_config", "ok")

    run_phase_start("step1.Experiment", "W&B logger + config dump")
    experiment = Experiment(config)
    run_phase_end("step1.Experiment", "Experiment object ready")

    experiment.prepare_trainer(SmsSpamDataModule, TransformerClassifier)

    # -- Log dataset artifact + summary to W&B --------------------------
    run_phase_start("step1.wandb_metadata", "ExperimentLogger + tables")
    run = experiment.wandb.experiment
    exp_logger = ExperimentLogger(run, experiment.data.label_encoder.vocab)
    exp_logger.log_experiment_metadata(config)

    class_sizes = experiment.data.train_class_sizes
    exp_logger.log_dataset_summary(class_sizes)
    run_phase_end("step1.wandb_metadata", "dataset summary on run")

    print(f"  Project : {config['project_name']}")
    print(f"  Model   : {config['encoder_model']}")
    print(f"  Epochs  : {config['num_epochs']}")
    print(f"  Labels  : {experiment.data.label_encoder.vocab}")
    print(f"  Train   : {class_sizes}")
    print(f"  W&B URL : {run.get_url()}")

    # -- Train + Test ---------------------------------------------------
    experiment.train()
    experiment.test()

    # -- Log a prediction table from the test set ----------------------
    run_phase_start("step1.collect_test_predictions", "iterating test_dataloader for W&B table")
    test_dl = experiment.data.test_dataloader()
    all_texts, all_true, all_pred, all_conf = [], [], [], []
    experiment.model.eval()
    n_batches = 0

    with torch.no_grad():
        for batch in test_dl:
            inputs, targets, _ = batch
            out = experiment.model(**inputs)
            logits = out["logits"]
            probs = torch.softmax(logits, dim=-1)
            confs, preds = probs.max(dim=-1)

            true_indices = targets["labels"]
            for i in range(len(preds)):
                text = experiment.data.tokenizer.tokenizer.decode(
                    inputs["tokens"][i], skip_special_tokens=True
                )
                all_texts.append(text[:120])
                all_true.append(experiment.data.label_encoder.decode(true_indices[i]))
                all_pred.append(experiment.data.label_encoder.decode(preds[i]))
                all_conf.append(confs[i].item())
            n_batches += 1
            if n_batches == 1 or n_batches % 10 == 0:
                run_note(f"Prediction batches processed: {n_batches}  (rows so far: {len(all_texts)})")

    run_phase_end(
        "step1.collect_test_predictions",
        f"batches={n_batches}  rows={len(all_texts)}",
    )

    run_phase_start("step1.wandb_log_tables", "predictions table + confusion matrix")
    pred_table = build_prediction_table(all_texts, all_true, all_pred, all_conf)
    run.log({"test/predictions": pred_table})

    # Confusion matrix
    exp_logger.log_confusion_matrix(all_true, all_pred, stage="test")
    run_phase_end("step1.wandb_log_tables", "logged test/predictions + confusion matrix")

    # Per-class F1 (as a demo — compute manually)
    from sklearn.metrics import classification_report
    labels = experiment.data.label_encoder.vocab
    run_phase_start("step1.classification_report", "sklearn metrics on CPU")
    print("\n  Classification Report (Test Set):")
    print(classification_report(all_true, all_pred, target_names=labels))
    run_phase_end("step1.classification_report", "done")

    # Update W&B summary with best metrics
    acc = sum(1 for t, p in zip(all_true, all_pred) if t == p) / len(all_true)
    run_phase_start("step1.wandb_finish", "sync + close run")
    exp_logger.update_summary("test/accuracy", acc)

    wandb.finish()
    run_phase_end("step1.wandb_finish", f"final test/accuracy={acc:.4f}")
    run_phase_end("STEP 1", "transformer demo complete")
    return config, experiment


# ===================================================================
# STEP 2 — KNN classification on the same data via vector index
# ===================================================================
def step2_knn_classification():
    separator("STEP 2: KNN Classification (FAISS Vector Index)")

    try:
        import faiss  # noqa: F401
    except ImportError:
        run_note("[SKIP] faiss-cpu not installed. Run: pip install sheepy[vector]")
        run_phase_end("STEP 2", "skipped (no faiss)")
        return

    from sheepy.vector.embedding_classifier import EmbeddingClassifier
    from sheepy.vector.vector_index import VectorIndex
    from sheepy.vector.knn_classifier import KNNClassifier

    run_phase_start("step2.load_sms_corpus", sms_path := os.path.join(DATA_DIR, "SMSSpamCollection"))
    texts, labels = [], []
    with open(sms_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                labels.append(parts[0])
                texts.append(parts[1])
    run_phase_end("step2.load_sms_corpus", f"lines={len(texts)}")

    # Simple 80/20 split
    run_phase_start("step2.split_and_embed", "80/20 split + hash embedding (demo)")
    split = int(len(texts) * 0.8)
    train_texts, test_texts = texts[:split], texts[split:]
    train_labels, test_labels = labels[:split], labels[split:]

    # Use a lightweight embedding function (hash-based, no model download)
    # In production you'd use: EmbeddingClassifier(model_name="all-MiniLM-L6-v2")
    DIM = 64
    def fast_embed(texts_list):
        vecs = []
        for t in texts_list:
            rng = np.random.RandomState(hash(t) % 2**31)
            vecs.append(rng.randn(DIM).astype(np.float32))
        return np.array(vecs)

    embedder = EmbeddingClassifier(embed_fn=fast_embed)
    index = VectorIndex(backend="faiss", dim=DIM, metric="l2")
    knn = KNNClassifier(embedder, index, k=5)

    run_note(f"Indexing {len(train_texts)} training samples into FAISS...")
    knn.fit(train_texts, train_labels)
    run_note(f"Index size: {index.size}")

    run_note(f"Predicting {len(test_texts)} test samples...")
    preds = knn.predict(test_texts)
    acc = sum(1 for t, p in zip(test_labels, preds) if t == p) / len(test_labels)
    run_phase_end("step2.split_and_embed", f"KNN accuracy={acc:.4f}")
    print(f"  KNN Accuracy: {acc:.4f}")

    # Show a few predictions
    run_phase_start("step2.sample_predictions", "first 5 test rows")
    for i in range(min(5, len(test_texts))):
        print(f"    [{test_labels[i]:>4s} -> {preds[i]:>4s}] {test_texts[i][:80]}")
    run_phase_end("STEP 2", "KNN demo complete")


# ===================================================================
# STEP 3 — RAG retrieval-augmented classification (BM25)
# ===================================================================
def step3_rag_classification():
    separator("STEP 3: RAG Classification (BM25 Retrieval + Majority Vote)")

    try:
        from rank_bm25 import BM25Okapi  # noqa: F401
    except ImportError:
        run_note("[SKIP] rank-bm25 not installed. Run: pip install sheepy[rag]")
        run_phase_end("STEP 3", "skipped (no rank_bm25)")
        return

    from sheepy.retrieval.retriever import BM25Retriever
    from sheepy.retrieval.rag_pipeline import RAGPipeline

    run_phase_start("step3.load_corpus", DATA_DIR)
    texts, labels = [], []
    sms_path = os.path.join(DATA_DIR, "SMSSpamCollection")
    with open(sms_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                labels.append(parts[0])
                texts.append(parts[1])
    run_phase_end("step3.load_corpus", f"n={len(texts)}")

    split = int(len(texts) * 0.8)
    train_texts, test_texts = texts[:split], texts[split:]
    train_labels, test_labels = labels[:split], labels[split:]

    run_phase_start("step3.bm25_index", f"{len(train_texts)} docs")
    retriever = BM25Retriever()
    retriever.index(train_texts, train_labels)
    run_phase_end("step3.bm25_index", "BM25Okapi ready")

    pipeline = RAGPipeline(retriever, k=5)

    # Classify a small test set
    n_eval = min(200, len(test_texts))
    run_phase_start("step3.rag_eval", f"classify {n_eval} test messages")
    correct = 0
    for i in range(n_eval):
        result = pipeline.classify(test_texts[i])
        if result["label"] == test_labels[i]:
            correct += 1
        if i == 0 or (i + 1) % 50 == 0:
            run_note(f"RAG eval progress: {i + 1}/{n_eval}")

    acc = correct / n_eval
    run_phase_end("step3.rag_eval", f"accuracy={acc:.4f}")
    print(f"  BM25 RAG Accuracy ({n_eval} samples): {acc:.4f}")

    # Show example retrievals
    example = "WINNER!! You have been selected for a prize"
    result = pipeline.classify(example)
    print(f"\n  Query: '{example}'")
    print(f"  Predicted: {result['label']}")
    print(f"  Retrieved {len(result['retrieved_docs'])} docs:")
    for doc in result["retrieved_docs"][:3]:
        print(f"    [{doc['label']}] {doc['text'][:80]}...")
    run_phase_end("STEP 3", "RAG demo complete")


# ===================================================================
# STEP 4 — LLM zero-shot classification (optional)
# ===================================================================
def step4_llm_classification():
    separator("STEP 4: LLM Zero-Shot Classification (via litellm)")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        run_note("[SKIP] No OPENAI_API_KEY set. Export it to enable this step.")
        run_phase_end("STEP 4", "skipped (no API key)")
        return

    try:
        import litellm  # noqa: F401
    except ImportError:
        run_note("[SKIP] litellm not installed. Run: pip install sheepy[llm]")
        run_phase_end("STEP 4", "skipped (no litellm)")
        return

    from sheepy.llm.prompt_classifier import PromptClassifier

    run_phase_start("step4.build_prompt_classifier", "gpt-4o-mini + few-shot examples")
    clf = PromptClassifier(
        model="gpt-4o-mini",
        labels=["spam", "ham"],
        few_shot_examples=[
            {"text": "WINNER!! Claim your free iPhone now!", "label": "spam"},
            {"text": "Hey, are we still on for lunch tomorrow?", "label": "ham"},
        ],
        temperature=0.0,
    )
    run_phase_end("step4.build_prompt_classifier", "PromptClassifier ready")

    test_messages = [
        "Congratulations! You've won a $1000 gift card. Click here!",
        "Can you pick up some milk on the way home?",
        "FREE entry to our prize draw, text WIN to 80085",
        "Meeting moved to 3pm, see you in the conference room",
        "Ur account has been compromised. Verify at http://scam.link",
    ]

    run_phase_start("step4.llm_calls", f"{len(test_messages)} messages via {clf.model}")
    for idx, msg in enumerate(test_messages, start=1):
        run_note(f"LLM request {idx}/{len(test_messages)} …")
        result = clf.classify(msg)
        print(f"    [{result['label']:>4s}] ({result['latency_ms']:.0f}ms) {msg[:70]}")

    summary = clf.usage_summary
    run_phase_end(
        "step4.llm_calls",
        f"calls={summary['total_calls']}  est_cost_usd={summary['total_cost_usd']:.4f}",
    )
    print(f"\n  Total LLM calls: {summary['total_calls']}")
    print(f"  Total tokens:    {summary['total_prompt_tokens'] + summary['total_completion_tokens']}")
    print(f"  Estimated cost:  ${summary['total_cost_usd']:.4f}")
    run_phase_end("STEP 4", "LLM demo complete")


# ===================================================================
# Main
# ===================================================================
def main():
    configure_demo_logging()
    run_context()
    run_banner("SHEEPY v1.0 — End-to-End Demo (script entry)")
    print("""
    ┌─────────────────────────────────────────────────┐
    │         SHEEPY v1.0 — End-to-End Demo           │
    │                                                 │
    │  This script demonstrates:                      │
    │    1. Transformer training   (Lightning + W&B)  │
    │    2. KNN vector search      (FAISS)            │
    │    3. RAG classification     (BM25)             │
    │    4. LLM zero-shot          (litellm)          │
    └─────────────────────────────────────────────────┘
    Timestamps on lines starting with [date time] show phase boundaries.
    """)

    run_phase_start("main.step1", "Train SMS spam transformer")
    step1_train_transformer()
    run_phase_end("main.step1", "finished")

    run_phase_start("main.step2", "KNN / FAISS")
    step2_knn_classification()
    run_phase_end("main.step2", "finished")

    run_phase_start("main.step3", "RAG / BM25")
    step3_rag_classification()
    run_phase_end("main.step3", "finished")

    run_phase_start("main.step4", "LLM (optional)")
    step4_llm_classification()
    run_phase_end("main.step4", "finished")

    separator("DONE — all demo steps attempted")
    print("  Open your W&B dashboard to explore the results:")
    print("  https://wandb.ai  → project 'sheepy_demo'\n")
    print("  In the W&B UI you will find:")
    print("    • Charts tab    — loss curves, F1, accuracy, AUROC over epochs")
    print("    • Tables tab    — test/predictions table (filter by 'correct')")
    print("    • Tables tab    — dataset/summary table (label distribution)")
    print("    • Artifacts tab — versioned dataset snapshot")
    print("    • System tab    — GPU/CPU utilization, memory")
    print()


if __name__ == "__main__":
    main()
