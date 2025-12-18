"""
experiment.py — News & Market Sentiment Analytics (Exam Project)
===============================================================

Expected data/CSV at: data/sp500_news.csv with columns:
- Date  : trading date
- Title : news headline
- CP    : S&P 500 closing price

Models:
1) Naive baseline (always predict up)
2) TF-IDF bag-of-words + Logistic Regression
3) Custom financial dictionary (log-odds) + Logistic Regression
4) Transformer sentiment benchmark (FinBERT) + Logistic Regression on daily sentiment
5) Fine-tuned FinBERT sequence classifier (freeze most layers, unfreeze top N) + daily aggregation

Evaluation:
- Time-series split: train <= TRAIN_END, test >= TEST_START
- Metrics: Accuracy, Balanced Accuracy, AUC
- Rolling AUC on test set (drift)

Outputs:
- results/model_summary.csv
- results/backtest.csv
- results/figures/metrics_bar.png
- results/figures/rolling_auc.png
- results/figures/prob_distributions.png
- results/figures/backtest_curve.png

Notes on fine-tuning benchmark:
- We weakly supervise headline-level labels using the daily next-day direction (up_fwd) for that date.
- We cap headlines/day consistently with the OOTB transformer benchmark to reduce distribution shift.
- We fine-tune only on TRAIN dates, then infer only on TEST dates to avoid leakage.
"""

from __future__ import annotations

import os
import re
import math
import time
import string
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List

# Workaround for occasional Windows OpenMP duplication issues
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")


# Logging with duplicate-suppression
_LOG_LAST_PRINT: Dict[str, float] = {}
_LOG_SUPPRESS_SECONDS = 30.0


def log_progress(step: str) -> None:
    """Print timestamped progress messages but suppress rapid duplicates.

    If the same message was printed within `_LOG_SUPPRESS_SECONDS`, skip printing.
    This reduces console spam when the same status is emitted repeatedly.
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    now = time.time()
    last = _LOG_LAST_PRINT.get(step)
    if last is not None and (now - last) < _LOG_SUPPRESS_SECONDS:
        return
    _LOG_LAST_PRINT[step] = now
    print(f"[{ts}] {step}")


# Config
@dataclass
class Config:
    DATA_PATH: str = os.path.join("data", "sp500_news.csv")
    RESULTS_DIR: str = "results"
    FIG_DIR: str = os.path.join("results", "figures")
    SUMMARY_CSV: str = os.path.join("results", "model_summary.csv")
    BACKTEST_CSV: str = os.path.join("results", "backtest.csv")

    TRAIN_END: str = "2018-12-31"
    TEST_START: str = "2021-01-01"

    # TF–IDF
    MIN_DF: int = 5
    NGRAM_RANGE: Tuple[int, int] = (1, 2)
    MAX_FEATURES: int = 50_000

    # Dictionary
    LEX_MIN_DF: int = 5
    LEX_TOPK: int = 3000
    LEX_ALPHA: float = 1.0

    # Transformer (OOTB)
    USE_TRANSFORMER: bool = True
    TRANSFORMER_MODEL: str = "ProsusAI/finbert"  # alt: "yiyanghkust/finbert-tone"
    MAX_HEADLINES_PER_DAY: int = 25
    TRANSFORMER_BATCH_SIZE: int = 64

    # Extra diagnostics
    DO_WALK_FORWARD: bool = True
    WALK_N_SPLITS: int = 5
    RUN_BENCHMARK: bool = True

    # Fine-tuned Transformer
    USE_FINETUNED_TRANSFORMER: bool = True
    FINETUNE_OUTPUT_DIR: str = os.path.join("results", "finbert_finetuned")
    FINETUNE_MAX_LENGTH: int = 64
    FINETUNE_EPOCHS: int = 2
    FINETUNE_LR: float = 2e-5
    FINETUNE_WEIGHT_DECAY: float = 0.01
    FINETUNE_WARMUP_RATIO: float = 0.1
    FINETUNE_UNFREEZE_LAST_N_LAYERS: int = 2
    FINETUNE_GRAD_ACCUM_STEPS: int = 1
    FINETUNE_SEED: int = 42

    # Rolling AUC
    ROLLING_WINDOW: int = 60

    # Backtest
    INCLUDE_BACKTEST: bool = True
    TRANSACTION_COST_BPS: float = 1.0

    # Classification threshold
    USE_BASE_RATE_THRESHOLD: bool = True

    # Diagnostics
    PRINT_TORCH_INFO: bool = True
    PRINT_TRANSFORMER_LABELS: bool = True


CFG = Config()


# Data utilities
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(csv_path: str) -> pd.DataFrame:
    log_progress(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)

    df = df.rename(columns={"Date": "date", "Title": "headline", "CP": "close"})
    required = {"date", "headline", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found: {df.columns.tolist()}"
        )

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["headline"] = df["headline"].astype(str)
    df["clean_headline"] = df["headline"].map(clean_text)

    log_progress(
        f"Loaded {len(df):,} rows spanning {df['date'].min().date()} → {df['date'].max().date()}"
    )
    return df


def make_daily_docs(df: pd.DataFrame) -> pd.DataFrame:
    log_progress("Aggregating headlines → daily documents ...")
    daily = (
        df.groupby("date")["clean_headline"]
        .apply(lambda x: " ".join(x))
        .reset_index()
        .rename(columns={"clean_headline": "doc"})
    )
    log_progress(f"Created {len(daily):,} daily documents")
    return daily


def add_targets(daily: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    log_progress("Computing next-day targets ...")
    prices = df[["date", "close"]].drop_duplicates("date").sort_values("date").copy()
    prices["ret1"] = np.log(prices["close"]).diff()
    prices["ret1_fwd"] = prices["ret1"].shift(-1)
    prices["up_fwd"] = (prices["ret1_fwd"] > 0).astype(int)

    panel = daily.merge(
        prices[["date", "ret1", "ret1_fwd", "up_fwd"]], on="date", how="inner"
    )
    panel = panel.dropna(subset=["ret1_fwd", "up_fwd"]).reset_index(drop=True)
    log_progress(f"Panel size: {len(panel):,} days")
    return panel


def split_time(
    panel: pd.DataFrame, train_end: str, test_start: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = panel[panel["date"] <= pd.to_datetime(train_end)].copy()
    test = panel[panel["date"] >= pd.to_datetime(test_start)].copy()
    log_progress(
        f"Train: {len(train):,} days (≤ {train_end}) | Test: {len(test):,} days (≥ {test_start})"
    )
    log_progress(f"Test up-rate: {test['up_fwd'].mean():.3f}")
    return train, test


# Evaluation
def evaluate_probs(
    y_true: np.ndarray, prob: np.ndarray, threshold: float
) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)

    # Confusion matrix layout for labels [0,1]:
    # [[TN, FP],
    #  [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    # Standard (positive class = 1, i.e., "up") metrics
    precision_up = precision_score(y_true, pred, zero_division=0)
    recall_up = recall_score(y_true, pred, zero_division=0)
    f1_up = f1_score(y_true, pred, zero_division=0)

    # Down-class metrics (treat class 0 = "down" as the positive class)
    recall_down = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision_down = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1_down = (
        2 * precision_down * recall_down / (precision_down + recall_down)
        if (precision_down + recall_down) > 0
        else 0.0
    )

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_up),
        "recall": float(recall_up),
        "f1": float(f1_up),
        "auc": float(roc_auc_score(y_true, prob)),
        "precision_down": float(precision_down),
        "recall_down": float(recall_down),
        "f1_down": float(f1_down),
    }


def bootstrap_ci(
    values: List[float], n_boot: int = 1000, alpha: float = 0.05
) -> Tuple[float, float]:
    """Compute bootstrap CI for the mean of `values`."""
    vals = np.asarray(values)
    if len(vals) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        boots.append(sample.mean())
    lo = np.percentile(boots, 100 * (alpha / 2))
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def walk_forward_evaluate(
    panel: pd.DataFrame, raw_df: pd.DataFrame, cfg: Config, n_splits: int = 5
) -> pd.DataFrame:
    """Run an expanding-window walk-forward evaluation and return summary metrics with bootstrap CIs.

    Splits: divide unique dates into n_splits+1 contiguous blocks; for i in 1..n_splits use
    train = concat(blocks[:i]) and test = blocks[i].
    """
    dates = np.array(sorted(panel["date"].unique()))
    blocks = np.array_split(dates, n_splits + 1)
    results = {
        "model": [],
        "fold": [],
        "accuracy": [],
        "balanced_accuracy": [],
        "auc": [],
    }

    for i in range(1, len(blocks)):
        train_dates = np.concatenate(blocks[:i])
        test_dates = blocks[i]
        train = panel[panel["date"].isin(train_dates)].reset_index(drop=True)
        test = panel[panel["date"].isin(test_dates)].reset_index(drop=True)
        if len(train) < 10 or len(test) < 1:
            continue

        # Transformer
        if cfg.USE_TRANSFORMER:
            prob_tfm = model_transformer(train, test, raw_df, cfg)
            m = evaluate_probs(
                test["up_fwd"].values, prob_tfm, threshold=float(train["up_fwd"].mean())
            )
            results["model"].append("Transformer")
            results["fold"].append(i)
            results["accuracy"].append(m["accuracy"])
            results["balanced_accuracy"].append(m["balanced_accuracy"])
            results["auc"].append(m["auc"])

        # TF-IDF
        prob_tfidf = model_tfidf(train, test, cfg)
        m = evaluate_probs(
            test["up_fwd"].values, prob_tfidf, threshold=float(train["up_fwd"].mean())
        )
        results["model"].append("TFIDF")
        results["fold"].append(i)
        results["accuracy"].append(m["accuracy"])
        results["balanced_accuracy"].append(m["balanced_accuracy"])
        results["auc"].append(m["auc"])

        # Dictionary
        prob_lex = model_dictionary(train, test, cfg)
        m = evaluate_probs(
            test["up_fwd"].values, prob_lex, threshold=float(train["up_fwd"].mean())
        )
        results["model"].append("Dictionary")
        results["fold"].append(i)
        results["accuracy"].append(m["accuracy"])
        results["balanced_accuracy"].append(m["balanced_accuracy"])
        results["auc"].append(m["auc"])

    df = pd.DataFrame(results)

    # Aggregate per-model mean and bootstrap CI
    rows = []
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        acc_lo, acc_hi = bootstrap_ci(sub["accuracy"].values)
        ba_lo, ba_hi = bootstrap_ci(sub["balanced_accuracy"].values)
        auc_lo, auc_hi = bootstrap_ci(sub["auc"].values)
        rows.append(
            {
                "model": model,
                "accuracy_mean": sub["accuracy"].mean(),
                "accuracy_lo": acc_lo,
                "accuracy_hi": acc_hi,
                "balanced_accuracy_mean": sub["balanced_accuracy"].mean(),
                "balanced_accuracy_lo": ba_lo,
                "balanced_accuracy_hi": ba_hi,
                "auc_mean": sub["auc"].mean(),
                "auc_lo": auc_lo,
                "auc_hi": auc_hi,
                "folds": len(sub),
            }
        )

    return pd.DataFrame(rows)


def benchmark_runtime(
    train: pd.DataFrame, test: pd.DataFrame, raw_df: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Measure wall-time and per-sample latency for each model on the given train/test split."""
    timings = []
    n_samples = max(1, len(test))

    def time_wrapper(name: str, fn, *args, **kwargs):
        t0 = time.time()
        try:
            _ = fn(*args, **kwargs)
            ok = True
        except Exception as e:
            log_progress(f"[BENCH] {name} failed during run: {e}")
            ok = False
        t1 = time.time()
        timings.append(
            {
                "model": name,
                "time_s": t1 - t0,
                "per_sample_s": (t1 - t0) / n_samples,
                "ok": ok,
            }
        )

    # Transformer (OOTB)
    if cfg.USE_TRANSFORMER:
        time_wrapper("Transformer", model_transformer, train, test, raw_df, cfg)

    # Fine-tuned Transformer
    if cfg.USE_FINETUNED_TRANSFORMER:
        time_wrapper(
            "Transformer (finetuned)",
            model_transformer_finetuned,
            train,
            test,
            raw_df,
            cfg,
        )

    # TF-IDF
    time_wrapper("TFIDF", model_tfidf, train, test, cfg)

    # Dictionary
    time_wrapper("Dictionary", model_dictionary, train, test, cfg)

    # Naive baseline
    def run_naive():
        _p, _m = naive_up(test)
        return _p

    time_wrapper("Naive", run_naive)

    df = pd.DataFrame(timings)
    log_progress("Runtime benchmark complete")
    return df


def naive_up(test: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
    y = test["up_fwd"].values
    # Predict the majority class (most common direction) for the naive baseline
    majority = int(y.mean() >= 0.5)
    prob = np.full(len(y), y.mean())
    preds = np.full(len(y), majority)
    return prob, {
        "accuracy": float(accuracy_score(y, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "auc": float(roc_auc_score(y, prob)),
    }


# Model 1: TF–IDF
def model_tfidf(train: pd.DataFrame, test: pd.DataFrame, cfg: Config) -> np.ndarray:
    log_progress("Training TF–IDF + Logistic Regression ...")
    vec = TfidfVectorizer(
        min_df=cfg.MIN_DF,
        ngram_range=cfg.NGRAM_RANGE,
        max_features=cfg.MAX_FEATURES,
    )
    Xtr = vec.fit_transform(train["doc"])
    Xte = vec.transform(test["doc"])

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, train["up_fwd"].values)
    prob = clf.predict_proba(Xte)[:, 1]

    log_progress("TF–IDF model complete")
    return prob


# Model 2: Dictionary (log-odds)
def build_lexicon_logodds(
    docs: pd.Series,
    labels: np.ndarray,
    min_df: int,
    alpha: float,
    topk: int,
) -> Dict[str, float]:
    cv = CountVectorizer(min_df=min_df)
    X = cv.fit_transform(docs)
    vocab = np.array(cv.get_feature_names_out())

    y = labels.astype(int)
    pos = y == 1
    neg = y == 0

    c_pos = np.asarray(X[pos].sum(axis=0)).ravel()
    c_neg = np.asarray(X[neg].sum(axis=0)).ravel()

    log_odds = np.log((c_pos + alpha) / (c_neg + alpha))
    order = np.argsort(log_odds)
    keep = np.concatenate([order[:topk], order[-topk:]])
    return {vocab[i]: float(log_odds[i]) for i in keep}


def score_doc_lexicon(doc: str, lexicon: Dict[str, float]) -> float:
    toks = doc.split()
    if not toks:
        return 0.0
    s = 0.0
    for t in toks:
        s += lexicon.get(t, 0.0)
    return s / math.sqrt(len(toks) + 1)


def model_dictionary(
    train: pd.DataFrame, test: pd.DataFrame, cfg: Config
) -> np.ndarray:
    log_progress("Training Dictionary (log-odds) model ...")
    lex = build_lexicon_logodds(
        docs=train["doc"],
        labels=train["up_fwd"].values,
        min_df=cfg.LEX_MIN_DF,
        alpha=cfg.LEX_ALPHA,
        topk=cfg.LEX_TOPK,
    )
    log_progress(f"Lexicon size: {len(lex):,}")

    Xtr = np.array([score_doc_lexicon(d, lex) for d in train["doc"]]).reshape(-1, 1)
    Xte = np.array([score_doc_lexicon(d, lex) for d in test["doc"]]).reshape(-1, 1)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, train["up_fwd"].values)
    prob = clf.predict_proba(Xte)[:, 1]

    log_progress("Dictionary model complete")
    return prob


# Model 3: Transformer (FinBERT) OOTB sentiment + LR
def finbert_to_signed_score(o: Dict) -> float:
    lab = str(o.get("label", "")).strip().lower()
    sc = float(o.get("score", 0.0))
    if "pos" in lab:
        return sc
    if "neg" in lab:
        return -sc
    return 0.0


def compute_daily_transformer_sentiment(
    raw_df: pd.DataFrame, dates: pd.Series, cfg: Config
) -> pd.DataFrame:
    import torch
    from transformers import pipeline

    if cfg.PRINT_TORCH_INFO:
        log_progress(
            f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}, torch_cuda={torch.version.cuda}"
        )

    device = 0 if torch.cuda.is_available() else -1
    if device == -1:
        log_progress(
            "[WARN] CUDA not available in this environment; transformer will run on CPU."
        )
    else:
        log_progress(f"Using GPU device 0: {torch.cuda.get_device_name(0)}")

    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=cfg.TRANSFORMER_MODEL,
        tokenizer=cfg.TRANSFORMER_MODEL,
        truncation=True,
        device=device,
    )

    df = raw_df[raw_df["date"].isin(dates)].copy()
    df["rn"] = df.groupby("date").cumcount()
    df = df[df["rn"] < cfg.MAX_HEADLINES_PER_DAY].copy()

    texts = df["clean_headline"].tolist()
    if len(texts) == 0:
        return pd.DataFrame({"date": dates.unique(), "sent_score": 0.0})

    outputs = sentiment_pipe(texts, batch_size=cfg.TRANSFORMER_BATCH_SIZE)

    if cfg.PRINT_TRANSFORMER_LABELS:
        labs = sorted(set(o.get("label") for o in outputs[:200]))
        log_progress(f"Transformer example labels: {labs}")

    df["sent_score"] = [finbert_to_signed_score(o) for o in outputs]
    daily = df.groupby("date")["sent_score"].mean().reset_index()
    return daily


def model_transformer(
    train: pd.DataFrame, test: pd.DataFrame, raw_df: pd.DataFrame, cfg: Config
) -> np.ndarray:
    log_progress("Training Transformer (FinBERT) benchmark ...")
    dates_union = pd.concat([train["date"], test["date"]]).drop_duplicates()

    daily_sent = compute_daily_transformer_sentiment(raw_df, dates_union, cfg)
    train2 = train.merge(daily_sent, on="date", how="left").fillna(0.0)
    test2 = test.merge(daily_sent, on="date", how="left").fillna(0.0)

    log_progress(f"Transformer train sent_score std: {train2['sent_score'].std():.6f}")
    log_progress(f"Transformer test  sent_score std: {test2['sent_score'].std():.6f}")

    clf = LogisticRegression(max_iter=2000)
    clf.fit(train2[["sent_score"]].values, train2["up_fwd"].values)
    prob = clf.predict_proba(test2[["sent_score"]].values)[:, 1]

    log_progress(
        f"Transformer prob std: {prob.std():.6f} | min/max: {prob.min():.6f}/{prob.max():.6f}"
    )
    log_progress("Transformer model complete")
    return prob


# Model 4: Fine-tuned FinBERT (sequence classifier) + daily aggregation
def make_headline_dataset(
    raw_df: pd.DataFrame,
    panel: pd.DataFrame,
    dates: pd.Series,
    max_headlines_per_day: int,
) -> pd.DataFrame:
    # Map daily labels onto headlines (weak supervision)
    y_map = panel.set_index("date")["up_fwd"].to_dict()

    df = raw_df[raw_df["date"].isin(dates)].copy()
    df["label"] = df["date"].map(y_map)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    # Cap headlines/day
    df["rn"] = df.groupby("date").cumcount()
    df = df[df["rn"] < max_headlines_per_day].copy()

    return df[["date", "clean_headline", "label"]].reset_index(drop=True)


def model_transformer_finetuned(
    train: pd.DataFrame, test: pd.DataFrame, raw_df: pd.DataFrame, cfg: Config
) -> np.ndarray:
    log_progress("Training Fine-tuned Transformer (FinBERT) benchmark ...")

    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
        set_seed,
    )

    set_seed(cfg.FINETUNE_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log_progress("[WARN] CUDA not available; fine-tuning will be slow on CPU.")

    # Use panel for label lookup (includes up_fwd); concatenate is safe here
    panel_all = pd.concat([train, test], axis=0)

    df_tr = make_headline_dataset(
        raw_df=raw_df,
        panel=panel_all,
        dates=train["date"],
        max_headlines_per_day=cfg.MAX_HEADLINES_PER_DAY,
    )

    df_te = make_headline_dataset(
        raw_df=raw_df,
        panel=panel_all,
        dates=test["date"],
        max_headlines_per_day=cfg.MAX_HEADLINES_PER_DAY,
    )

    tok = AutoTokenizer.from_pretrained(cfg.TRANSFORMER_MODEL)
    # ProsusAI/finbert is a 3-class sentiment model (pos/neu/neg). For binary fine-tuning
    # we replace the classification head (2 labels) and ignore the checkpoint head size mismatch.
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.TRANSFORMER_MODEL,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    # Freeze all params
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze classifier head
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True

    # Unfreeze last N encoder layers (BERT-style)
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        layers = model.bert.encoder.layer
        n = min(cfg.FINETUNE_UNFREEZE_LAST_N_LAYERS, len(layers))
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
    else:
        log_progress(
            "[WARN] Unexpected model architecture; could not unfreeze last N layers."
        )

    class TextDS(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, i):
            enc = tok(
                self.texts[i],
                truncation=True,
                max_length=cfg.FINETUNE_MAX_LENGTH,
            )
            enc["labels"] = int(self.labels[i])
            return enc

    tr_ds = TextDS(df_tr["clean_headline"].tolist(), df_tr["label"].values)
    te_ds = TextDS(df_te["clean_headline"].tolist(), df_te["label"].values)

    collator = DataCollatorWithPadding(tokenizer=tok)

    # TrainingArguments signature varies across transformers versions.
    # Try the modern signature first, fall back to a minimal compatible set.
    try:
        args = TrainingArguments(
            output_dir=cfg.FINETUNE_OUTPUT_DIR,
            evaluation_strategy="no",
            save_strategy="no",
            learning_rate=cfg.FINETUNE_LR,
            per_device_train_batch_size=cfg.TRANSFORMER_BATCH_SIZE,
            per_device_eval_batch_size=cfg.TRANSFORMER_BATCH_SIZE,
            num_train_epochs=cfg.FINETUNE_EPOCHS,
            weight_decay=cfg.FINETUNE_WEIGHT_DECAY,
            warmup_ratio=cfg.FINETUNE_WARMUP_RATIO,
            gradient_accumulation_steps=cfg.FINETUNE_GRAD_ACCUM_STEPS,
            logging_steps=50,
            report_to=[],
        )
    except TypeError:
        log_progress(
            "TrainingArguments signature mismatch: falling back to minimal args for compatibility"
        )
        args = TrainingArguments(
            output_dir=cfg.FINETUNE_OUTPUT_DIR,
            learning_rate=cfg.FINETUNE_LR,
            per_device_train_batch_size=cfg.TRANSFORMER_BATCH_SIZE,
            num_train_epochs=cfg.FINETUNE_EPOCHS,
            weight_decay=cfg.FINETUNE_WEIGHT_DECAY,
            logging_steps=50,
        )

    trainer = Trainer(
        model=model.to(device),
        args=args,
        train_dataset=tr_ds,
        data_collator=collator,
    )

    trainer.train()

    pred = trainer.predict(te_ds)
    logits = pred.predictions
    prob_h = torch.softmax(torch.tensor(logits), dim=1)[:, 1].cpu().numpy()

    df_te = df_te.copy()
    df_te["prob_up"] = prob_h
    daily_prob = df_te.groupby("date")["prob_up"].mean().reset_index()

    out = test[["date"]].merge(daily_prob, on="date", how="left").fillna(0.5)
    prob_daily = out["prob_up"].values.astype(float)

    log_progress(
        f"Fine-tuned FinBERT prob std: {prob_daily.std():.6f} | min/max: {prob_daily.min():.6f}/{prob_daily.max():.6f}"
    )
    log_progress("Fine-tuned Transformer model complete")
    return prob_daily


# Plots
def ensure_dirs(cfg: Config) -> None:
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    os.makedirs(cfg.FIG_DIR, exist_ok=True)
    os.makedirs(cfg.FINETUNE_OUTPUT_DIR, exist_ok=True)


def plot_metric_bars(summary: pd.DataFrame, cfg: Config) -> None:
    df = summary.sort_values("auc", ascending=False).copy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(df["model"], df["auc"])
    plt.xticks(rotation=25, ha="right")
    plt.title("AUC by model")
    plt.grid(axis="y", alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.bar(df["model"], df["balanced_accuracy"])
    plt.xticks(rotation=25, ha="right")
    plt.title("Balanced Accuracy by model")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(cfg.FIG_DIR, "metrics_bar.png")
    plt.savefig(out, dpi=200)
    plt.close()
    log_progress(f"Saved plot: {out}")


def rolling_auc(y: np.ndarray, prob: np.ndarray, window: int) -> np.ndarray:
    vals = np.full_like(prob, fill_value=np.nan, dtype=float)
    for i in range(window - 1, len(y)):
        y_w = y[i - window + 1 : i + 1]
        p_w = prob[i - window + 1 : i + 1]
        if len(np.unique(y_w)) < 2:
            continue
        vals[i] = roc_auc_score(y_w, p_w)
    return vals


def plot_rolling_auc(
    test_dates: pd.Series, y: np.ndarray, probs: Dict[str, np.ndarray], cfg: Config
) -> None:
    plt.figure(figsize=(10, 5))
    for name, prob in probs.items():
        ra = rolling_auc(y, prob, cfg.ROLLING_WINDOW)
        plt.plot(test_dates, ra, label=name, alpha=0.9)

    plt.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.7)
    plt.title(f"Rolling AUC (window={cfg.ROLLING_WINDOW} days) on Test Period")
    plt.ylabel("AUC")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(alpha=0.3)

    out = os.path.join(cfg.FIG_DIR, "rolling_auc.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    log_progress(f"Saved plot: {out}")


def plot_pr_curves(
    probs: Dict[str, np.ndarray], y_true: np.ndarray, cfg: Config
) -> None:
    plt.figure(figsize=(8, 6))
    for name, prob in probs.items():
        try:
            prec, rec, _ = precision_recall_curve(y_true, prob)
            plt.plot(rec, prec, label=name)
        except Exception:
            continue
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves on Test Set")
    plt.legend()
    plt.grid(alpha=0.3)
    out = os.path.join(cfg.FIG_DIR, "pr_curves.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    log_progress(f"Saved plot: {out}")


def plot_f1_vs_threshold(
    probs: Dict[str, np.ndarray], y_true: np.ndarray, cfg: Config
) -> None:
    plt.figure(figsize=(8, 6))
    thresholds = np.linspace(0.01, 0.99, 99)
    for name, prob in probs.items():
        try:
            f1s = []
            for t in thresholds:
                pred = (prob >= t).astype(int)
                f1s.append(f1_score(y_true, pred, zero_division=0))
            plt.plot(thresholds, f1s, label=name)
        except Exception:
            continue
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Classification Threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    out = os.path.join(cfg.FIG_DIR, "f1_vs_threshold.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    log_progress(f"Saved plot: {out}")


def plot_prob_distributions(probs: Dict[str, np.ndarray], cfg: Config) -> None:
    plt.figure(figsize=(10, 5))
    for name, prob in probs.items():
        plt.hist(prob, bins=30, alpha=0.5, label=name)
    plt.title("Distribution of Predicted Probabilities on Test Set")
    plt.xlabel("Predicted P(up)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.3)

    out = os.path.join(cfg.FIG_DIR, "prob_distributions.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    log_progress(f"Saved plot: {out}")


def backtest_long_cash(
    test: pd.DataFrame, prob: np.ndarray, threshold: float, tc_bps: float
) -> pd.DataFrame:
    df = test[["date", "ret1_fwd"]].copy().reset_index(drop=True)
    pos = (prob >= threshold).astype(int)
    df["pos"] = pos

    tc = tc_bps / 10_000.0
    df["pos_prev"] = df["pos"].shift(1).fillna(0).astype(int)
    df["turnover"] = (df["pos"] - df["pos_prev"]).abs()
    df["tc"] = df["turnover"] * tc

    df["strategy_ret"] = df["pos"] * df["ret1_fwd"] - df["tc"]
    df["bh_ret"] = df["ret1_fwd"]

    df["cum_strategy"] = np.exp(df["strategy_ret"].cumsum())
    df["cum_bh"] = np.exp(df["bh_ret"].cumsum())
    return df


def compute_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized Sharpe ratio from a series of (log) returns.

    Returns are expected to be daily log returns (strategy_ret). If standard deviation
    is zero or series is empty, returns 0.0.
    """
    if returns is None or len(returns) == 0:
        return 0.0
    arr = np.asarray(returns)
    # drop nan
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    if std == 0.0:
        return 0.0
    return float((mean / std) * np.sqrt(periods_per_year))


def plot_backtest(bt: pd.DataFrame, model_name: str, cfg: Config) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(bt["date"], bt["cum_strategy"], label=f"Strategy ({model_name})")
    plt.plot(bt["date"], bt["cum_bh"], label="Buy & Hold")
    plt.title("Cumulative Growth: Long/Cash Strategy vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Growth multiple")
    plt.legend()
    plt.grid(alpha=0.3)

    out = os.path.join(cfg.FIG_DIR, "backtest_curve.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    log_progress(f"Saved plot: {out}")


def plot_roc_curves(
    probs: Dict[str, np.ndarray], y_true: np.ndarray, cfg: Config
) -> None:
    plt.figure(figsize=(8, 6))
    for name, prob in probs.items():
        try:
            fpr, tpr, _ = roc_curve(y_true, prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, prob):.3f})")
        except Exception:
            continue
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves on Test Set")
    plt.legend()
    plt.grid(alpha=0.3)
    out = os.path.join(cfg.FIG_DIR, "roc_curves.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    log_progress(f"Saved plot: {out}")


def plot_confusion_matrices(
    test: pd.DataFrame, probs: Dict[str, np.ndarray], threshold: float, cfg: Config
) -> None:
    names = list(probs.keys())
    n = len(names)
    cols = 3
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(4 * cols, 3 * rows))
    for i, name in enumerate(names, start=1):
        prob = probs[name]
        pred = (prob >= threshold).astype(int)
        cm = confusion_matrix(test["up_fwd"].values, pred)
        plt.subplot(rows, cols, i)
        plt.imshow(cm, cmap="Blues", interpolation="nearest")
        plt.title(name)
        plt.colorbar()
        ticks = [0, 1]
        plt.xticks(ticks, ["down", "up"])
        plt.yticks(ticks, ["down", "up"])
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                plt.text(c, r, int(cm[r, c]), ha="center", va="center", color="black")
        plt.xlabel("Predicted")
        plt.ylabel("True")
    out = os.path.join(cfg.FIG_DIR, "confusion_matrices.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    log_progress(f"Saved plot: {out}")


def plot_calibration(
    probs: Dict[str, np.ndarray], y_true: np.ndarray, cfg: Config
) -> None:
    plt.figure(figsize=(8, 6))
    for name, prob in probs.items():
        try:
            frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10)
            plt.plot(mean_pred, frac_pos, marker="o", label=name)
        except Exception:
            continue
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration plots")
    plt.legend()
    plt.grid(alpha=0.3)
    out = os.path.join(cfg.FIG_DIR, "calibration.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    log_progress(f"Saved plot: {out}")


def plot_sentiment_vs_returns(
    raw_df: pd.DataFrame, panel: pd.DataFrame, cfg: Config
) -> None:
    # compute daily transformer sentiment for the panel dates
    try:
        dates = pd.concat([panel["date"]]).drop_duplicates()
        daily_sent = compute_daily_transformer_sentiment(raw_df, dates, cfg)
    except Exception as e:
        log_progress(f"Could not compute daily transformer sentiment for plot: {e}")
        return
    merged = panel.merge(daily_sent, on="date", how="left").fillna(0.0)
    merged = merged.sort_values("date")
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(
        merged["date"],
        merged["sent_score"],
        color="C0",
        label="Daily Sentiment (FinBERT)",
    )
    ax1.set_ylabel("Sentiment (avg)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        merged["date"],
        merged["ret1_fwd"],
        color="C1",
        alpha=0.6,
        label="Next-day log return",
    )
    ax2.set_ylabel("Next-day log return")
    ax2.legend(loc="upper right")

    plt.title("Daily FinBERT Sentiment vs Next-day Return")
    out = os.path.join(cfg.FIG_DIR, "sentiment_vs_returns.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    log_progress(f"Saved plot: {out}")


# Main
def main():
    ensure_dirs(CFG)

    log_progress("=" * 72)
    log_progress("Starting experiment")
    log_progress("=" * 72)

    df_raw = load_data(CFG.DATA_PATH)
    daily = make_daily_docs(df_raw)
    panel = add_targets(daily, df_raw)
    train, test = split_time(panel, CFG.TRAIN_END, CFG.TEST_START)

    thr = float(train["up_fwd"].mean()) if CFG.USE_BASE_RATE_THRESHOLD else 0.5
    log_progress(
        f"Classification threshold: {thr:.3f}"
        + (" (train up-rate)" if CFG.USE_BASE_RATE_THRESHOLD else "")
    )

    results: List[Dict] = []
    probs_for_plots: Dict[str, np.ndarray] = {}

    # Transformer (OOTB)
    if CFG.USE_TRANSFORMER:
        prob_tfm = model_transformer(train, test, df_raw, CFG)
        # backtest + sharpe
        bt_tfm = backtest_long_cash(
            test, prob_tfm, threshold=thr, tc_bps=CFG.TRANSACTION_COST_BPS
        )
        sharpe_tfm = (
            compute_sharpe(bt_tfm["strategy_ret"]) if bt_tfm is not None else 0.0
        )
        results.append(
            {
                "model": "Transformer (FinBERT)",
                **evaluate_probs(test["up_fwd"].values, prob_tfm, threshold=thr),
                "sharpe": float(sharpe_tfm),
            }
        )
        probs_for_plots["Transformer"] = prob_tfm

    # Fine-tuned Transformer (separate benchmark)
    if CFG.USE_FINETUNED_TRANSFORMER:
        prob_ft = model_transformer_finetuned(train, test, df_raw, CFG)
        bt_ft = backtest_long_cash(
            test, prob_ft, threshold=thr, tc_bps=CFG.TRANSACTION_COST_BPS
        )
        sharpe_ft = compute_sharpe(bt_ft["strategy_ret"]) if bt_ft is not None else 0.0
        results.append(
            {
                "model": "Transformer (FinBERT) fine-tuned",
                **evaluate_probs(test["up_fwd"].values, prob_ft, threshold=thr),
                "sharpe": float(sharpe_ft),
            }
        )
        probs_for_plots["FinBERT-FT"] = prob_ft

    # Naive
    prob_naive, metrics_naive = naive_up(test)
    bt_naive = backtest_long_cash(
        test, prob_naive, threshold=thr, tc_bps=CFG.TRANSACTION_COST_BPS
    )
    sharpe_naive = (
        compute_sharpe(bt_naive["strategy_ret"]) if bt_naive is not None else 0.0
    )
    metrics_naive["sharpe"] = float(sharpe_naive)
    results.append({"model": "Naive", **metrics_naive})
    probs_for_plots["Naive"] = prob_naive

    # TF–IDF
    prob_tfidf = model_tfidf(train, test, CFG)
    bt_tfidf = backtest_long_cash(
        test, prob_tfidf, threshold=thr, tc_bps=CFG.TRANSACTION_COST_BPS
    )
    sharpe_tfidf = (
        compute_sharpe(bt_tfidf["strategy_ret"]) if bt_tfidf is not None else 0.0
    )
    results.append(
        {
            "model": "TFIDF",
            **evaluate_probs(test["up_fwd"].values, prob_tfidf, threshold=thr),
            "sharpe": float(sharpe_tfidf),
        }
    )
    probs_for_plots["TFIDF"] = prob_tfidf

    # Dictionary
    prob_lex = model_dictionary(train, test, CFG)
    bt_lex = backtest_long_cash(
        test, prob_lex, threshold=thr, tc_bps=CFG.TRANSACTION_COST_BPS
    )
    sharpe_lex = compute_sharpe(bt_lex["strategy_ret"]) if bt_lex is not None else 0.0
    results.append(
        {
            "model": "Dictionary",
            **evaluate_probs(test["up_fwd"].values, prob_lex, threshold=thr),
            "sharpe": float(sharpe_lex),
        }
    )
    probs_for_plots["Dictionary"] = prob_lex

    summary = pd.DataFrame(results)
    summary.to_csv(CFG.SUMMARY_CSV, index=False)

    # Optional: walk-forward evaluation and runtime benchmarking
    if CFG.DO_WALK_FORWARD:
        log_progress("Running walk-forward evaluation...")
        wf = walk_forward_evaluate(panel, df_raw, CFG, n_splits=CFG.WALK_N_SPLITS)
        out_wf = os.path.join(CFG.RESULTS_DIR, "walk_forward_summary.csv")
        wf.to_csv(out_wf, index=False)
        log_progress(f"Saved walk-forward summary: {out_wf}")

    if CFG.RUN_BENCHMARK:
        log_progress("Running runtime benchmark (this will run each model once)...")
        bench = benchmark_runtime(train, test, df_raw, CFG)
        out_b = os.path.join(CFG.RESULTS_DIR, "runtime_benchmark.csv")
        bench.to_csv(out_b, index=False)
        log_progress(f"Saved runtime benchmark: {out_b}")

    log_progress("Final Results:")
    print(summary)
    log_progress(f"Saved summary: {CFG.SUMMARY_CSV}")

    # Plots
    plot_metric_bars(summary, CFG)
    plot_rolling_auc(
        test["date"].reset_index(drop=True), test["up_fwd"].values, probs_for_plots, CFG
    )
    plot_prob_distributions(probs_for_plots, CFG)
    plot_pr_curves(probs_for_plots, test["up_fwd"].values, CFG)
    plot_f1_vs_threshold(probs_for_plots, test["up_fwd"].values, CFG)
    plot_roc_curves(probs_for_plots, test["up_fwd"].values, CFG)
    plot_calibration(probs_for_plots, test["up_fwd"].values, CFG)
    plot_confusion_matrices(test, probs_for_plots, thr, CFG)
    # Sentiment vs returns (runs transformer scoring again if needed)
    if CFG.USE_TRANSFORMER:
        plot_sentiment_vs_returns(df_raw, panel, CFG)

    # Backtest on best (non-naive) AUC model
    if CFG.INCLUDE_BACKTEST:
        non_naive = summary[summary["model"] != "Naive"].copy()
        if len(non_naive) > 0:
            best_name = non_naive.sort_values("auc", ascending=False).iloc[0]["model"]
            key_map = {
                "Transformer (FinBERT)": "Transformer",
                "Transformer (FinBERT) fine-tuned": "FinBERT-FT",
                "TFIDF": "TFIDF",
                "Dictionary": "Dictionary",
            }
            best_key = key_map.get(best_name, None)

            if best_key and best_key in probs_for_plots:
                bt = backtest_long_cash(
                    test,
                    probs_for_plots[best_key],
                    threshold=thr,
                    tc_bps=CFG.TRANSACTION_COST_BPS,
                )
                bt.to_csv(CFG.BACKTEST_CSV, index=False)
                plot_backtest(bt, best_name, CFG)
                log_progress(f"Saved backtest data: {CFG.BACKTEST_CSV}")

    log_progress("Experiment complete!")
    log_progress("=" * 72)


if __name__ == "__main__":
    main()
