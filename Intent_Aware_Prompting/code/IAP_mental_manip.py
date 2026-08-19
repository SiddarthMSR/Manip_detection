import pandas as pd
import json
import time
import os
import csv
import re
import ast
import traceback
from pathlib import Path
from dotenv import load_dotenv
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
    zero_one_loss,
)
from sklearn.preprocessing import MultiLabelBinarizer

# ==========================================
# ⚙️ SETUP
# ==========================================
load_dotenv()
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "facebook/bart-large-mnli")
HF_INTENT_MODEL_NAME = os.getenv("HF_INTENT_MODEL_NAME", "google/flan-t5-small")
INPUT_CSV = "mentalmanip_con_cleaned.csv"
OUTPUT_FILE = "results_iap_multilabel.csv"
MAX_ROWS = 1000

SET_M_TACTICS = [
    "Denial", "Evasion", "Feigning Innocence", "Rationalization", 
    "Playing the Victim Role", "Playing the Servant Role", "Shaming or Belittlement", 
    "Intimidation", "Brandishing Anger", "Accusation", "Persuasion or Seduction"
]

SAFE_LABELS = {"Safe / No Manipulation", "Safe", "No Manipulation"}
MANIPULATION_THRESHOLD = 0.52
TACTIC_THRESHOLD = 0.33
_classifier = None
_intent_generator = None


def get_classifier():
    global _classifier
    if _classifier is not None:
        return _classifier

    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except Exception:
        device = -1

    _classifier = pipeline(
        "zero-shot-classification",
        model=HF_MODEL_NAME,
        device=device,
    )
    print(f"Loaded local HF model: {HF_MODEL_NAME} (device={'cuda' if device == 0 else 'cpu'})")
    return _classifier


def get_intent_generator():
    global _intent_generator
    if _intent_generator is not None:
        return _intent_generator

    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except Exception:
        device = -1

    _intent_generator = pipeline(
        "text2text-generation",
        model=HF_INTENT_MODEL_NAME,
        device=device,
    )
    print(f"Loaded local intent model: {HF_INTENT_MODEL_NAME} (device={'cuda' if device == 0 else 'cpu'})")
    return _intent_generator


def extract_intent(text: str) -> str:
    generator = get_intent_generator()
    prompt = (
        "Summarize the speaker's underlying intent in one concise sentence. "
        "Focus on motivation and desired effect on the listener.\n"
        f"Text: {text}"
    )
    result = generator(prompt, max_new_tokens=40, do_sample=False)
    intent = result[0]["generated_text"].strip()
    return intent if intent else "Intent unclear from text."

# ==========================================
# 🧠 IAP PIPELINE LOGIC
# ==========================================
def run_iap_pipeline(text: str) -> tuple:
    max_retries = 3
    clf = get_classifier()
    for attempt in range(max_retries):
        try:
            # 1) Intent extraction (local, non-hardcoded).
            intent = extract_intent(text)

            # 1) Binary gate using local NLI model.
            gate = clf(
                f"Text: {text}\nIntent: {intent}",
                candidate_labels=["manipulative communication", "non-manipulative communication"],
                multi_label=False,
                hypothesis_template="This text is {}.",
            )
            gate_scores = dict(zip(gate["labels"], gate["scores"]))
            manip_score = gate_scores.get("manipulative communication", 0.0)
            is_manipulative = manip_score >= MANIPULATION_THRESHOLD

            if not is_manipulative:
                return ["Safe / No Manipulation"], intent

            # 2) Multi-label tactic ranking using same local NLI model.
            tactic_res = clf(
                f"Text: {text}\nIntent: {intent}",
                candidate_labels=SET_M_TACTICS,
                multi_label=True,
                hypothesis_template="This text uses {}.",
            )

            scored_tactics = list(zip(tactic_res["labels"], tactic_res["scores"]))
            preds = [label for label, score in scored_tactics if score >= TACTIC_THRESHOLD][:3]

            # If gate says manipulative but scores are diffuse, keep strongest tactic.
            if not preds and scored_tactics and scored_tactics[0][1] >= 0.25:
                preds = [scored_tactics[0][0]]

            if not preds:
                return ["Safe / No Manipulation"], intent

            return preds, intent
            
        except Exception as e:
            print(f"[run_iap_pipeline] Attempt {attempt + 1}/{max_retries} failed")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("Traceback:")
            print(traceback.format_exc())
            time.sleep(5)
    return ["Error"], "Error"


def detect_text_column(df: pd.DataFrame) -> str:
    for candidate in ["Dialogue", "dialogue", "text", "Text", "utterance", "Utterance"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"No text column found. Available columns: {list(df.columns)}")


def detect_label_column(df: pd.DataFrame) -> str:
    for candidate in ["Label", "label", "Technique", "technique", "class", "Class"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"No label column found. Available columns: {list(df.columns)}")


def parse_labels(value):
    if pd.isna(value):
        return set()

    text = str(value).strip()
    if not text:
        return set()

    labels = []
    # Handle python-list strings like "['Denial', 'Evasion']".
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                labels = [str(x).strip() for x in parsed]
            else:
                labels = [str(parsed).strip()]
        except Exception:
            labels = [s.strip() for s in text.replace("[", "").replace("]", "").replace("'", "").split(",")]
    else:
        labels = [s.strip() for s in text.split(",")]

    filtered = {label for label in labels if label and label not in SAFE_LABELS and label != "Error"}
    return filtered

# ==========================================
# 📊 MULTI-LABEL METRICS
# ==========================================
def calculate_metrics(file_path, allowed_indices=None):
    df = pd.read_csv(file_path)

    if allowed_indices is not None and "Original_Index" in df.columns:
        idx_series = pd.to_numeric(df["Original_Index"], errors="coerce")
        df = df[idx_series.isin(allowed_indices)]

    if df.empty:
        print("No rows available to evaluate.")
        return

    y_true = df['True_Label'].apply(parse_labels)
    y_pred = df['AI_Predictions'].apply(parse_labels)

    # Binary Metrics (Is any manipulation present?)
    true_bin = y_true.apply(lambda x: 1 if any(t in SET_M_TACTICS for t in x) else 0)
    pred_bin = y_pred.apply(lambda x: 1 if any(t in SET_M_TACTICS for t in x) else 0)

    print("\n" + "="*40)
    print("🛡️ BINARY PERFORMANCE (Manipulation vs Safe)")
    print(f"Accuracy: {accuracy_score(true_bin, pred_bin):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(true_bin, pred_bin):.4f}")
    print(f"Precision: {precision_score(true_bin, pred_bin, zero_division=0):.4f}")
    print(f"Recall: {recall_score(true_bin, pred_bin, zero_division=0):.4f}")
    print(f"F1: {f1_score(true_bin, pred_bin, zero_division=0):.4f}")

    tn, fp, fn, tp = confusion_matrix(true_bin, pred_bin, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    print(f"Specificity: {specificity:.4f}")
    print("Confusion Matrix [rows=true, cols=pred]:")
    print(confusion_matrix(true_bin, pred_bin, labels=[0, 1]))
    print("\nDetailed Report:")
    print(classification_report(true_bin, pred_bin, target_names=["Safe", "Manipulative"]))

    print("🎯 MULTI-LABEL PERFORMANCE (Set M tactics)")
    mlb = MultiLabelBinarizer(classes=SET_M_TACTICS)
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    subset_acc = 1.0 - zero_one_loss(y_true_bin, y_pred_bin)
    print(f"Subset Accuracy (Exact Match): {subset_acc:.4f}")
    print(f"Hamming Loss: {hamming_loss(y_true_bin, y_pred_bin):.4f}")
    print(f"Jaccard (samples): {jaccard_score(y_true_bin, y_pred_bin, average='samples', zero_division=0):.4f}")
    print(f"Jaccard (micro): {jaccard_score(y_true_bin, y_pred_bin, average='micro', zero_division=0):.4f}")
    print(f"Jaccard (macro): {jaccard_score(y_true_bin, y_pred_bin, average='macro', zero_division=0):.4f}")
    print(f"Precision (micro): {precision_score(y_true_bin, y_pred_bin, average='micro', zero_division=0):.4f}")
    print(f"Recall (micro): {recall_score(y_true_bin, y_pred_bin, average='micro', zero_division=0):.4f}")
    print(f"F1 (micro): {f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0):.4f}")
    print(f"Precision (macro): {precision_score(y_true_bin, y_pred_bin, average='macro', zero_division=0):.4f}")
    print(f"Recall (macro): {recall_score(y_true_bin, y_pred_bin, average='macro', zero_division=0):.4f}")
    print(f"F1 (macro): {f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0):.4f}")
    print("\nPer-label report:")
    print(classification_report(y_true_bin, y_pred_bin, target_names=SET_M_TACTICS, zero_division=0))
    print("="*40 + "\n")

# ==========================================
# 🚀 EXECUTION ENGINE
# ==========================================
def main():
    df = pd.read_csv(INPUT_CSV)
    df = df.head(MAX_ROWS).copy()
    text_col = detect_text_column(df)
    label_col = detect_label_column(df)
    subset_indices = set(df.index.tolist())

    processed_indices = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            output_df = pd.read_csv(OUTPUT_FILE)
            if "Original_Index" in output_df.columns:
                parsed_idx = pd.to_numeric(output_df["Original_Index"], errors="coerce").dropna().astype(int)
                processed_indices = set(parsed_idx[parsed_idx.isin(subset_indices)].tolist())
        except pd.errors.EmptyDataError:
            pass
    
    remaining = df[~df.index.isin(processed_indices)]
    print(f"Running IAP on first {MAX_ROWS} rows: {len(remaining)} rows left.")

    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not processed_indices:
            writer.writerow(['Original_Index', 'Text', 'True_Label', 'AI_Predictions', 'Extracted_Intent'])

        for idx, row in remaining.iterrows():
            preds, intent = run_iap_pipeline(str(row[text_col]))
            writer.writerow([idx, row[text_col], row[label_col], preds, intent])
            f.flush()
            print(f"Row {idx} | Actual: {row[label_col]} | Preds: {preds}")
            time.sleep(1.5)

    calculate_metrics(OUTPUT_FILE, allowed_indices=subset_indices)

if __name__ == "__main__":
    main()