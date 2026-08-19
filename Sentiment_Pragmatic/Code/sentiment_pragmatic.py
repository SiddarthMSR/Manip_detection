import csv
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# -----------------------------
# 1. Load Consensus Dataset
# -----------------------------
def load_dataset(path):
    dialogues, labels_bin, labels_multi = [], [], []
    with open(path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            dialogues.append(row['Dialogue'])
            labels_bin.append(int(row['Manipulative']))  # 0/1
            if row['Technique']:
                labels_multi.append(row['Technique'].split(','))
            else:
                labels_multi.append([])
    return dialogues, labels_bin, labels_multi

dataset_path = Path(__file__).resolve().parent.parent.parent / "Dataset" / "mentalmanip_con.csv"
dialogues, y_bin, y_multi = load_dataset(dataset_path)

# -----------------------------
# 2. Load Encoders
# -----------------------------
tok_text = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model_text = AutoModel.from_pretrained("distilbert-base-uncased")

tok_sent = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model_sent = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# -----------------------------
# 3. Feature Extraction
# -----------------------------
def extract_features(utterance):
    # Semantic embedding
    inputs = tok_text(utterance, return_tensors="pt", truncation=True, max_length=128)
    emb_text = model_text(**inputs).last_hidden_state.mean(dim=1).detach().numpy()

    # Sentiment embedding
    inputs_s = tok_sent(utterance, return_tensors="pt", truncation=True, max_length=128)
    emb_sent = model_sent(**inputs_s).last_hidden_state.mean(dim=1).detach().numpy()

    # Pragmatic cues
    prag_feats = np.array([
        utterance.lower().count("you"),
        utterance.lower().count("must"),
        utterance.count("?"),
        utterance.count("!")
    ]).reshape(1,-1)

    return np.concatenate([emb_text, emb_sent, prag_feats], axis=1)

# -----------------------------
# 4. Build Feature Matrix
# -----------------------------
X = [extract_features(u) for u in dialogues]
X = np.vstack(X)

unique_techniques = sorted({t for sublist in y_multi for t in sublist})
technique_to_id = {t: i for i, t in enumerate(unique_techniques)}

y_multi_single = []
for techniques in y_multi:
    if techniques:
        y_multi_single.append(technique_to_id[techniques[0]])
    else:
        y_multi_single.append(-1)

# -----------------------------
# 5. Train/Test Split + Scaling + Dimensionality Reduction
# -----------------------------
X_train, X_test, y_bin_train, y_bin_test, y_multi_train, y_multi_test = train_test_split(
    X, y_bin, y_multi_single, test_size=0.2, random_state=42, stratify=y_bin
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svd = TruncatedSVD(n_components=200, random_state=42)
X_train = svd.fit_transform(X_train)
X_test = svd.transform(X_test)

# -----------------------------
# 6. Train Separate Models
# -----------------------------
# Binary detector
clf_bin = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs", class_weight="balanced")
clf_bin.fit(X_train, y_bin_train)

# Technique classifier (XGBoost + oversampling)
mask_train = (np.array(y_bin_train) == 1)
X_multi = X_train[mask_train]
y_multi_only = np.array(y_multi_train)[mask_train]

# Remove -1 labels
valid_mask = (y_multi_only != -1)
X_multi = X_multi[valid_mask]
y_multi_only = y_multi_only[valid_mask]

ros = RandomOverSampler(random_state=42)
X_bal, y_bal = ros.fit_resample(X_multi, y_multi_only)

clf_multi = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=len(unique_techniques),
    random_state=42
)
clf_multi.fit(X_bal, y_bal)

# -----------------------------
# 7. Evaluation
# -----------------------------
y_pred_bin = clf_bin.predict(X_test)
print("Binary Manipulation Detection Report:")
print(classification_report(y_bin_test, y_pred_bin))

mask_test = (np.array(y_bin_test) == 1)
y_multi_eval = np.array(y_multi_test)[mask_test]
valid_mask_test = (y_multi_eval != -1)
y_multi_eval = y_multi_eval[valid_mask_test]
y_pred_multi = clf_multi.predict(X_test[mask_test][valid_mask_test])

print("Technique Classification Report:")
print(classification_report(y_multi_eval, y_pred_multi))

# Confusion Matrices
def plot_confusion(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.show()

plot_confusion(y_bin_test, y_pred_bin, labels=[0,1], title="Binary Manipulation Confusion Matrix")
plot_confusion(y_multi_eval, y_pred_multi,
               labels=list(range(len(unique_techniques))), title="Technique Confusion Matrix")

# -----------------------------
# 8. Cross-Validation (Binary)
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf_bin, svd.transform(scaler.transform(X)), y_bin, cv=cv)
print("Binary CV Accuracy:", scores.mean())

# -----------------------------
# 9. Runtime Example
# -----------------------------
test_utterance = "You never listen to me!"
features = extract_features(test_utterance)
features = scaler.transform(features)
features = svd.transform(features)

pred_bin = clf_bin.predict(features)[0]
print("\nTest Utterance:", test_utterance)
print("Manipulation Detected:", bool(pred_bin))

if pred_bin == 1:
    pred_multi = clf_multi.predict(features)[0]
    print("Technique:", unique_techniques[pred_multi])
else:
    print("Technique: None")
