import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
import torch
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Config 
# -----------------------------
DATA_PATH = "mentalmanip_con.csv"
SEQ2SEQ_MODEL = "facebook/bart-base" 
EMBED_MODEL = "all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)
# Ensure no NaNs slip into the tokenizer causing empty tensors
df["Dialogue"] = df["Dialogue"].fillna("").astype(str)
df["Manipulative"] = df["Manipulative"].astype(int)

# Parse multi-label techniques
def parse_techniques(tech_str):
    if pd.isna(tech_str) or tech_str == "":
        return []
    if isinstance(tech_str, str):
        return [t.strip() for t in tech_str.split(',')]
    return []

df["Techniques"] = df["Technique"].apply(parse_techniques)
# Extract first technique for single-label classification
df["Technique_Primary"] = df["Techniques"].apply(lambda x: x[0] if len(x) > 0 else None)

# Get all unique techniques
unique_techniques = sorted({t for techs in df["Techniques"] for t in techs if t})
technique_to_id = {t: i for i, t in enumerate(unique_techniques)}
df["Technique_ID"] = df["Technique_Primary"].map(lambda x: technique_to_id.get(x, -1))

# -----------------------------
# Mirror generation (FIXED)
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(SEQ2SEQ_MODEL)

# FIX 1: Add attn_implementation="eager" to bypass the SDPA CUDA bug
model = AutoModelForSeq2SeqLM.from_pretrained(
    SEQ2SEQ_MODEL, 
    attn_implementation="eager" 
).to(DEVICE)

def generate_mirror(texts, batch_size=16):
    results = []
    prefix = "Reframing this to be healthy: " 
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        prompts = [prefix + t for t in batch]
        
        # FIX 2: Explicitly set max_length in tokenizer to prevent index out of bounds
        enc = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=256
        ).to(DEVICE)
        
        with torch.no_grad():
            out = model.generate(**enc, max_length=64, num_beams=4, early_stopping=True)
            
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        results.extend(decoded)
        torch.cuda.empty_cache()
    return results

print("Generating Mirrors with BART...")
df["Mirror"] = generate_mirror(df["Dialogue"].tolist(), batch_size=16)

# -----------------------------
# Embeddings
# -----------------------------
embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
orig_embs = embedder.encode(df["Dialogue"].tolist(), convert_to_numpy=True)
mir_embs = embedder.encode(df["Mirror"].tolist(), convert_to_numpy=True)

# -----------------------------
# Contrastive features
# -----------------------------
def contrastive_features(orig_embs, mir_embs, orig_texts, mir_texts):
    diff = orig_embs - mir_embs
    # Magnitude of change (Semantic Shift)
    shift_mag = np.linalg.norm(diff, axis=1)
    
    # Cosine Similarity
    cosine = np.sum(orig_embs * mir_embs, axis=1) / (
        np.linalg.norm(orig_embs, axis=1) * np.linalg.norm(mir_embs, axis=1) + 1e-8
    )
    
    # String Edit Distance (Syntactic Change)
    edit = np.array([1 - SequenceMatcher(None, a, b).ratio() for a, b in zip(orig_texts, mir_texts)])
    
    X_full = np.hstack([diff, cosine.reshape(-1,1), edit.reshape(-1,1)])
    X_viz = pd.DataFrame({
        'Semantic Shift': shift_mag,
        'Semantic Similarity': cosine,
        'Syntactic Change': edit
    })
    return X_full, X_viz

X, X_viz = contrastive_features(orig_embs, mir_embs, df["Dialogue"].tolist(), df["Mirror"].tolist())
y = df["Manipulative"].values

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Binary Classifier
# -----------------------------
clf_bin = LogisticRegression(max_iter=200, solver="liblinear", class_weight="balanced")
clf_bin.fit(X_train, y_train)

# Technique Classifier (XGBoost on manipulative samples only)
# Filter training data: only manipulative samples
mask_train = (y_train == 1)
X_multi_train = X_train[mask_train]
technique_ids_train = df.iloc[X_train.index[mask_train]]["Technique_ID"].values

# Remove invalid labels (-1)
valid_mask = (technique_ids_train != -1)
X_multi_train = X_multi_train[valid_mask]
y_multi_train = technique_ids_train[valid_mask]

# Oversample to balance techniques
if len(np.unique(y_multi_train)) > 1:
    ros = RandomOverSampler(random_state=42)
    X_multi_bal, y_multi_bal = ros.fit_resample(X_multi_train, y_multi_train)
    
    clf_multi = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=len(unique_techniques),
        random_state=42,
        verbosity=0
    )
    clf_multi.fit(X_multi_bal, y_multi_bal)
    technique_model_trained = True
else:
    technique_model_trained = False
    print("Warning: Only one or zero techniques in training set, skipping technique classifier.")

# -----------------------------
# Visualization Logic 
# -----------------------------
clf_viz = LogisticRegression(class_weight="balanced")
clf_viz.fit(X_viz, y)

sns.set_theme(style="whitegrid")

# Plot A: Decision Boundary
plt.figure(figsize=(10, 6))
sns.scatterplot(data=X_viz, x='Semantic Shift', y='Syntactic Change', hue=y, palette='coolwarm', alpha=0.6)

b0 = clf_viz.intercept_[0]
b1, b2, b3 = clf_viz.coef_[0]
mean_sim = X_viz['Semantic Similarity'].mean()
x_range = np.linspace(X_viz['Semantic Shift'].min(), X_viz['Semantic Shift'].max(), 100)
y_range = -(b0 + b1 * x_range + b2 * mean_sim) / b3 

plt.plot(x_range, y_range, '--k', linewidth=2, label='Decision Boundary (at mean similarity)')
plt.title("Plot A: BART Decision Boundary")
plt.legend(title="Class", labels=["Manipulative", "Healthy", "Boundary"])
plt.tight_layout()
plt.savefig("plot_a_decision_boundary.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot B: Feature Importance 
plt.figure(figsize=(8, 5))
feature_importance = pd.Series(clf_viz.coef_[0], index=X_viz.columns)
feature_importance.sort_values().plot(kind='barh', color='salmon')
plt.axvline(0, color='black', lw=0.8)
plt.title("Plot B: BART Feature Importance (LR Coefficients)")
plt.xlabel("Impact on Manipulation Probability")
plt.tight_layout()
plt.show()

# Plot C: Distribution Plots (KDE)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
features = X_viz.columns

for i, col in enumerate(features):
    sns.kdeplot(data=X_viz, x=col, hue=y, fill=True, common_norm=False, palette='coolwarm', ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Density")

plt.suptitle("Plot C: Feature Distributions by Class", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# -----------------------------
# Final Stats & Confusion Matrix
# -----------------------------
y_pred = clf_bin.predict(X_test)
print("\n=== BART Performance Report ===")
print(classification_report(y_test, y_pred, target_names=["Non-Manipulative","Manipulative"]))

cm = confusion_matrix(y_test, y_pred, labels=[0,1])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Non-Manipulative","Manipulative"],
            yticklabels=["Non-Manipulative","Manipulative"])
plt.title("BART Confusion Matrix (Binary)")
plt.show()

# Technique Classification Evaluation
# =====================================
if technique_model_trained:
    print("\n=== Technique Classification Results ===")
    
    # Evaluate on test set (only manipulative samples)
    mask_test = (y_test == 1)
    X_multi_test = X_test[mask_test]
    technique_ids_test = df.iloc[X_test.index[mask_test]]["Technique_ID"].values
    
    valid_mask_test = (technique_ids_test != -1)
    X_multi_test = X_multi_test[valid_mask_test]
    y_multi_test = technique_ids_test[valid_mask_test]
    
    if len(y_multi_test) > 0:
        y_multi_pred = clf_multi.predict(X_multi_test)
        print(f"\nTechnique Classification Report ({len(y_multi_test)} samples):")
        print(classification_report(y_multi_test, y_multi_pred, target_names=unique_techniques))
        
        # Technique confusion matrix
        cm_multi = confusion_matrix(y_multi_test, y_multi_pred, labels=range(len(unique_techniques)))
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_multi, annot=True, fmt="d", cmap="Blues",
                    xticklabels=unique_techniques, yticklabels=unique_techniques, cbar_kws={'label': 'Count'})
        plt.title("BART Technique Classification Confusion Matrix")
        plt.xlabel("Predicted Technique")
        plt.ylabel("True Technique")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        # Per-technique accuracy
        print("\nPer-Technique Accuracy:")
        for i, tech in enumerate(unique_techniques):
            mask = (y_multi_test == i)
            if mask.sum() > 0:
                acc = (y_multi_pred[mask] == i).mean()
                count = mask.sum()
                print(f"  {tech}: {acc:.3f} ({count} samples)")
    else:
        print("No manipulative samples in test set for technique evaluation.")