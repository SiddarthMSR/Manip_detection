import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv("mentalmanip_con.csv")
df["Dialogue"] = df["Dialogue"].fillna("").astype(str)
df["Manipulative"] = df["Manipulative"].astype(int)

# Parse techniques
def parse_techniques(tech_str):
    if pd.isna(tech_str) or tech_str == "":
        return []
    if isinstance(tech_str, str):
        return [t.strip() for t in tech_str.split(',')]
    return []

df["Techniques"] = df["Technique"].apply(parse_techniques)
df["Technique_Primary"] = df["Techniques"].apply(lambda x: x[0] if len(x) > 0 else None)

unique_techniques = sorted({t for techs in df["Techniques"] for t in techs if t})
technique_to_id = {t: i for i, t in enumerate(unique_techniques)}
df["Technique_ID"] = df["Technique_Primary"].map(lambda x: technique_to_id.get(x, -1))

print(f"Dataset: {len(df)} samples, {len(unique_techniques)} techniques")

# Create features
print("Extracting features...")
X = np.column_stack([
    df["Dialogue"].apply(len),
    df["Dialogue"].apply(lambda x: x.lower().count("you")),
    df["Dialogue"].apply(lambda x: x.lower().count("must")),
    df["Dialogue"].apply(lambda x: x.count("!")),
    df["Dialogue"].apply(lambda x: x.count("?")),
    df["Dialogue"].apply(lambda x: len(x.split())),
])
y = df["Manipulative"].values
technique_ids = df["Technique_ID"].values

# Train/test split
X_train, X_test, y_train, y_test, tech_train, tech_test = train_test_split(
    X, y, technique_ids, test_size=0.2, random_state=42, stratify=y
)

# Binary classifier
print("Training binary classifier...")
clf_bin = LogisticRegression(max_iter=300, solver="liblinear", class_weight="balanced")
clf_bin.fit(X_train, y_train)
y_pred = clf_bin.predict(X_test)

# Technique classifier
print("Training technique classifier...")
mask_train = (y_train == 1)
X_multi_train = X_train[mask_train]
tech_multi_train = tech_train[mask_train]

valid_mask = (tech_multi_train != -1)
X_multi_train = X_multi_train[valid_mask]
y_multi_train = tech_multi_train[valid_mask]

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

# Test evaluation
mask_test = (y_test == 1)
X_multi_test = X_test[mask_test]
tech_multi_test = tech_test[mask_test]

valid_mask_test = (tech_multi_test != -1)
X_multi_test = X_multi_test[valid_mask_test]
y_multi_test = tech_multi_test[valid_mask_test]

y_multi_pred = clf_multi.predict(X_multi_test)

# Compute metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_multi_test, y_multi_pred, labels=range(len(unique_techniques)), zero_division=0
)
cm = confusion_matrix(y_multi_test, y_multi_pred, labels=range(len(unique_techniques)))

# Create visualizations
sns.set_theme(style="whitegrid")

# 1. F1 Score per Technique
fig, ax = plt.subplots(figsize=(14, 6))
colors = ['#2ecc71' if f > 0.2 else '#e74c3c' for f in f1]
bars = ax.barh(unique_techniques, f1, color=colors, edgecolor='black', linewidth=1.5)
ax.set_xlabel('F1 Score', fontsize=12, weight='bold')
ax.set_title('Technique Classification: F1 Score by Technique', fontsize=14, weight='bold')
ax.set_xlim(0, max(f1) * 1.15)

# Add value labels
for i, (bar, score) in enumerate(zip(bars, f1)):
    ax.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=10, weight='bold')

plt.tight_layout()
plt.savefig('slide_assets/technique_f1_scores.png', dpi=220, bbox_inches='tight', facecolor='white')
print("✓ Saved: slide_assets/technique_f1_scores.png")
plt.close()

# 2. Precision, Recall, F1 Comparison
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(unique_techniques))
width = 0.25

bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db', edgecolor='black', linewidth=1)
bars2 = ax.bar(x, recall, width, label='Recall', color='#e74c3c', edgecolor='black', linewidth=1)
bars3 = ax.bar(x + width, f1, width, label='F1 Score', color='#2ecc71', edgecolor='black', linewidth=1)

ax.set_ylabel('Score', fontsize=12, weight='bold')
ax.set_title('Technique Classification: Precision, Recall, and F1 Score by Technique', fontsize=14, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(unique_techniques, rotation=45, ha='right')
ax.legend(fontsize=11, loc='upper right')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('slide_assets/technique_metrics_comparison.png', dpi=220, bbox_inches='tight', facecolor='white')
print("✓ Saved: slide_assets/technique_metrics_comparison.png")
plt.close()

# 3. Confusion Matrix Heatmap
fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(cm, cmap='Blues', aspect='auto')

ax.set_xticks(np.arange(len(unique_techniques)))
ax.set_yticks(np.arange(len(unique_techniques)))
ax.set_xticklabels(unique_techniques, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(unique_techniques, fontsize=10)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Count', fontsize=11, weight='bold')

# Add text annotations
for i in range(len(unique_techniques)):
    for j in range(len(unique_techniques)):
        text = ax.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black",
                      fontsize=9, weight='bold')

ax.set_ylabel('True Technique', fontsize=12, weight='bold')
ax.set_xlabel('Predicted Technique', fontsize=12, weight='bold')
ax.set_title('Technique Classification Confusion Matrix', fontsize=14, weight='bold')

plt.tight_layout()
plt.savefig('slide_assets/technique_confusion_matrix.png', dpi=220, bbox_inches='tight', facecolor='white')
print("✓ Saved: slide_assets/technique_confusion_matrix.png")
plt.close()

# 4. Support Distribution
fig, ax = plt.subplots(figsize=(14, 6))
colors_support = plt.cm.viridis(np.linspace(0, 1, len(unique_techniques)))
bars = ax.bar(unique_techniques, support, color=colors_support, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Number of Samples', fontsize=12, weight='bold')
ax.set_title('Technique Classification: Sample Distribution in Test Set', fontsize=14, weight='bold')
ax.set_xticklabels(unique_techniques, rotation=45, ha='right')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{int(height)}', ha='center', va='bottom', fontsize=10, weight='bold')

plt.tight_layout()
plt.savefig('slide_assets/technique_support_distribution.png', dpi=220, bbox_inches='tight', facecolor='white')
print("✓ Saved: slide_assets/technique_support_distribution.png")
plt.close()

# 5. Summary Metrics Table
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

# Calculate overall metrics
y_pred_bin = clf_bin.predict(X_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

summary_data = [
    ['Metric', 'Binary Classification', 'Technique Classification'],
    ['Accuracy', f"{accuracy_score(y_test, y_pred_bin):.3f}", f"{(y_multi_pred == y_multi_test).mean():.3f}"],
    ['Macro Precision', f"{precision_score(y_test, y_pred_bin, average='macro'):.3f}", f"{precision.mean():.3f}"],
    ['Macro Recall', f"{recall_score(y_test, y_pred_bin, average='macro'):.3f}", f"{recall.mean():.3f}"],
    ['Macro F1', f"{f1_score(y_test, y_pred_bin, average='macro'):.3f}", f"{f1.mean():.3f}"],
    ['Weighted F1', f"{f1_score(y_test, y_pred_bin, average='weighted'):.3f}", f"{(f1 * support).sum() / support.sum():.3f}"],
]

table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                colWidths=[0.35, 0.32, 0.32])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor('#264653')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_data)):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        else:
            table[(i, j)].set_facecolor('#ffffff')

plt.title('Overall Classification Performance', fontsize=14, weight='bold', pad=20)
plt.savefig('slide_assets/technique_summary_metrics.png', dpi=220, bbox_inches='tight', facecolor='white')
print("✓ Saved: slide_assets/technique_summary_metrics.png")
plt.close()

# 6. Per-Technique Accuracy with Support
fig, ax = plt.subplots(figsize=(14, 6))

# Calculate per-technique accuracy
per_tech_acc = []
for i in range(len(unique_techniques)):
    mask = (y_multi_test == i)
    if mask.sum() > 0:
        acc = (y_multi_pred[mask] == i).mean()
        per_tech_acc.append(acc)
    else:
        per_tech_acc.append(0)

per_tech_acc = np.array(per_tech_acc)

# Create scatter plot with size = support
ax.scatter(support, per_tech_acc, s=support*20, alpha=0.6, c=per_tech_acc, 
          cmap='RdYlGn', edgecolors='black', linewidth=1.5, vmin=0, vmax=1)

# Add labels for each point
for i, (sup, acc) in enumerate(zip(support, per_tech_acc)):
    ax.annotate(unique_techniques[i], (sup, acc), fontsize=9, 
               xytext=(5, 5), textcoords='offset points', weight='bold')

ax.set_xlabel('Sample Support (Test Set)', fontsize=12, weight='bold')
ax.set_ylabel('Accuracy', fontsize=12, weight='bold')
ax.set_title('Technique Classification Accuracy vs. Sample Support\n(bubble size = support)', 
            fontsize=14, weight='bold')
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('Accuracy', fontsize=11, weight='bold')

plt.tight_layout()
plt.savefig('slide_assets/technique_accuracy_vs_support.png', dpi=220, bbox_inches='tight', facecolor='white')
print("✓ Saved: slide_assets/technique_accuracy_vs_support.png")
plt.close()

# Print summary
print("\n" + "="*70)
print("TECHNIQUE CLASSIFICATION SUMMARY")
print("="*70)
print(f"\nTest Set Size: {len(y_multi_test)} samples")
print(f"Number of Techniques: {len(unique_techniques)}")
print(f"\nMacro-Averaged Metrics:")
print(f"  Precision: {precision.mean():.3f}")
print(f"  Recall:    {recall.mean():.3f}")
print(f"  F1 Score:  {f1.mean():.3f}")
print(f"\nBest Performing Technique: {unique_techniques[np.argmax(f1)]} (F1={f1.max():.3f})")
print(f"Worst Performing Technique: {unique_techniques[np.argmax(1-f1)]} (F1={f1.min():.3f})")
print("\n" + "="*70)
print("All visualizations saved to slide_assets/")
print("="*70)
