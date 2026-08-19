import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
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

# Get unique techniques
unique_techniques = sorted({t for techs in df["Techniques"] for t in techs if t})
technique_to_id = {t: i for i, t in enumerate(unique_techniques)}
df["Technique_ID"] = df["Technique_Primary"].map(lambda x: technique_to_id.get(x, -1))

print(f"\nDataset Info:")
print(f"  Total samples: {len(df)}")
print(f"  Manipulative samples: {(df['Manipulative'] == 1).sum()}")
print(f"  Non-manipulative samples: {(df['Manipulative'] == 0).sum()}")
print(f"  Unique techniques: {len(unique_techniques)}")
print(f"  Techniques: {unique_techniques}")

# Create simple features from dialogue length and word frequency
print("\nExtracting features...")
X = np.column_stack([
    df["Dialogue"].apply(len),  # Dialogue length
    df["Dialogue"].apply(lambda x: x.lower().count("you")),  # "you" frequency
    df["Dialogue"].apply(lambda x: x.lower().count("must")),  # "must" frequency
    df["Dialogue"].apply(lambda x: x.count("!")),  # Exclamation marks
    df["Dialogue"].apply(lambda x: x.count("?")),  # Question marks
    df["Dialogue"].apply(lambda x: len(x.split())),  # Word count
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

print("\n" + "="*60)
print("BINARY MANIPULATION CLASSIFICATION RESULTS")
print("="*60)
print(classification_report(y_test, y_pred, target_names=["Non-Manipulative", "Manipulative"]))

# Technique classifier (on manipulative samples only)
print("\nTraining technique classifier...")
mask_train = (y_train == 1)
X_multi_train = X_train[mask_train]
tech_multi_train = tech_train[mask_train]

# Remove invalid labels
valid_mask = (tech_multi_train != -1)
X_multi_train = X_multi_train[valid_mask]
y_multi_train = tech_multi_train[valid_mask]

if len(np.unique(y_multi_train)) > 1:
    # Oversample
    ros = RandomOverSampler(random_state=42)
    X_multi_bal, y_multi_bal = ros.fit_resample(X_multi_train, y_multi_train)
    
    # Train XGBoost
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
    
    # Evaluate on test set (manipulative samples only)
    mask_test = (y_test == 1)
    X_multi_test = X_test[mask_test]
    tech_multi_test = tech_test[mask_test]
    
    valid_mask_test = (tech_multi_test != -1)
    X_multi_test = X_multi_test[valid_mask_test]
    y_multi_test = tech_multi_test[valid_mask_test]
    
    if len(y_multi_test) > 0:
        y_multi_pred = clf_multi.predict(X_multi_test)
        
        print("\n" + "="*60)
        print("TECHNIQUE CLASSIFICATION RESULTS")
        print("="*60)
        print(f"Samples evaluated: {len(y_multi_test)}")
        print("\n" + classification_report(y_multi_test, y_multi_pred, target_names=unique_techniques))
        
        # Per-technique accuracy
        print("\nPer-Technique Accuracy:")
        print("-" * 50)
        for i, tech in enumerate(unique_techniques):
            mask = (y_multi_test == i)
            if mask.sum() > 0:
                acc = (y_multi_pred[mask] == i).mean()
                count = mask.sum()
                print(f"  {tech:.<40} {acc:.3f} ({count:3d} samples)")
        
        # Confusion matrix data
        cm = confusion_matrix(y_multi_test, y_multi_pred, labels=range(len(unique_techniques)))
        print("\n" + "="*60)
        print("Confusion Matrix (rows=true, cols=predicted)")
        print("="*60)
        # Print simplified version
        print(f"{'Technique':<30} {'Correct':>8} {'Total':>8}")
        print("-" * 50)
        for i, tech in enumerate(unique_techniques):
            correct = cm[i, i]
            total = cm[i].sum()
            print(f"{tech:<30} {correct:>8} {total:>8}")
else:
    print("Warning: Not enough technique diversity in training set for technique classifier.")

print("\n" + "="*60)
print("Classification Complete!")
print("="*60)
