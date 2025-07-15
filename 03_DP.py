import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# === Shared Configuration ===
DELTA_FEATURES = [
    "delta_log_score_full_2", "delta_entropy_2gram", "delta_variance_2gram",
    "delta_log_score_full_3", "delta_entropy_full_3", "delta_variance_full_3",
    "delta_log_score_full_4", "delta_entropy_4gram", "delta_variance_4gram",
    "delta_log_score_full_5", "delta_entropy_5gram", "delta_variance_5gram"
]

TEMP_LIST = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
DATASETS = {
    "pubmed": "../pubmed_output_DP",
    "hc3": "../hc3_output_DP",
    "writing": "../writing_output_DP",
    "xsum": "../xsum_output_DP"
}

# === Model Configurations ===
model_groups = {
    "Full 2-Gram Only": DELTA_FEATURES[0:3],
    "Full 3-Gram Only": DELTA_FEATURES[3:6],
    "Full 4-Gram Only": DELTA_FEATURES[6:9],
    "Full 5-Gram Only": DELTA_FEATURES[9:],
    "Full 2-Gram + Full 3-Gram": DELTA_FEATURES[0:6],
    "Full 2-Gram + Full 4-Gram": DELTA_FEATURES[0:3] + DELTA_FEATURES[6:9],
    "Full 2-Gram + Full 5-Gram": DELTA_FEATURES[0:3] + DELTA_FEATURES[9:],
    "Full 3-Gram + Full 4-Gram": DELTA_FEATURES[3:9],
    "Full 3-Gram + Full 5-Gram": DELTA_FEATURES[3:6] + DELTA_FEATURES[9:],
    "Full 4-Gram + Full 5-Gram": DELTA_FEATURES[6:],
    "All Combined": DELTA_FEATURES
}

for dataset, data_dir in DATASETS.items():
    print(f"\n=== Processing Dataset: {dataset.upper()} ===")
    results = []
    all_df = []

    for temp in TEMP_LIST:
        file_path = os.path.join(data_dir, f"kenlm_filtered_features_{dataset}_temp{temp}.csv")
        if not os.path.exists(file_path):
            print(f" Missing file: {file_path}")
            continue

        print(f"\n Evaluating temperature = {temp}")
        df = pd.read_csv(file_path)
        df["label_num"] = df["label"].map({"human": 0, "ai": 1})

        X = df[DELTA_FEATURES]
        y = df["label_num"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=3)
        cm = confusion_matrix(y_test, y_pred)

        print(report)
        print(f"ROC-AUC: {auc:.4f} | Accuracy: {acc:.4f}")

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
        plt.title(f"Confusion Matrix ({dataset.upper()} Temp {temp})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        cm_path = os.path.join(data_dir, f"confusion_matrix_{dataset}_temp{temp}.png")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()

        results.append({"Temperature": temp, "Accuracy": acc, "ROC-AUC": auc})
        df["temperature"] = temp
        all_df.append(df)

    if results:
        summary_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
        print(f"\n=== Temperature-wise Performance Summary ({dataset.upper()}) ===")
        print(summary_df.to_string(index=False))
        summary_df.to_csv(os.path.join(data_dir, f"temp_wise_model_summary_{dataset}.csv"), index=False)

    if all_df:
        merged_df = pd.concat(all_df, ignore_index=True)
        model_results = []

        for name, features in model_groups.items():
            X = merged_df[features]
            y = merged_df["label_num"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, y_prob)
            acc = accuracy_score(y_test, y_pred)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

            model_results.append({
                "Model Set": name,
                "Test AUC": round(auc, 4),
                "Test Accuracy": round(acc, 4),
                "CV AUC Mean": round(np.mean(cv_scores), 4),
                "CV AUC Std": round(np.std(cv_scores), 4),
                "Overfit Gap": round(abs(auc - np.mean(cv_scores)), 4)
            })

        model_df = pd.DataFrame(model_results).sort_values(by="Test AUC", ascending=False)
        print(f"\n=== Model Performance Comparison ({dataset.upper()}) ===")
        print(model_df.to_string(index=False))
        model_df.to_csv(os.path.join(data_dir, f"model_wise_comparison_{dataset}.csv"), index=False)

print("\n All dataset evaluations completed.")
