# === Imports ===
import os
import pandas as pd
import numpy as np
import nltk
import kenlm
from collections import Counter
from scipy.stats import entropy as calc_entropy
from pathlib import Path
nltk.download('punkt')

# === KenLM Model Paths ===
MODEL_DIR = '../models'
MODEL_FULL_2GRAM = os.path.join(MODEL_DIR, '2-gram.arpa')
MODEL_FULL_3GRAM = os.path.join(MODEL_DIR, '3-gram.arpa')
MODEL_FULL_4GRAM = os.path.join(MODEL_DIR, '4-gram.arpa')
MODEL_FULL_5GRAM = os.path.join(MODEL_DIR, '5-gram.arpa')  

# === Load KenLM Models ===
print('Loading KenLM models...')
lm_full_2 = kenlm.Model(MODEL_FULL_2GRAM)
lm_full_3 = kenlm.Model(MODEL_FULL_3GRAM)
lm_full_4 = kenlm.Model(MODEL_FULL_4GRAM)
lm_full_5 = kenlm.Model(MODEL_FULL_5GRAM)  
print('Models loaded.')

# === Utility Functions ===

def get_ngrams(text, n=3):
    tokens = text.lower().split()
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def score_log(model, text):
    """
    Calculates the log-probability score of a given text using the KenLM model.

    Args:
        model (kenlm.Model): Preloaded KenLM language model (e.g., 3-gram or 4-gram).
        text (str): Sentence or passage to be scored.

    Returns:
        float: Log-probability score (base e). Higher = more fluent.
    """
    return model.score(text)

def calculate_entropy(ngrams):
    count = Counter(ngrams)
    total = sum(count.values())
    probs = [v / total for v in count.values()]
    return calc_entropy(probs)

def frequency_variance(ngrams):
    count = Counter(ngrams)
    values = list(count.values())
    return np.var(values) if len(values) > 1 else 0

# === Core Feature Extraction === 
def extract_all_features(text, sample_id=None, temp_label=None, variation_id=None):
    ngrams_2 = get_ngrams(text, 2)
    ngrams_3 = get_ngrams(text, 3)
    ngrams_4 = get_ngrams(text, 4)
    ngrams_5 = get_ngrams(text, 5)

    # === Save the n-grams to CSV ===
    if sample_id is not None and temp_label is not None and variation_id is not None:
        base_out = f"../ngrams_output/sample{sample_id}_temp{temp_label}_var{variation_id}"
        os.makedirs("../ngrams_output", exist_ok=True)

        pd.DataFrame(ngrams_2, columns=["2-gram"]).to_csv(f"{base_out}_2gram.csv", index=False)
        pd.DataFrame(ngrams_3, columns=["3-gram"]).to_csv(f"{base_out}_3gram.csv", index=False)
        pd.DataFrame(ngrams_4, columns=["4-gram"]).to_csv(f"{base_out}_4gram.csv", index=False)
        pd.DataFrame(ngrams_5, columns=["5-gram"]).to_csv(f"{base_out}_5gram.csv", index=False)

    return {
        "log_score_full_2": score_log(lm_full_2, text),
        "entropy_2gram": calculate_entropy(ngrams_2),           
        "variance_2gram": frequency_variance(ngrams_2),
        "log_score_full_3": score_log(lm_full_3, text),
        "entropy_full_3": calculate_entropy(ngrams_3),  
        "variance_full_3": frequency_variance(ngrams_3),
        "log_score_full_4": score_log(lm_full_4, text),
        "entropy_4gram": calculate_entropy(ngrams_4),
        "variance_4gram": frequency_variance(ngrams_4),
        "log_score_full_5": score_log(lm_full_5, text),  
        "entropy_5gram": calculate_entropy(ngrams_5),    
        "variance_5gram": frequency_variance(ngrams_5),  
    }



#def extract_all_features(text):
#    ngrams_2 = get_ngrams(text, 2)
#    ngrams_3 = get_ngrams(text, 3)
#    ngrams_4 = get_ngrams(text, 4)
#    ngrams_5 = get_ngrams(text, 5)  

#    return {
#        "log_score_full_2": score_log(lm_full_2, text),
#        "entropy_2gram": calculate_entropy(ngrams_2),           
#        "variance_2gram": frequency_variance(ngrams_2),
#        "log_score_full_3": score_log(lm_full_3, text),
#        "entropy_full_3": calculate_entropy(ngrams_3),  
#        "variance_full_3": frequency_variance(ngrams_3),
#        "log_score_full_4": score_log(lm_full_4, text),
#        "entropy_4gram": calculate_entropy(ngrams_4),
#        "variance_4gram": frequency_variance(ngrams_4),
#        "log_score_full_5": score_log(lm_full_5, text),  
#        "entropy_5gram": calculate_entropy(ngrams_5),    
#        "variance_5gram": frequency_variance(ngrams_5),  
#    }



# === Feature Extraction ===

def process_variations(df, temp_label):
    rows = []
    skipped = 0
    for i, row in df.iterrows():
        original_text = str(row["original"])
        label = row["tag"].strip().lower()

        try:
            original_feats = extract_all_features(original_text, i, temp_label, "original")
        except:
            print(f"Failed to extract features from original at row {i}")
            skipped += 1
            continue

        variant_found = False

        for j in range(1, 11):
            colname = f"variant_{j}"
            if colname not in row or pd.isna(row[colname]):
                continue

            variant = str(row[colname]).strip()
            if len(variant.split()) < 3:
                continue

            try:
                variant_feats = extract_all_features(variant, i, temp_label, j)
            except:
                print(f"⚠️ Failed variant features at row {i}, col {colname}")
                continue

            delta_feats = {
                f"delta_{k}": abs(original_feats[k] - variant_feats[k])
                for k in original_feats
            }

            row_data = {
                "sample_id": i,
                "temperature": temp_label,
                "variation_id": j,
                "label": label,
                "source_text": original_text,
                "variation_text": variant,
            }
            row_data.update(delta_feats)
            rows.append(row_data)
            variant_found = True

        if not variant_found:
            skipped += 1

    print(f"Processed temp {temp_label}: {len(rows)} pairs extracted, {skipped} rows skipped.")
    return pd.DataFrame(rows)


# === Batch Processing Functions ===

# --- HC3 ---
def batch_process_hc3():
    base_path = "."  # same folder as script
    output_dir = "../hc3_output_DP"
    os.makedirs(output_dir, exist_ok=True)

    hc3_files = {
        0.1: "hc3_deepseek.raw_data_variants_temp0.1.csv",
        0.3: "hc3_deepseek.raw_data_variants_temp0.3.csv",
        0.5: "hc3_deepseek.raw_data_variants_temp0.5.csv",
        0.7: "hc3_deepseek.raw_data_variants_temp0.7.csv",
        0.9: "hc3_deepseek.raw_data_variants_temp0.9.csv",
        1.1: "hc3_deepseek.raw_data_variants_temp1.1.csv",
    }

    all_hc3 = []

    for temp, filename in hc3_files.items():
        file_path = os.path.join(base_path, filename)
        output_path = os.path.join(output_dir, f"kenlm_filtered_features_hc3_temp{temp}.csv")

        if not os.path.exists(file_path):
            print(f"HC3 file not found: {file_path}")
            continue

        print(f"\\nReading HC3: {file_path}")
        df = pd.read_csv(file_path)

        expected = ['original', 'tag'] + [f"variant_{i}" for i in range(1, 11)]
        if any(col not in df.columns for col in expected):
            print(f"Missing required columns in {file_path}")
            continue

        df_feat = process_variations(df, temp)
        if not df_feat.empty:
            df_feat.to_csv(output_path, index=False)
            print(f"Saved HC3 features to: {output_path}")
            all_hc3.append(df_feat)
        else:
            print(" No valid HC3 variants processed.")

    return all_hc3

# === Run === #
if __name__ == "__main__":
    print("\\n=== Processing HC3 Dataset ===")
    hc3_data = batch_process_hc3()
    print("\\n HC3 dataset processed.")



# --- PUBMED ---
def batch_process_pubmed():
    base_path = "."  # same folder as script
    output_dir = "../pubmed_output_DP"
    os.makedirs(output_dir, exist_ok=True)

    pubmed_files = {
        0.1: "pubmed_gpt-4.raw_data_variants_deepseek_temp0.1.csv",
        0.3: "pubmed_gpt-4.raw_data_variants_deepseek_temp0.3.csv",
        0.5: "pubmed_gpt-4.raw_data_variants_deepseek_temp0.5.csv",
        0.7: "pubmed_gpt-4.raw_data_variants_deepseek_temp0.7.csv",
        0.9: "pubmed_gpt-4.raw_data_variants_deepseek_temp0.9.csv",
        1.1: "pubmed_gpt-4.raw_data_variants_deepseek_temp1.1.csv",
    }

    all_pubmed = []

    for temp, filename in pubmed_files.items():
        file_path = os.path.join(base_path, filename)
        output_path = os.path.join(output_dir, f"kenlm_filtered_features_pubmed_temp{temp}.csv")

        if not os.path.exists(file_path):
            print(f"PubMed file not found: {file_path}")
            continue

        print(f"\nReading PubMed: {file_path}")
        df = pd.read_csv(file_path)

        expected = ['original', 'tag'] + [f"variant_{i}" for i in range(1, 11)]
        if any(col not in df.columns for col in expected):
            print(f"Missing required columns in {file_path}")
            continue

        df_feat = process_variations(df, temp)
        if not df_feat.empty:
            df_feat.to_csv(output_path, index=False)
            print(f"Saved PubMed features to: {output_path}")
            all_pubmed.append(df_feat)
        else:
            print(" No valid PubMed variants processed.")

    return all_pubmed

# === Run === #
if __name__ == "__main__":
    print("\n=== Processing PubMed Dataset ===")
    pubmed_data = batch_process_pubmed()
    print("\n PubMed dataset processed.")


# --- WRITING ---
def batch_process_writing():
    base_path = "."  # current folder
    output_dir = "../writing_output_DP"
    os.makedirs(output_dir, exist_ok=True)

    writing_files = {
        0.1: "writing_gpt-4.raw_data_variants_deepseek_temp0.1.csv",
        0.3: "writing_gpt-4.raw_data_variants_deepseek_temp0.3.csv",
        0.5: "writing_gpt-4.raw_data_variants_deepseek_temp0.5.csv",
        0.7: "writing_gpt-4.raw_data_variants_deepseek_temp0.7.csv",
        0.9: "writing_gpt-4.raw_data_variants_deepseek_temp0.9.csv",
        1.1: "writing_gpt-4.raw_data_variants_deepseek_temp1.1.csv",
    }

    all_frames = []

    for temp, filename in writing_files.items():
        file_path = os.path.join(base_path, filename)
        output_path = os.path.join(output_dir, f"kenlm_filtered_features_writing_temp{temp}.csv")

        if not os.path.exists(file_path):
            print(f"Writing file not found: {file_path}")
            continue

        print(f"\nReading Writing Data: {file_path}")
        df = pd.read_csv(file_path)

        expected = ['original', 'tag'] + [f"variant_{i}" for i in range(1, 11)]
        if any(col not in df.columns for col in expected):
            print(f"Missing required columns in {file_path}")
            continue

        df_feat = process_variations(df, temp)
        if not df_feat.empty:
            df_feat.to_csv(output_path, index=False)
            print(f"Saved Writing features to: {output_path}")
            all_frames.append(df_feat)
        else:
            print(" No valid writing variants processed.")

    return all_frames

# === Run === #
if __name__ == "__main__":
    print("\n=== Processing Writing Dataset ===")
    writing_data = batch_process_writing()
    print("\n Writing dataset processed.")


# --- XSUM ---
def batch_process_xsum():
    base_path = "."  # current directory
    output_dir = "../xsum_output_DP"
    os.makedirs(output_dir, exist_ok=True)

    xsum_files = {
        0.1: "xsum_gpt-4.raw_data_variants_deepseek_temp0.1.csv",
        0.3: "xsum_gpt-4.raw_data_variants_deepseek_temp0.3.csv",
        0.5: "xsum_gpt-4.raw_data_variants_deepseek_temp0.5.csv",
        0.7: "xsum_gpt-4.raw_data_variants_deepseek_temp0.7.csv",
        0.9: "xsum_gpt-4.raw_data_variants_deepseek_temp0.9.csv",
        1.1: "xsum_gpt-4.raw_data_variants_deepseek_temp1.1.csv",
    }

    all_data = []

    for temp, filename in xsum_files.items():
        file_path = os.path.join(base_path, filename)
        output_path = os.path.join(output_dir, f"kenlm_filtered_features_xsum_temp{temp}.csv")

        if not os.path.exists(file_path):
            print(f"XSum file not found: {file_path}")
            continue

        print(f"\nReading XSum file: {file_path}")
        df = pd.read_csv(file_path)

        expected = ['original', 'tag'] + [f"variant_{i}" for i in range(1, 11)]
        if any(col not in df.columns for col in expected):
            print(f"Missing required columns in {file_path}")
            continue

        df_feat = process_variations(df, temp)
        if not df_feat.empty:
            df_feat.to_csv(output_path, index=False)
            print(f" Saved XSum features to: {output_path}")
            all_data.append(df_feat)
        else:
            print("⚠️ No valid variants processed.")

    return all_data

# === Run === #
if __name__ == "__main__":
    print("\n=== Processing XSum Dataset ===")
    xsum_data = batch_process_xsum()
    print("\n XSum dataset processed.")


if __name__ == "__main__":
    print("\n=== Processing All Datasets ===")
    batch_process_pubmed()
    batch_process_hc3()
    batch_process_writing()
    batch_process_xsum()
    print("\n All datasets processed.")
