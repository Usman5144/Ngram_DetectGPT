import os
import time
import csv
import re
import pandas as pd
from tqdm import tqdm
import openai

# -------------------------------
NUM_SAMPLES = 300
INPUT_FILES = ["xsum_gpt-4.raw_data.csv"]
TEMPERATURES = [0.5, 0.9] #[0.3, 0.7, 1.1] #will write temp here
MODEL_NAME = "deepseek-chat"
OUTPUT_DIR = "deepseek_output"
API_KEY = "sk-90e828ed73c54ff5b7a952ac8d103a27"
WAIT_SECONDS = 3
# -------------------------------

client = openai.OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/v1")

class DeepSeekParaphraser:
    def __init__(self, temperature, writer, file_handle):
        self.temperature = temperature
        self.writer = writer
        self.file_handle = file_handle

    def build_messages(self, text: str):
        return [
            {
                "role": "user",
                "content": (
                    f"Give me exactly 10 high-quality, fluent paraphrases of the following sentence. "
                    f"Return them as a numbered list ONLY, no explanations or intro text.\n\n"
                    f"Text: \"{text.strip()}\""
                )
            }
        ]

    def extract_variants_from_list(self, raw):
        lines = raw.strip().split("\n")
        variants = []
        for line in lines:
            match = re.match(r"^\s*\d+\.\s*(.+)", line)
            if match:
                variants.append(match.group(1).strip())
        return variants if len(variants) == 10 else []

    def generate_and_stream_variations(self, texts, tags):
        for i, (text, tag) in enumerate(tqdm(zip(texts, tags), total=len(texts), desc=f"Generating (T={self.temperature})", unit="row")):
            if not text.strip():
                self.writer.writerow([text, tag] + [""] * 10)
                continue

            for attempt in range(2):
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        temperature=self.temperature,
                        messages=self.build_messages(text)
                    )
                    content = response.choices[0].message.content.strip()
                    variants = self.extract_variants_from_list(content)

                    if len(variants) == 10:
                        self.writer.writerow([text, tag] + variants)
                        break
                except Exception as e:
                    print(f"⚠️ Row {i+1} attempt {attempt+1} failed: {e}")
                    time.sleep(1)
            else:
                self.writer.writerow([text, tag] + ["FAILED_VARIANT"] * 10)

            self.file_handle.flush()
            time.sleep(WAIT_SECONDS)

def main():
    for file in INPUT_FILES:
        df = pd.read_csv(file)
        df = df.dropna()
        df = df[df["label"].isin(["human_answers", "chatgpt_answers"])]
        df = df.sample(n=min(NUM_SAMPLES, len(df)), random_state=42)

        originals = df["text"].tolist()
        tags = ["human" if lbl == "human_answers" else "ai" for lbl in df["label"].tolist()]

        for temp in TEMPERATURES:
            base = os.path.splitext(file)[0]
            out_path = os.path.join(OUTPUT_DIR, f"{base}_variants_deepseek_temp{temp}.csv")
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["original", "tag"] + [f"variant_{i+1}" for i in range(10)]
                writer.writerow(header)

                gen = DeepSeekParaphraser(temperature=temp, writer=writer, file_handle=f)
                gen.generate_and_stream_variations(originals, tags)

if __name__ == "__main__":
    main()
