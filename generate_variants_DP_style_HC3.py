import os
import time
import csv
import json
import requests
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

# -------------------------------
NUM_SAMPLES_PER_CLASS = 150
TEMPERATURES = [0.1, 0.5, 0.9]
MODEL_NAME = "deepseek-chat"
OUTPUT_DIR = "deepseek_output"
API_KEY = "sk-90e828ed73c54ff5b7a952ac8d103a27"
WAIT_SECONDS = 3
# -------------------------------

DEESEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}


class DeepSeekParaphraser:
    def __init__(self, temperature, writer, file_handle):
        self.temperature = temperature
        self.writer = writer
        self.file_handle = file_handle
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def build_payload(self, text):
        prompt = (
            "You are a JSON generator. Rephrase the input text into exactly 10 high-quality, fluent, "
            "semantically consistent paraphrases. Return ONLY this format:\n\n"
            '{"variants": [\n'
            '{"text": "..."},\n'
            '{"text": "..."},\n'
            '... 10 items total\n]}'
            f'\n\nInput:\n\"\"\"{text}\"\"\"'
        )
        return {
            "model": MODEL_NAME,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}]
        }

    def extract_json_block(self, raw_text):
        # Remove markdown-style formatting like ```json ... ```
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").strip("json").strip()
        return raw_text

    def generate_and_stream_variations(self, texts, tags):
        for i, (text, tag) in enumerate(tqdm(zip(texts, tags), total=len(texts), desc=f"Generating (T={self.temperature})", unit="row")):
            if not text.strip():
                self.writer.writerow([text, tag] + [""] * 10)
                continue

            for attempt in range(2):  # Allow 1 retry
                try:
                    payload = self.build_payload(text)
                    response = requests.post(DEESEEK_ENDPOINT, headers=HEADERS, json=payload, timeout=60)
                    response.raise_for_status()
                    content = response.json()

                    reply = content["choices"][0]["message"]["content"]
                    cleaned = self.extract_json_block(reply)

                    parsed = json.loads(cleaned)
                    variants = [v["text"].strip() for v in parsed.get("variants", [])]

                    if len(variants) == 10 and all(variants):
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
    print(" Loading HC3 dataset...")
    ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train", trust_remote_code=True)

    human = [e["human_answers"][0] for e in ds if e["human_answers"]]
    ai = [e["chatgpt_answers"][0] for e in ds if e["chatgpt_answers"]]

    human = human[:NUM_SAMPLES_PER_CLASS]
    ai = ai[:NUM_SAMPLES_PER_CLASS]

    originals = human + ai
    tags = ["human"] * NUM_SAMPLES_PER_CLASS + ["ai"] * NUM_SAMPLES_PER_CLASS

    for temp in TEMPERATURES:
        out_path = os.path.join(OUTPUT_DIR, f"hc3_deepseek.raw_data_variants_temp{temp}.csv")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["original", "tag"] + [f"variant_{i+1}" for i in range(10)]
            writer.writerow(header)

            gen = DeepSeekParaphraser(temperature=temp, writer=writer, file_handle=f)
            gen.generate_and_stream_variations(originals, tags)

if __name__ == "__main__":
    main()
