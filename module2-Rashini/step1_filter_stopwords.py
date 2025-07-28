# step1_filter_stopwords.py

from indicnlp.tokenize import indic_tokenize
import os
import re

SCRIPT_PATH = "/content/drive/MyDrive/FYP Tech Novas/Model 1/predicted_sentences.txt"
STOPWORDS_PATH = "/content/drive/MyDrive/FYP Tech Novas/Model 2/stopword list.txt"
OUTPUT_PATH = "/content/filtered_script.txt"

def load_stopwords(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        stopwords = set(line.strip() for line in f if line.strip())
    return stopwords

def process_script(script_path, stopwords):
    with open(script_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Sort stopwords by length (to remove longer phrases first)
    sorted_stopwords = sorted(stopwords, key=len, reverse=True)

    processed_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            processed_lines.append("")
            continue

        # Tokenize first to avoid substring issues
        words = indic_tokenize.trivial_tokenize(line, lang='si')

        # Reconstruct sentence while removing exact matches only
        filtered_words = []

        i = 0
        while i < len(words):
            matched = False
            # Try to match multi-word stopwords first
            for sw in sorted_stopwords:
                sw_tokens = sw.split()
                sw_len = len(sw_tokens)
                if words[i:i+sw_len] == sw_tokens:
                    matched = True
                    i += sw_len  # Skip matched stopword phrase
                    break
            if not matched:
                filtered_words.append(words[i])
                i += 1

        processed_sentence = " ".join(filtered_words)
        processed_lines.append(processed_sentence.strip())

    return processed_lines

def main():
    stopwords = load_stopwords(STOPWORDS_PATH)
    print(f"✅ Loaded {len(stopwords)} stopwords.")

    filtered_lines = process_script(SCRIPT_PATH, stopwords)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for line in filtered_lines:
            f.write(line + "\n")

    print(f"✅ Filtered script saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
