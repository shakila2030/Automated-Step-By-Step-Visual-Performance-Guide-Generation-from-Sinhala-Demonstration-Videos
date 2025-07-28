from difflib import SequenceMatcher
import os

def sentence_similarity(a, b):
    return SequenceMatcher(None, a.strip(), b.strip()).ratio()

# Load files
with open('/content/final_instructions.txt', 'r', encoding='utf-8') as f:
    predictions = [line.strip() for line in f if line.strip()]

with open('/content/drive/MyDrive/FYP Tech Novas/groundTruth_instructions.txt', 'r', encoding='utf-8') as f:
    references = [line.strip() for line in f if line.strip()]

# Parameters
threshold = 0.7  # Consider a match if similarity ≥ 70%

# Matching loop
found_count = 0
for idx, ref in enumerate(references, 1):
    best_match = ""
    best_score = 0.0

    for pred in predictions:
        score = sentence_similarity(ref, pred)
        if score > best_score:
            best_score = score
            best_match = pred

    print(f"[{idx}] Reference: {ref}")
    if best_score >= threshold:
        found_count += 1
        print(f"  ✓ Found Match (Score: {best_score:.2f}): {best_match}\n")
    else:
        print(f"  ✗ No Good Match (Best Score: {best_score:.2f})\n")

# Summary
print("--- Final Recall Summary ---")
print(f"Total Reference Sentences   : {len(references)}")
print(f"Sentences Found (>{threshold*100:.0f}% sim) : {found_count}")
print(f"Recall                      : {found_count / len(references):.2%}")

