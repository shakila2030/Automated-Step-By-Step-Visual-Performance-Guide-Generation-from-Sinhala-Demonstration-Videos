# step2_sentence_classification.py

import sys
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
import re

# Add your project path to import the rule-based classifier
sys.path.append('/content/drive/MyDrive/FYP Tech Novas/Model 2')  # Adjust if needed

from rule_based_instructionClassifier import SinhalaRuleCategorizer

# Function to split Sinhala text into sentences based on punctuation
def split_sinhala_sentences(text):
    # Split on Sinhala and common punctuation marks
    return [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]

# Load classifier and embedder
rule_categorizer = SinhalaRuleCategorizer()
clf = joblib.load('/content/drive/MyDrive/FYP Tech Novas/Model 2/instructionClassifier.joblib')
embedder = SentenceTransformer('sentence-transformers/LaBSE')

# Sample sentences for grid search tuning (you can keep or remove this)
new_sentences = [
    "දැන් තෙල් දමන්න.",
    "මේක හරිම ලේසියි",
    "ඊළඟට මිරිස් කුඩු දමන්න.",
    "ගෙදරදිම හරිම ලේසියෙන් ඉක්මනට එග් ෆ්‍රයිඩ් රයිස් හදන හැටි මං අද ඔයාලට කියලා දෙනවා",
    "කෝප්ප තුනක් කිරි දාන්න.",
    "සබ්ස්ක්‍රයිබ් කරන්න අමතක කරන්න එපා!",
    "හරිම පදමට මේක විනාඩියක් නීඩ් කර ගතයුතුයි"
]

# Rule-based classification on sample data
rule_based_results = []
for s in new_sentences:
    category, keep = rule_categorizer.categorize_and_filter(s)
    rule_based_results.append((category, keep))

# ML model predictions on sample data
new_embeddings = embedder.encode(new_sentences)
probs = clf.predict_proba(new_embeddings)

# Grid search to find best weights and threshold
true_labels = [1, 0, 1, 0, 1, 0, 1]  # Your ground truth for tuning
best_score = 0
best_params = None

def confidence_score(rule_keep, ml_conf, rule_weight, ml_weight):
    rule_score = 1.0 if rule_keep == 1 else 0.0
    return rule_weight * rule_score + ml_weight * ml_conf

for rule_weight in [0.3, 0.4, 0.5, 0.6, 0.7]:
    ml_weight = 1.0 - rule_weight
    for threshold in [0.4, 0.5, 0.6, 0.7]:
        preds = []
        for (_, rule_keep), prob in zip(rule_based_results, probs):
            ml_conf = prob[1]
            score = confidence_score(rule_keep, ml_conf, rule_weight, ml_weight)
            final_label = 1 if score >= threshold else 0
            preds.append(final_label)
        acc = accuracy_score(true_labels, preds)
        if acc > best_score:
            best_score = acc
            best_params = (rule_weight, ml_weight, threshold)

if best_params is None:
    print("⚠️ No best params found. Using defaults.")
    rule_weight, ml_weight, threshold = 0.6, 0.4, 0.5
else:
    rule_weight, ml_weight, threshold = best_params

print(f"\n🏆 Best Params from Grid Search: rule={rule_weight}, ml={ml_weight}, threshold={threshold}, accuracy={best_score:.2f}\n")

print("🔍 Final Confidence-Based Scoring Results:\n")
for sentence, (category, rule_keep), prob in zip(new_sentences, rule_based_results, probs):
    ml_conf = prob[1]
    rule_score = 1.0 if rule_keep == 1 else 0.0
    final_score = confidence_score(rule_keep, ml_conf, rule_weight, ml_weight)
    final_label = 1 if final_score >= threshold else 0
    final_mark = "✅" if final_label == 1 else "❌"
    print(f"{final_mark} Score: {final_score:.2f} | Rule: {'✅' if rule_keep else '❌'} | ML: {ml_conf:.2f} | [{category}] - {sentence}")

# --- Now process the filtered_script.txt file ---

INPUT_FILE = "/content/filtered_script.txt"
OUTPUT_FILE = "/content/final_classified_instructions.txt"

sentences = []
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Split long lines into shorter sentences
        parts = split_sinhala_sentences(line)
        sentences.extend(parts)

pred_labels = []
final_instruction_lines = []

print("\n📄 Prediction Results for Filtered Script:\n")
for sent in sentences:
    category, rule_keep = rule_categorizer.categorize_and_filter(sent)
    if category in ["Introduction", "Engagement"]:
        final_label = rule_keep
        ml_conf = None
        final_score = None
    else:
        embedding = embedder.encode([sent])
        ml_conf = clf.predict_proba(embedding)[0][1]
        rule_score = 1.0 if rule_keep == 1 else 0.0
        final_score = rule_weight * rule_score + ml_weight * ml_conf
        final_label = 1 if final_score >= threshold else 0

    pred_labels.append(final_label)

    if final_label == 1:
        final_instruction_lines.append(sent)

    final_mark = "✅" if final_label == 1 else "❌"
    ml_part = "" if ml_conf is None else f" | ML: {ml_conf:.2f}"
    print(f"{final_mark} | Score: {final_score if final_score is not None else '-'} | Rule: {category} |{ml_part} {sent}")

# Save instructions only to file
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for line in final_instruction_lines:
        f.write(line + "."+ '\n')

print(f"\n✅ Final instructions saved to: {OUTPUT_FILE}")
