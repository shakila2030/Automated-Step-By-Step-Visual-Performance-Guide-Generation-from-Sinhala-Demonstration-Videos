# masked_model_predictor.py

import os
import re
from indicnlp import common
from indicnlp.tokenize import indic_tokenize
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

# --- Set Indic resources path ---
INDIC_RESOURCES_PATH = './indic_nlp_resources'
common.set_resources_path(INDIC_RESOURCES_PATH)

# --- Load fine-tuned Sinhala BERT MLM model ---
MODEL_PATH = '/content/drive/MyDrive/FYP Tech Novas/Model 1/sinhala_berto_finetuned'  # update if needed
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH, local_files_only=True)
fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)

def is_valid_token(token_str):
    token_clean = token_str.strip().replace("▁", "")
    token_clean = re.sub(r'[^\u0D80-\u0DFF]', '', token_clean)
    return len(token_clean) >= 2

def load_bow_words(bow_path):
    with open(bow_path, 'r', encoding='utf-8') as f:
        bow_words = set([line.strip() for line in f if line.strip()])
    print(f"✅ Loaded {len(bow_words)} words from {bow_path}")
    return bow_words

def combined_score(mlm_score, token_str, bow_words, mlm_weight=0.8, bow_weight=0.2):
    token_clean = token_str.strip().replace("▁", "")
    token_clean = re.sub(r'[^\u0D80-\u0DFF]', '', token_clean)
    bow_score = 1.0 if token_clean in bow_words else 0.0
    return mlm_weight * mlm_score + bow_weight * bow_score

def predict_word(masked_sentence, bow_words):
    results = fill_mask(masked_sentence, top_k=30)
    best_score = -1.0
    best_token = None

    for pred in results:
        token_str = pred['token_str']
        mlm_score = pred['score']
        if is_valid_token(token_str):
            score = combined_score(mlm_score, token_str, bow_words)
            if score > best_score:
                best_score = score
                best_token = token_str.strip()
    return best_token

def process_sentence(sentence, bow_words, window_size=20):
    flagged_pattern = r'\[\[(.*?)\]\]'
    flagged_words = re.findall(flagged_pattern, sentence)
    
    if not flagged_words:
        return sentence
    
    tokens = indic_tokenize.trivial_tokenize(sentence, lang='si')
    new_sentence = sentence

    for flagged in flagged_words:
        try:
            idx = tokens.index(flagged)
        except ValueError:
            print(f"⚠️ Warning: flagged word '{flagged}' not found in tokens, skipping.")
            continue

        start = max(0, idx - window_size)
        end = min(len(tokens), idx + window_size + 1)
        context_tokens = tokens[start:end]
        masked_idx = idx - start
        context_tokens[masked_idx] = tokenizer.mask_token

        masked_sentence = " ".join(context_tokens)
        prediction = predict_word(masked_sentence, bow_words)

        if prediction:
            new_sentence = new_sentence.replace(f"[[{flagged}]]", f"<{prediction}>")
        else:
            print(f"⚠️ Warning: No valid prediction found for '{flagged}'.")

    return new_sentence

def run_masked_prediction(input_file, bow_file, output_file="predicted_sentences.txt"):
    print("----------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("Predicting Flagged Words Using Masked Modelling and Bag of Words")
    bow_words = load_bow_words(bow_file)

    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    print(f"✅ Loaded {len(sentences)} sentences from {input_file}")

    processed_sentences = []
    print("\n--- Input and Output ---")
    for sentence in sentences:
        updated = process_sentence(sentence, bow_words)
        processed_sentences.append(updated)
        print("Input:  ", sentence)
        print("Output: ", updated)
        print()

    with open(output_file, "w", encoding='utf-8') as out_file:
        for s in processed_sentences:
            out_file.write(s + "\n")

    print(f"\n✅ All sentences processed and saved to '{output_file}'.")
