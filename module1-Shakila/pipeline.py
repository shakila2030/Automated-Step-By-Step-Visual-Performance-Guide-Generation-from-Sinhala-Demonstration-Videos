# pipeline.py (Hybrid: transcript.txt in Colab, rest in Drive)

import sys
import os
from google.colab import drive


# === Step 1: Set base path to your project folder inside Google Drive ===
drive_base = '/content/drive/MyDrive/FYP Tech Novas/Model 1'

# === Step 2: Add base path to Python's sys.path so you can import your modules ===
sys.path.append(drive_base)

# === Step 3: Import modules from Google Drive ===
from breaking_into_sentences import preprocess_file
from merge_words import merge_words
from unmerge import fix_incorrect_merges
from correct_misspelled_words import correct_misspellings
from english_word_correction import run_english_word_correction
from masked_modelling import run_masked_prediction

# === Step 4: Define file paths ===

# 📍 transcript.txt is stored in Colab local /content
input_text_file     = '/content/transcript.txt'

# 🗂️ All other files are in Google Drive
step1_output        = os.path.join(drive_base, 'after_breaking_sentences.txt')
corpus_file         = os.path.join(drive_base, 'word_dataset.txt')
step2_output        = os.path.join(drive_base, 'with_merged_words.txt')
step3_output        = os.path.join(drive_base, 'corrected_output.txt')
step4_output        = os.path.join(drive_base, 'afterSpellingCorrection.txt')
unmatched_output    = os.path.join(drive_base, 'unmatched_words.txt')
corpus2_file        = os.path.join(drive_base, 'corpus2.txt')
kenlm_model_file    = os.path.join(drive_base, 'sinhala.arpa.bin')
step5_output        = os.path.join(drive_base, 'afterEnglishCorrection.txt')
bow_file            = os.path.join(drive_base, 'sentence_dataset.txt')
final_output        = os.path.join(drive_base, 'predicted_sentences.txt')

# === Step 5: Run Pipeline Steps ===


preprocess_file(input_text_file, step1_output)


merge_words(corpus_file, step1_output, step2_output)


fix_incorrect_merges(corpus_file, step2_output, step3_output)


correct_misspellings(step3_output, step4_output, unmatched_output,
                     corpus_file, corpus2_file, kenlm_model_file)


run_english_word_correction(step4_output, step5_output)


run_masked_prediction(step5_output, bow_file, final_output)

print("✅ All steps completed. Final output saved to:", final_output)
