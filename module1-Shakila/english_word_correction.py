# english_word_correction.py

import re
import spacy
from googletrans import Translator
from sinlingua.singlish.rulebased_transliterator import RuleBasedTransliterator

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
translator = Translator()

# Instantiate the Sinlingua transliterator
transliterator = RuleBasedTransliterator()

def is_english_word(word):
    return bool(re.match("^[a-zA-Z]+$", word))

def process_text(text):
    words = text.split()
    output_words = []
    change_log = []

    for word in words:
        original_word = word
        if is_english_word(word):
            doc = nlp(word)
            pos = doc[0].pos_

            if pos == 'VERB':
                try:
                    translated = translator.translate(word, src='en', dest='si').text
                    if translated.endswith("කරන්න"):
                        translated = translated[:-len("කරන්න")].strip()
                    output_words.append(translated)
                    change_log.append(f"{original_word} ➜ {translated}")
                except Exception as e:
                    print(f"Translation failed for '{word}': {e}")
                    transliterated = transliterator.transliterator(word)
                    output_words.append(transliterated)
                    change_log.append(f"{original_word} ➜ {transliterated} (fallback)")
            elif pos == 'NOUN':
                transliterated = transliterator.transliterator(word)
                output_words.append(transliterated)
                change_log.append(f"{original_word} ➜ {transliterated}")
            else:
                transliterated = transliterator.transliterator(word)
                flagged = f"[[{transliterated}]]"
                output_words.append(flagged)
                change_log.append(f"{original_word} ➜ {flagged} (flagged)")
        else:
            output_words.append(word)

    print("\n🔍 Changes Made:\n")
    for change in change_log:
        print(change)

    return ' '.join(output_words)

def run_english_word_correction(input_path, output_path="afterEnglishCorrection.txt"):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("Correcting English Words in Transcript")
    print("📥 Original Input:\n")
    print(text)

    processed = process_text(text)
    print(processed)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(processed)

    print(f"\n✅ Processed output saved to '{output_path}'.")
