import re
import unicodedata
from sentence_transformers import SentenceTransformer, util
from sinling import SinhalaTokenizer, POSTagger

# --- Unicode normalization helper ---
def normalize(text):
    return unicodedata.normalize("NFC", text)

# --- Load model and tools ---
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
tokenizer = SinhalaTokenizer()
pos_tagger = POSTagger()

# --- Define supporting lists and logic ---
ingredient_intros = ['අවශ්‍ය', 'අවශ්‍යයි', 'අවශ්‍ය වෙනවා', 'ගන්න']

# Expanded measurement units with variants
measurement_units = [
    "කෝප්ප", "ග්‍රෑම්",
    "මේස හැඳි", "මේස හැදි",
    "හැඳි", "හැදි",
    "මීලීලීටර්", "ලීටර්", "කිලෝ"
]
measurement_units = [normalize(unit) for unit in measurement_units]

# Load ingredient list from file
INGREDIENT_LIST_PATH = '/content/drive/MyDrive/FYP Tech Novas/Model 2/ingredientAndTool.txt'
with open(INGREDIENT_LIST_PATH, 'r', encoding='utf-8') as f:
    known_ingredients = [line.strip() for line in f if line.strip()]

def has_intro_keyword(sentence):
    norm_sentence = normalize(sentence)
    return any(kw in norm_sentence for kw in ingredient_intros)

def extract_ingredients(sentence):
    norm_sentence = normalize(sentence)
    return [ing for ing in known_ingredients if normalize(ing) in norm_sentence]

def is_measurement_sentence(sentence, ingredient=None):
    norm_sentence = normalize(sentence)
    tokens = tokenizer.tokenize(sentence)
    tags = pos_tagger.predict([tokens])[0]

    found_unit = any(unit in norm_sentence for unit in measurement_units)
    found_num = any(tag == "NUM" for (_, tag) in tags)
    has_ingredient = ingredient in sentence if ingredient else True
    return (found_num or found_unit) and has_ingredient

def mentions_known_ingredient(sentence):
    norm_sentence = normalize(sentence)
    return any(normalize(ingredient) in norm_sentence for ingredient in known_ingredients)

def is_important_step(sentence):
    tokens = tokenizer.tokenize(sentence)
    tags = pos_tagger.predict([tokens])[0]

    has_verb = any(tag.startswith("V") for _, tag in tags)
    has_number = any(tag == "NUM" for _, tag in tags)
    norm_sentence = normalize(sentence)
    has_measurement = any(unit in norm_sentence for unit in measurement_units)
    has_known_ingr = mentions_known_ingredient(sentence)

    return has_verb or has_number or has_measurement or has_known_ingr

def semantic_similarity(a, b, model=model):
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

def insert_ingredients(sentence, ingredients):
    norm_sentence = normalize(sentence)
    for unit in measurement_units:
        if unit in norm_sentence:
            split_point = norm_sentence.index(unit)
            before = sentence[:split_point]
            after = sentence[split_point:]
            ingr_text = ", ".join(ingredients) + " "
            return before + ingr_text + after
    # fallback if no unit matched
    ingr_text = ", ".join(ingredients) + " "
    return ingr_text + sentence

def summarize_sentences(sentences, sim_threshold=0.8, semantic_threshold=0.98):
    summarized = []
    n = len(sentences)
    skip_next = False

    for i, curr in enumerate(sentences):
        if skip_next:
            skip_next = False
            continue

        prev = sentences[i-1] if i > 0 else ""
        nxt = sentences[i+1] if (i+1) < n else ""

        sim_sem_prev = semantic_similarity(curr, prev, model) if prev else 0.0
        sim_sem_next = semantic_similarity(curr, nxt, model) if nxt else 0.0

        print(f'\nSentence: {curr}')
        print(f'  Semantic similarity with prev: {sim_sem_prev:.2f}')
        print(f'  Semantic similarity with next: {sim_sem_next:.2f}')

        # --- Skip semantic duplicates only if not important ---
        if prev and sim_sem_prev >= semantic_threshold and not is_important_step(curr):
            print('  -> Skipped (semantic duplicate with previous and not important)')
            continue

        # --- Ingredient intro + measurement in next sentence ---
        if nxt and has_intro_keyword(curr):
            curr_ingredients = extract_ingredients(curr)
            next_ingredients = extract_ingredients(nxt)
            norm_curr = normalize(curr)
            curr_has_meas = any(unit in norm_curr for unit in measurement_units)

            if curr_ingredients and not curr_has_meas and is_measurement_sentence(nxt):
                next_ingredient_set = set(next_ingredients)
                curr_ingredient_set = set(curr_ingredients)

                if not next_ingredient_set or curr_ingredient_set == next_ingredient_set:
                    print('  -> Ingredient intro detected + measurement in next sentence.')
                    if not next_ingredient_set:
                        modified_next = insert_ingredients(nxt, curr_ingredients)
                        print(f'     Modified next sentence: {modified_next}')
                        summarized.append(modified_next)
                    else:
                        summarized.append(nxt)
                    skip_next = True
                    continue

        # --- Skip measurement line ONLY if it's not an important step ---
        if prev and has_intro_keyword(prev):
            prev_ingredients = extract_ingredients(prev)
            if prev_ingredients and is_measurement_sentence(curr) and not is_important_step(curr):
                print('  -> Measurement sentence already added with previous ingredient intro, skipping.')
                continue

        summarized.append(curr)
        print('  -> Kept')
    return summarized

# === Run this block to summarize final_instructions.txt ===
INPUT_FILE = "final_classified_instructions.txt"
OUTPUT_FILE = "final_summary.txt"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_sentences = [line.strip() for line in f if line.strip()]

final_sentences = summarize_sentences(raw_sentences)

# Save final summary
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for line in final_sentences:
        f.write(line + '\n')

print(f"\n✅ Final summarized instructions saved to: {OUTPUT_FILE}")

