# 2. LOAD INGREDIENT LIST FROM DRIVE
INGREDIENT_PATH = '/content/drive/MyDrive/FYP Tech Novas/Model 2/ingredientAndTool.txt'  # Update path if needed
with open(INGREDIENT_PATH, 'r', encoding='utf-8') as f:
    ingredients = [line.strip() for line in f if line.strip()]
print("Ingredients:", ingredients)


# 4. SPLIT SCRIPT INTO SENTENCES
import re

def split_sinhala_sentences(text):
    # Splits on Sinhala full stop (.) and danda (।), trimming whitespace
    return [s.strip() for s in re.split(r'[.।]+', text) if s.strip()]

with open(f'/content/formalized_summary.txt', 'r', encoding='utf-8') as f:
    script_text = f.read()

sentences = split_sinhala_sentences(script_text)
print(f"Total sentences found: {len(sentences)}")


from sinling import SinhalaTokenizer

tokenizer = SinhalaTokenizer()

# 6. DEFINE COMMON SUFFIXES AND GENERATE VARIANTS

common_suffixes = ['', 'ව', 'වට', 'වේ', 'ක්', 'වක්', 'යක්', 'යට', 'ට', 'ෙ', 'ගෙ']

def generate_variants(word):
    variants = set()
    for suffix in common_suffixes:
        variants.add(word + suffix)
    return variants

# 7. PREPARE VARIANTS DICT FOR INGREDIENTS (no stemming, just tokenizing)

ingredient_variants = dict()  # original ingredient -> set of variant token word strings

for ing in ingredients:
    variants = generate_variants(ing)
    variant_token_forms = set()

    for var in variants:
        tokens = tokenizer.tokenize(var)
        words = []
        for token in tokens:
            if isinstance(token, tuple):
                w = token[0]
            else:
                w = token
            words.append(w)
        variant_token_forms.add(' '.join(words))

    ingredient_variants[ing] = variant_token_forms

print("Ingredient variants prepared.")
print(ingredient_variants)

# 8. FUNCTION TO TOKENIZE SENTENCE INTO WORDS (no stemming)

def get_token_words(text):
    tokens = tokenizer.tokenize(text)
    words = []
    for token in tokens:
        if isinstance(token, tuple):
            words.append(token[0])
        else:
            words.append(token)
    return words

# 9. FUNCTION TO MATCH INGREDIENTS IN SENTENCE

def match_ingredients(sentence):
    sentence_words = get_token_words(sentence)
    matched = []
    for original_ing, variant_set in ingredient_variants.items():
        for variant in variant_set:
            variant_words = variant.split()
            if all(word in sentence_words for word in variant_words):
                matched.append(original_ing)
                break
    return matched

# 10. SEGMENTATION LOGIC INCLUDING ALL SENTENCES:
# Group sentences if their ingredient sets are exactly the same and length <= 3,
# otherwise standalone segments for sentences with no matched ingredients

segments = []
i = 0
n = len(sentences)

while i < n:
    curr_matched = match_ingredients(sentences[i])
    if curr_matched:
        # Try to form a group with adjacent sentences matched to same ingredients,
        # max 3 sentences total
        group = [sentences[i]]
        indices = [i]

        # Try to include previous sentence if exactly same matched ingredients
        prev_i = i - 1
        if prev_i >= 0:
            prev_matched = match_ingredients(sentences[prev_i])
            if set(prev_matched) == set(curr_matched) and prev_matched:
                group.insert(0, sentences[prev_i])
                indices.insert(0, prev_i)

        # Try to include next sentences with exactly same ingredients, up to max group size 3
        next_i = i + 1
        while next_i < n and len(group) < 3:
            next_matched = match_ingredients(sentences[next_i])
            if set(next_matched) == set(curr_matched) and next_matched:
                group.append(sentences[next_i])
                indices.append(next_i)
                next_i += 1
            else:
                break

        # Join sentences with Sinhala full stop and space + final full stop
        combined_segment = '. '.join(group) + '.'
        segments.append({
            'text': combined_segment,
            'start_idx': indices[0],
            'end_idx': indices[-1],
            'ingredients': curr_matched
        })

        i = max(indices) + 1

    else:
        # Sentence with no ingredients → treat as its own segment
        combined_segment = sentences[i].strip() + '.'
        segments.append({
            'text': combined_segment,
            'start_idx': i,
            'end_idx': i,
            'ingredients': []
        })
        i += 1

# 11. PRINT SEGMENTED STEPS WITH NUMBERING (in script order)
# Specify output file path (save in /content or Drive folder)
output_filepath = '/content/drive/MyDrive/FYP Tech Novas/Model 3/final_instructions.txt'
output_filepath2 = '/content/final_instructions.txt'

with open(output_filepath, 'w', encoding='utf-8') as f:
    for idx, seg in enumerate(segments, 1):
        f.write(f"{idx}. {seg['text']}\n\n")

with open(output_filepath2, 'w', encoding='utf-8') as f:
    for idx, seg in enumerate(segments, 1):
        f.write(f"{idx}. {seg['text']}\n\n")

print(f"Segmented steps saved to: {output_filepath2}")

print("\nSegmented Steps (All sentences included):\n")
for idx, seg in enumerate(segments, 1):
    print(f"{idx}. {seg['text']}\n")
