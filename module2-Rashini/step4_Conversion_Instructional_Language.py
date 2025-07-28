# --- Helper Functions ---
def join_tokens_smart(tokens):
    output = ""
    for token in tokens:
        if token in ['.', '!', '?', '।', ',', ':', ';']:
            output = output.rstrip() + token  # remove any trailing space and add punctuation
        else:
            output += " " + token
    return output.strip()

def load_mapping(file_path):
    mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '-' in line:
                informal, formal = [x.strip() for x in line.split('-', 1)]
                mapping[informal] = formal
    return mapping

def replace_phrases(tokens, phrase_map):
    max_phrase_len = max(len(key.split()) for key in phrase_map.keys()) if phrase_map else 0
    i = 0
    new_tokens = []
    n = len(tokens)

    while i < n:
        replaced = False
        for phrase_len in range(max_phrase_len, 0, -1):
            if i + phrase_len <= n:
                candidate = " ".join(tokens[i:i+phrase_len])
                if candidate in phrase_map:
                    new_phrase = phrase_map[candidate].split()
                    new_tokens.extend(new_phrase)
                    i += phrase_len
                    replaced = True
                    break
        if not replaced:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

from sinling import POSTagger
from indicnlp.tokenize import indic_tokenize

def formalize_sentence(
        sentence,
        known_last_verb,
        known_other_verbs,
        model=None,
        tokenizer=None,
        MBART_SINHA_CODE="si_LK",
        device="cpu"
    ):
    tagger = POSTagger()
    used_model = False
    tokens = indic_tokenize.trivial_tokenize(sentence, lang='si')
    changed_last = set()
    changed_other = set()

    last_real_token_idx = len(tokens) - 1
    while last_real_token_idx >= 0 and tokens[last_real_token_idx] in {'.', '!', '?', '।'}:
        last_real_token_idx -= 1

    if last_real_token_idx >= 0:
        last_token = tokens[last_real_token_idx]
        if last_token in known_last_verb:
            tokens[last_real_token_idx] = known_last_verb[last_token]
            changed_last.add(last_real_token_idx)
        else:
            used_model = True
            input_text = f"<{MBART_SINHA_CODE}> {sentence}"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
            output_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[MBART_SINHA_CODE],
                max_length=512,
                num_beams=5,
                early_stopping=True,
            )
            pred = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            tokens = indic_tokenize.trivial_tokenize(pred, lang='si')
            changed_last = set()
            changed_other = set()

    for i, tok in enumerate(tokens):
        if i in changed_last:
            continue
        if tok in known_other_verbs:
            tokens = replace_phrases(tokens, known_other_verbs)
            changed_other.add(i)

    tagged_tokens = tagger.predict([tokens])[0]
    for i, (tok, tag) in enumerate(tagged_tokens):
        if i in changed_last or i in changed_other:
            continue
        if tag.upper().startswith('V') and tok.endswith("ත්ත"):
            tokens[i] = tok[:-3] + "ත්"

    return join_tokens_smart(tokens), used_model



# --- Load Model ---
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

mbart_model_dir = '/content/drive/MyDrive/FYP Tech Novas/Model 2/instructional_language_model'
tokenizer = AutoTokenizer.from_pretrained(mbart_model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(mbart_model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# --- Load Lexical Resources ---
last_verb_path = '/content/drive/MyDrive/FYP Tech Novas/Model 2/known_last_verbs.txt'
other_verb_path = '/content/drive/MyDrive/FYP Tech Novas/Model 2/known_other_verbs.txt'
known_last_verb = load_mapping(last_verb_path)
known_other_verbs = load_mapping(other_verb_path)

# --- Process Input File ---
input_path = '/content/final_summary.txt'
output_path = '/content/formalized_summary.txt'

with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        informal = line.strip()
        if not informal:
            continue
        formatted, _ = formalize_sentence(
            informal,
            known_last_verb=known_last_verb,
            known_other_verbs=known_other_verbs,
            model=model,
            tokenizer=tokenizer,
            MBART_SINHA_CODE="si_LK",
            device=device
        )
        outfile.write(formatted + '\n')
        print(f"IN : {informal}")
        print(f"OUT: {formatted}")
        print('---')
