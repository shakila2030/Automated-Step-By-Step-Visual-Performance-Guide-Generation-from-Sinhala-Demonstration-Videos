import re
import unicodedata
from ngram import NGram
import kenlm

# Sinhala grapheme marks
SINGHALA_VOWEL_LETTERS_AND_SIGNS = r'[\u0DCA-\u0DDF]'

def read_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def remove_pillam(word):
    return re.sub(SINGHALA_VOWEL_LETTERS_AND_SIGNS, '', word)

def improved_tokenize(text):
    pattern = r'[\u0D80-\u0DFF\u200d]+|[a-zA-Z]+|\d+|[^\w\s]'
    return re.findall(pattern, text)

def correct_word(word, check_corpus, ngram_index, threshold=0.4):
    if word in check_corpus:
        return word, []
    candidates = ngram_index.search(word, threshold=threshold)
    if candidates:
        best = max(candidates, key=lambda x: x[1])[0]
        return best, candidates
    return word, []

def is_english(word):
    return re.fullmatch(r'[a-zA-Z]+', word) is not None

def correct_misspellings(input_path, output_path, unmatched_path,
                         corpus1_path, corpus2_path, kenlm_model_path):

    sentences = read_lines(input_path)
    vertical_corpus = set(unicodedata.normalize('NFC', line) for line in read_lines(corpus1_path))
    corpus2 = set(unicodedata.normalize('NFC', line) for line in read_lines(corpus2_path))

    combined_corpus = vertical_corpus.union(corpus2)
    ngram_index = NGram(list(vertical_corpus), N=2)
    pillam_to_full = {}

    for word in vertical_corpus:
        pillam_form = remove_pillam(word)
        pillam_to_full.setdefault(pillam_form, []).append(word)

    model = kenlm.Model(kenlm_model_path)

    def sentence_score(sentence):
        return model.score(sentence, bos=True, eos=True)

    def get_candidates_ngram_and_pillam(token, tokens, i):
        candidate, candidates_list = correct_word(token, vertical_corpus, ngram_index)
        ngram_candidates = [c[0] for c in candidates_list] if candidates_list else []

        pillam_form = remove_pillam(token)
        pillam_candidates = pillam_to_full.get(pillam_form, [])

        all_candidates = set(ngram_candidates + pillam_candidates)
        all_candidates.discard(token)

        scored = []
        for cand in all_candidates:
            temp_tokens = tokens[:i] + [cand] + tokens[i+1:]
            temp_sent = ' '.join(temp_tokens)
            score = sentence_score(temp_sent)
            scored.append((cand, score))

        if scored:
            orig_sent = ' '.join(tokens)
            orig_score = sentence_score(orig_sent)
            best_cand, best_score = max(scored, key=lambda x: x[1])
            return best_cand if best_score >= orig_score else f"[[{token}]]" if not is_english(token) else token
        else:
            return f"[[{token}]]" if not is_english(token) else token

    corrected_sentences = []
    unmatched_words = set()

    for sentence in sentences:
        tokens = improved_tokenize(sentence)
        corrected_tokens = []

        for i, token in enumerate(tokens):
            if re.match(r'[\u0D80-\u0DFFa-zA-Z\d]', token):
                if token not in vertical_corpus:
                    best_token = get_candidates_ngram_and_pillam(token, tokens, i)
                    corrected_tokens.append(best_token)
                    if best_token == token or best_token.startswith('[['):
                        unmatched_words.add(token)
                else:
                    corrected_tokens.append(token)
            else:
                corrected_tokens.append(token)

        corrected_sentence = ""
        for j, token in enumerate(corrected_tokens):
            if j > 0:
                if re.match(r'[\u0D80-\u0DFFa-zA-Z\d]', token) or token.startswith('[['):
                    corrected_sentence += ' '
            corrected_sentence += token
        corrected_sentences.append(corrected_sentence.strip())

    # --- Print entire input and output on one line each ---
    print("Input:")
    print(" ".join(sentences))

    print("\nOutput:")
    print(" ".join(corrected_sentences))

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for line in corrected_sentences:
            f_out.write(line + '\n')

    with open(unmatched_path, 'w', encoding='utf-8') as f_unmatched:
        for word in sorted(unmatched_words):
            f_unmatched.write(word + '\n')

    print(f"✅ Corrected transcript saved to '{output_path}'")
    print(f"⚠️ Unmatched words saved to '{unmatched_path}'")
