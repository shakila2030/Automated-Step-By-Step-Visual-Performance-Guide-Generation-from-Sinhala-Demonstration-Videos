# step2_merge_words.py

import difflib
import re
import regex  # for Unicode grapheme cluster matching

def merge_words(corpus_path, sentence_path, output_path):
    print("----------------------------------------------------------------------------------------------------------------------------------------")
    print("Merging Words That Have Been Broken During Transcription")
    with open(corpus_path, encoding='utf-8') as f:
        corpus = [line.strip() for line in f if line.strip()]
    corpus_set = set(corpus)

    with open(sentence_path, encoding='utf-8') as f:
        text = f.read().strip()
    print("📝 Input Sentence:\n", text)

    def tokenize(text):
        tokens = []
        for tok in text.split():
            m = re.match(r"^(.+?)([.,?!:;–—…()\[\]{}''\"\"`]+)?$", tok)
            if m:
                word, punct = m.group(1), m.group(2)
                if word: tokens.append(word)
                if punct:
                    for p in punct:
                        tokens.append(p)
            else:
                tokens.append(tok)
        return tokens

    tokens = tokenize(text)
    corrected_tokens = tokens.copy()
    punctuation_set = {".", ",", "?", "!", ":", ";", "–", "—", "…", "(", ")", "''", '""'}

    def is_punctuation(token): return token in punctuation_set
    def get_similarity_score(a, b): return difflib.SequenceMatcher(None, a, b).ratio()
    def grapheme_len(token): return len(regex.findall(r'\X', token))

    i = 0
    while i < len(corrected_tokens) - 1:
        if corrected_tokens[i] == "සුදු" and corrected_tokens[i + 1] == "ළූණු":
            print(f"🔁 Manual merge: සුදු + ළූණු → සුදුළූණු")
            corrected_tokens = corrected_tokens[:i] + ["සුදුළූණු"] + corrected_tokens[i + 2:]
            i = max(i - 1, 0)
        else:
            i += 1

    i = 0
    while i < len(corrected_tokens):
        token = corrected_tokens[i]
        if grapheme_len(token) == 1 and not is_punctuation(token):
            candidates = []
            left_i = i - 1
            right_i = i + 1
            left = corrected_tokens[left_i] if left_i >= 0 else ''
            right = corrected_tokens[right_i] if right_i < len(corrected_tokens) else ''

            def all_in_corpus(tokens_to_check):
                return all(t in corpus_set for t in tokens_to_check if t)

            if left and right and not is_punctuation(left) and not is_punctuation(right):
                if not all_in_corpus([left, token, right]):
                    combo = left + token + right
                    for word in corpus:
                        score = get_similarity_score(combo, word)
                        candidates.append((2, score, word, left_i, right_i))

            if left and not is_punctuation(left):
                if not all_in_corpus([left, token]):
                    combo = left + token
                    for word in corpus:
                        score = get_similarity_score(combo, word)
                        candidates.append((2, score, word, left_i, i))

            if right and not is_punctuation(right):
                if not all_in_corpus([token, right]):
                    combo = token + right
                    for word in corpus:
                        score = get_similarity_score(combo, word)
                        candidates.append((2, score, word, i, right_i))

            if candidates:
                best_priority, best_score, best_word, start_i, end_i = max(candidates, key=lambda x: (x[1]))
                if best_score >= 0.8:
                    merged_phrase = corrected_tokens[start_i:end_i + 1]
                    print(f"🔁 Merged: {' + '.join(merged_phrase)} → {best_word}")
                    before = corrected_tokens[:start_i]
                    if end_i + 1 < len(corrected_tokens) and corrected_tokens[end_i + 1] == '.':
                        corrected_tokens = before + [best_word, '.'] + corrected_tokens[end_i + 2:]
                    else:
                        corrected_tokens = before + [best_word] + corrected_tokens[end_i + 1:]
                    i = max(start_i - 1, 0)
                    continue
        i += 1

    def detokenize(tokens):
        sentence = ''
        for i, token in enumerate(tokens):
            if i == 0:
                sentence = token
            elif is_punctuation(token):
                sentence += token
            else:
                sentence += ' ' + token
        return sentence

    corrected_sentence = detokenize(corrected_tokens)
    print("✅ Corrected Sentence:\n", corrected_sentence)

    with open(output_path, "w", encoding='utf-8') as f:
        f.write(corrected_sentence)

    print(f"✅ Merged output saved to: {output_path}")
