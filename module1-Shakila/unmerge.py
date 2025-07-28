# step3_fix_incorrect_merges.py

import grapheme

def load_corpus(corpus_path):
    with open(corpus_path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def load_lines(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def correct_merged_words(sentence, corpus):
    words = sentence.strip().split()
    corrected_words = []
    changes = []

    for word in words:
        if word in corpus:
            corrected_words.append(word)
            continue

        split_done = False
        g_list = list(grapheme.graphemes(word))

        for i in range(2, len(g_list)):
            left = ''.join(g_list[:i])
            right = ''.join(g_list[i:])
            if len(list(grapheme.graphemes(left))) <= 1 or len(list(grapheme.graphemes(right))) <= 1:
                continue
            if left in corpus and right in corpus:
                corrected_words.extend([left, right])
                changes.append((word, f"{left} {right}"))
                split_done = True
                break

        if not split_done:
            corrected_words.append(word)

    return " ".join(corrected_words), changes

def fix_incorrect_merges(corpus_path, input_path, output_path):
    print("--------------------------------------------------------------------------------------------------------------------------------------------------")
    print("Separaing Words That Have Been Incorrectly Merged")
    corpus = load_corpus(corpus_path)
    lines = load_lines(input_path)

    print("📄 Input Text File Content:")
    for line in lines:
        print(line)

    output_lines = []
    all_changes = []

    for line in lines:
        corrected_line, changes = correct_merged_words(line, corpus)
        output_lines.append(corrected_line)
        all_changes.extend(changes)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    if all_changes:
        print("Changed words:")
        for old, new in all_changes:
            print(f"  {old} → {new}")
    else:
        print("No words changed.")

    # ✅ Print final output file content
    print(f"\n✅ Final Output Text from '{output_path}':")
    with open(output_path, 'r', encoding='utf-8') as f:
        print(f.read())
