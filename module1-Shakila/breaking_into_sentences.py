import re
import os
from sinling import SinhalaTokenizer, POSTagger

def preprocess_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as file:
        document = file.read()
        print("\n🔹 Original Input Text:\n")
        print(document)

    tokenizer = SinhalaTokenizer()
    tagger = POSTagger()

    tokenized_sentences = tokenizer.split_sentences(document)
    text = ' '.join(s.strip() for s in tokenized_sentences)

    tagged_sentences = []
    pronouns = []

    for sentence in tokenized_sentences:
        tokens = tokenizer.tokenize(sentence)
        tagged = tagger.predict([tokens])
        tagged_sentences.append(tagged[0])
        for word, tag in tagged[0]:
            if tag == 'PRP':
                pronouns.append(word)

    print("Pronouns (PRP) found in text:")
    print(set(pronouns))

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Step 1: Add full stop after common verbs
    text = re.sub(r'(\S*නවා)(?=\s+(?!(නම්|නං|මේ ටික))[^.])', r'\1. ', text)
    text = re.sub(r'(?<!\S)(බලන්න)(?=\s|$)', r'\1. ', text)
    text = re.sub(r'(?<!\S)(කියල|කියලා)(?=\s|$)', r'\1. ', text)
    text = re.sub(r'(ගමු)(?=\s+(?!මේ ටික)\S|$)', r'\1. ', text)
    text = re.sub(r'(මේ ටික)(?=\s|$)', r'\1. ', text)
    text = re.sub(r'(විදිය\s+බලමු)(?=\s|$)', r'\1. ', text)

    # Step 2: Remove incorrect full stops between phrases
    text = re.sub(r'(පේනවා)\.\s*(ඇති)', r'\1 \2', text)
    text = re.sub(r'(කියල|කියලා)\.\s*(දුන\S*)', r'\1 \2', text)
    text = re.sub(r'(තියෙන්නෙ)\.\s*(ඒකට)', r'\1 \2', text)
    text = re.sub(r'(මොකද)\.\s*', r'\1 ', text)

    # Step 3: Full stop before pronoun + දැන්
    pronoun_group_escaped = '|'.join(re.escape(p) for p in sorted(set(pronouns), key=len, reverse=True))
    pattern_before = rf'(?<![.\n])\s*({pronoun_group_escaped})\s+(දැන්)'
    text = re.sub(pattern_before, r'. \1 \2', text)

    # Step 4: Full stop before "දැන්" if not preceded by a pronoun or keyword
    keywords = ["ඒකට", "ඉතින්", "එහෙනම්", "ඊළඟට", "ඒවගේම", "ඒ වගේම", "එතකොට", "ඊ්ට", "ඒවා", "ඊට පස්සේ"]
    avoid_before_dan = set(pronouns + keywords)
    negative_lookbehinds = ''.join(rf'(?<!{re.escape(w)})' for w in avoid_before_dan)
    pattern_before_dan = rf'(?<!\.){negative_lookbehinds}\s+(දැන්)'
    text = re.sub(pattern_before_dan, r'. \1', text)

    # Step 5: Full stop before listed keywords
    for kw in keywords:
        pattern = rf'(?<!\.)\s*({re.escape(kw)})'
        text = re.sub(pattern, r'. \1', text)

    # Step X: Add full stop after "න්න" cases
    text = re.sub(
        r'(\S*[^ෙ]න්න)(?=\s+(?!(විදිහ|තියන්න|පුළුවන්|බැරි|නම්|ගිය|කලින්))\S)',
        r'\1. ',
        text
    )
    text = re.sub(
        r'(\S*[^ෙ]න්න)\.\s+(ඕනි(?: නෑ)?)',
        r'\1 \2. ',
        text
    )

    # Final steps: Split, filter, and write
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    filtered_sentences = [s for s in sentences if len(s.split()) > 2]
    final_text = '. '.join(filtered_sentences) + '.'

    print("\n🔹 Final Processed Output Text:\n")
    print(final_text)

    with open(output_path, 'w', encoding='utf-8') as out_file:
        out_file.write(final_text)

    print(f"✅ Preprocessing complete. Output saved to: {output_path}")
