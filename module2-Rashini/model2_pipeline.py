import subprocess

def run_and_print(script_name):
    print(f"\n🚀 Running: {script_name}")
    result = subprocess.run(["python3", script_name], capture_output=True, text=True)
    print("----- STDOUT -----")
    print(result.stdout)
    print("----- STDERR -----")
    print(result.stderr)
    if result.returncode != 0:
        raise Exception(f"❌ {script_name} failed. See error above.")

# Step 1: Filter stopwords
run_and_print('/content/drive/MyDrive/FYP Tech Novas/Model 2/step1_filter_stopwords.py')

# Step 2: Sentence classification
run_and_print('/content/drive/MyDrive/FYP Tech Novas/Model 2/step2_sentence_classification.py')

# Step 3: Redundant Step Merging
run_and_print('/content/drive/MyDrive/FYP Tech Novas/Model 2/step3_redundant_stepMerging.py')

# Step 4: Converting to Sinhala Instructional Written Language
run_and_print('/content/drive/MyDrive/FYP Tech Novas/Model 2/step4_Conversion_Instructional_Language.py')

# Step 5: Segmenting
run_and_print('/content/drive/MyDrive/FYP Tech Novas/Model 2/step5_segment_instructions.py')

# Step 5: Evaluation
run_and_print('/content/drive/MyDrive/FYP Tech Novas/Model 2/model2_evaluation.py')
