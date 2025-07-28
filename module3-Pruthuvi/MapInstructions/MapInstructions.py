import os
import shutil
import re
from googletrans import Translator
from sentence_transformers import SentenceTransformer, util
import torch

# Paths
root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
map_instructions_folder = os.path.abspath(os.path.dirname(__file__))
original_video_folder = os.path.join(root_folder,'GenerateInstruction/videos_with_instructions')
new_video_folder = os.path.join(map_instructions_folder, 'videos_with_sinhala_instructions')
instruction_file_path = os.path.join(root_folder, 'final_instructions.txt')

sim_threshold = 0.35

print("Step 1: Creating new folder if it doesn't exist...")
os.makedirs(new_video_folder, exist_ok=True)
print(f"Folder '{new_video_folder}' ready.\n")

# Step 2: Read Sinhala instructions and extract indexes and texts
print("Step 2: Reading and extracting Sinhala instructions from file...")
pattern = re.compile(r'^(\d+)\.\s*(.*)$')
sinhala_segments = []
with open(instruction_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        match = pattern.match(line.strip())
        if match:
            index = match.group(1)
            text = match.group(2)
            sinhala_segments.append((index, text))
print(f"Found {len(sinhala_segments)} numbered Sinhala instruction segments.\n")

# Step 3: Translate Sinhala segments to English
print("Step 3: Translating Sinhala instructions to English...")
translator = Translator()
translated_segments = []
for idx, text in sinhala_segments:
    if text:
        translated_text = translator.translate(text, src='si', dest='en').text
        translated_segments.append((idx, text, translated_text))
        print(f"Translated [{idx}]: {translated_text}")
print(f"Completed translating {len(translated_segments)} segments.\n")

# Step 4: Load video files and prepare English video names for matching
print("Step 4: Loading video files and preparing for matching...")
video_files = [f for f in os.listdir(original_video_folder) if f.endswith('.mp4')]
print(f"Found {len(video_files)} video files in '{original_video_folder}'.")
video_names = [os.path.splitext(f)[0].replace('_', ' ') for f in video_files]

print("Loading sentence transformer model...")
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
print("Encoding video file names...")
video_embeddings = model.encode(video_names, convert_to_tensor=True)
print("Video embeddings created.\n")

# Step 5: Map each translated English instruction to at most one matching video filename, with threshold
print("Step 5: Mapping instructions to video files using semantic similarity with threshold...")

# Encode all translated instructions at once for efficiency
instruction_texts = [eng_text for _, _, eng_text in translated_segments]
instruction_embeddings = model.encode(instruction_texts, convert_to_tensor=True)

# Compute cosine similarity matrix: shape (num_instructions x num_videos)
cosine_scores = util.pytorch_cos_sim(instruction_embeddings, video_embeddings)

# Track which videos have been assigned to avoid duplicates
assigned_video_indices = set()
mapped_results = []

for i, (idx, sin_text, eng_text) in enumerate(translated_segments):
    scores = cosine_scores[i]
    # Sort video indices by descending similarity score
    sorted_vid_indices = torch.argsort(scores, descending=True)
    assigned_video = None
    assigned_score = None
    for vid_idx in sorted_vid_indices:
        score = scores[vid_idx].item()
        if score >= sim_threshold and vid_idx.item() not in assigned_video_indices:
            assigned_video = vid_idx.item()
            assigned_score = score
            assigned_video_indices.add(assigned_video)
            break
    if assigned_video is not None:
        matched_video_file = video_files[assigned_video]
        mapped_results.append((idx, sin_text, eng_text, matched_video_file, assigned_score))
        print(f"Instruction {idx} mapped to video '{matched_video_file}' with similarity {assigned_score:.3f}")
    else:
        print(f"Instruction {idx} has no matching video above threshold {sim_threshold}.")

print(f"\nMapping completed for {len(mapped_results)} instructions with valid video matches.\n")

# Step 6: Copy and rename matched videos to the new folder using the Sinhala index
print(f"Step 6: Copying and renaming matched videos to '{new_video_folder}'...")
for idx, sin_text, eng_text, video_file, score in mapped_results:
    src_path = os.path.join(original_video_folder, video_file)
    new_file_name = f"{idx}.mp4"
    dst_path = os.path.join(new_video_folder, new_file_name)
    print(f"Copying and renaming '{video_file}' as '{new_file_name}' (similarity: {score:.3f})...")
    shutil.copy(src_path, dst_path)  # Replace with shutil.move() if you want to move files
print("All matched videos copied and renamed.\n")

# (Optional) Print full mapping summary for reference
print("Summary of mappings:")
for idx, sin_text, eng_text, video_file, score in mapped_results:
    print(f"{idx}: Sinhala='{sin_text}' | English='{eng_text}' | VideoFile='{video_file}' | Similarity={score:.3f}")

print("\nProcess completed successfully.")
