import os
import cv2
from ultralytics import YOLO
import shutil
from collections import Counter

# Setup folder paths (modify as needed)
script_dir = os.path.abspath(os.path.dirname(__file__))
root_folder = os.path.abspath(os.path.join(script_dir, '..'))
input_folder = os.path.join(root_folder, 'ActionRecognition', 'detected_outputs', 'VID_007', 'actions')
output_folder = os.path.join(script_dir, 'videos_with_instructions')
os.makedirs(output_folder, exist_ok=True)

# Load YOLOv8 model
model_path = os.path.join(root_folder, 'ActionRecognition', 'yolov8n.pt')
print(f"Loading YOLOv8 model from: {model_path}")
model = YOLO(model_path)

ingredients = [
    "potato", "potatoes", "beans", "broccoli", "cabbage", "onion", "garlic",
    "chili", "butter", "oil", "salt", "pepper", "corn flour", "water",
    "orange", "banana", "apple", "carrot"
]

def sanitize_filename(filename):
    base, ext = os.path.splitext(filename)
    base = base.replace(" ", "_").replace(",", "")
    return base + ext

def extract_action_from_filename(filename):
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    action = parts[-1] if parts else "Do"
    action = action.capitalize()
    print(f"Extracted action '{action}' from filename '{filename}'")
    return action

# ----------- MANUAL RENAME DICTIONARY (update as needed) -----------
manual_rename_dict = {
    "scene_4_cutting.mp4": "cutting carrots.mp4",
    "scene_5_cutting.mp4": "cutting tuber.mp4",
    "scene_7_cutting.mp4": "cutting bean.mp4",
    "scene_8_cutting.mp4": "cutting onions.mp4",
    "scene_11_cutting.mp4": "cutting garlic.mp4",
    "scene_12_cutting.mp4": "frying gralic.mp4",
    "scene_13_mixing.mp4": "mixing onions.mp4",
    "scene_14_mixing.mp4": "mixing carrots.mp4",
    "scene_20_mixing.mp4": "mixing cornflour.mp4",
    "scene_24_mixing.mp4": "mixing food.mp4",
    "scene_25_cutting.mp4": "serving food.mp4"
}

def manual_rename(input_folder, output_folder, rename_dict):
    print("Running RENAMING routine...")
    for video_file in os.listdir(input_folder):
        if video_file in rename_dict:
            src = os.path.join(input_folder, video_file)
            dst = os.path.join(output_folder, sanitize_filename(rename_dict[video_file]))
            shutil.copy(src, dst)
            print(f"Renamed: {video_file} -> {sanitize_filename(rename_dict[video_file])}")
        else:
            print(f"Skipped (no mapping): {video_file}")
    print("✅ All mapped videos processed.\n")

print(f"Processing videos in folder: {input_folder}")

# ------------ TOGGLE BETWEEN MANUAL AND AUTO MODE -------------
# Uncomment ONE of the next two blocks as needed

# ========== MANUAL RENAME MODE (uncomment to use manual rename) ==========

manual_rename(input_folder, output_folder, manual_rename_dict)

# ========== AUTOMATIC INSTRUCTION MODE (comment out if testing manual rename) ==========

# for video_file in os.listdir(input_folder):
#     video_path = os.path.join(input_folder, video_file)
#     if not os.path.isfile(video_path):
#         print(f"Skipping '{video_file}': not a file")
#         continue
#
#     print(f"\nOpening video file: {video_file}")
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"ERROR: Cannot open video file {video_file}, skipping...")
#         continue
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f"Total frames: {frame_count}")
#
#     detected_ingredients = []
#     frame_idx = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         results = model(frame)
#         for r in results:
#             for c in r.boxes.cls:
#                 obj_name = model.names[int(c)].lower()
#                 if obj_name in ingredients:
#                     detected_ingredients.append(obj_name)
#                     print(f"Detected ingredient '{obj_name}' at frame {frame_idx}")
#         frame_idx += 1
#
#     cap.release()
#
#     if detected_ingredients:
#         most_common_ingredient, count = Counter(detected_ingredients).most_common(1)[0]
#         print(f"Most frequent ingredient in '{video_file}': '{most_common_ingredient}' ({count} detections)")
#     else:
#         most_common_ingredient = "food"
#         print(f"No ingredients detected in '{video_file}'. Defaulting ingredient to '{most_common_ingredient}'")
#
#     action = extract_action_from_filename(video_file)
#     new_filename = f"{action} {most_common_ingredient}.mp4"
#     new_filename = sanitize_filename(new_filename)
#     new_path = os.path.join(output_folder, new_filename)
#     shutil.copy(video_path, new_path)
#     print(f"Copied and renamed '{video_file}' to '{new_filename}'")
#
# print(f"\n✅ All videos processed. Output saved in '{output_folder}' folder.")

# ------------- END OF TOGGLE --------------

