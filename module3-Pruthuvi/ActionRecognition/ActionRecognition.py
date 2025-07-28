import cv2
import os
from ultralytics import YOLO
import numpy as np

# Setup paths relative to this script directory
script_dir = os.path.abspath(os.path.dirname(__file__))
root_folder = os.path.abspath(os.path.join(script_dir, '..'))

# Video path and ID
video_filename = "VID_007.mp4"
video_path = os.path.join(root_folder, video_filename)
video_id = os.path.splitext(os.path.basename(video_path))[0]

# Base output folders structured by video ID
base_output_folder = os.path.join(root_folder, "ActionRecognition", "detected_outputs")
video_output_folder = os.path.join(base_output_folder, video_id)
ingredient_image_folder = os.path.join(video_output_folder, "ingredients")
action_clip_folder = os.path.join(video_output_folder, "actions")

# Create output folders if missing
os.makedirs(ingredient_image_folder, exist_ok=True)
os.makedirs(action_clip_folder, exist_ok=True)

# Load YOLO model
model_path = os.path.join(root_folder, "ActionRecognition", "yolov8n.pt")
model = YOLO(model_path)

# Ingredient detection dictionary (ingredient -> set of possible object names)
INGREDIENT_OBJECTS = {
    "carrot": {"carrot", "carrots", "chopped carrot", "sliced carrot", "cut carrot"},
    "potato": {"potato", "potatoes", "sliced potato", "cut potato"},
    "beans": {"green beans", "beans", "cut beans"},
    "broccoli": {"broccoli"},
    "cabbage": {"cabbage", "torn cabbage"},
    "B onion": {"b onion", "onion", "chopped onion"},
    "garlic": {"garlic", "chopped garlic", "sliced garlic"},
    "baby corn": {"baby corn", "corn"},
    "oil": {"oil", "cooking oil", "sesame oil", "gingelly oil"},
    "salt": {"salt"},
    "pepper": {"black pepper", "pepper"},
    "vinegar": {"vinegar"},
    "corn flour": {"corn flour", "cornflour"},
    "water": {"water"},
}

# Actions to alternative sets of object names that trigger the action
ACTION_TOOLS = {
    "cutting": [{"knife"}, {"knife", "cutting board"}],
    "mixing": [{"bowl", "spoon"}, {"bowl", "fork"}, {"bowl", "spatula"}],
    "spreading": [{"knife", "butter"}, {"knife", "mayonnaise"}],
    "boiling": [{"pot", "stove", "water"}],
    "peeling": [{"knife", "peeler"}],
    "soaking": [{"bowl", "water", "salt"}],
    "frying": [{"pan", "oil", "spatula"}],
    "adding": [{"spoon", "hand"}],
    "serving": [{"plate", "fork"}, {"plate", "spoon"}],
}

def normalize_name(name):
    return name.lower().strip()

def detect_ingredients(detected_objects):
    detected_set = set(detected_objects)
    ingredients_found = set()
    # Ingredient is detected only if at least one of its possible object names is detected
    # Optional to require container presence - currently accept ingredient direct detection
    for ingredient, obj_set in INGREDIENT_OBJECTS.items():
        if detected_set.intersection(obj_set):
            ingredients_found.add(ingredient)
    return ingredients_found

def detect_actions(detected_objects, ingredients_present):
    detected_set = set(detected_objects)
    actions_found = set()
    for action, alternative_sets in ACTION_TOOLS.items():
        for obj_set in alternative_sets:
            if obj_set.issubset(detected_set):
                if ingredients_present:
                    actions_found.add(action)
                    break
    return actions_found

def group_consecutive_frames(frame_list, gap=30):
    if not frame_list:
        return []
    frame_list = sorted(frame_list)
    scenes = []
    temp = [frame_list[0]]
    for i in range(1, len(frame_list)):
        if frame_list[i] == frame_list[i-1] + 1:
            temp.append(frame_list[i])
        else:
            scenes.append(temp)
            temp = [frame_list[i]]
    scenes.append(temp)

    # Merge scenes closer than gap frames
    merged = [scenes[0]]
    for i in range(1, len(scenes)):
        prev = merged[-1]
        curr = scenes[i]
        if curr[0] <= prev[-1] + gap:
            merged[-1].extend(range(prev[-1]+1, curr[0]))
            merged[-1].extend(curr)
        else:
            merged.append(curr)
    return merged

def assign_frames_uniquely(action_frames_dict):
    frame_to_actions = {}
    for action, frames in action_frames_dict.items():
        for f in frames:
            frame_to_actions.setdefault(f, []).append(action)

    # Precompute scenes for each action
    action_scenes = {}
    for action, frames in action_frames_dict.items():
        scenes = group_consecutive_frames(frames)
        action_scenes[action] = scenes

    frame_action_assignment = {}

    for frame, actions in frame_to_actions.items():
        if len(actions) == 1:
            frame_action_assignment[frame] = actions[0]
        else:
            # Pick the action whose scene containing the frame is longest
            max_len = -1
            chosen_action = None
            for action in actions:
                for scene in action_scenes[action]:
                    if frame in scene:
                        if len(scene) > max_len:
                            max_len = len(scene)
                            chosen_action = action
                        break
            frame_action_assignment[frame] = chosen_action

    unique_action_frames = {action: [] for action in action_frames_dict.keys()}
    for frame, action in frame_action_assignment.items():
        unique_action_frames[action].append(frame)

    for action in unique_action_frames:
        unique_action_frames[action] = sorted(unique_action_frames[action])

    return unique_action_frames

def export_ingredient_images(cap, ingredient_frames_dict):
    for ingredient, frames in ingredient_frames_dict.items():
        if not frames:
            continue
        median_frame = frames[len(frames) // 2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, median_frame)
        ret, frame = cap.read()
        if ret:
            filename = f"{ingredient.replace(' ', '_')}.png"
            filepath = os.path.join(ingredient_image_folder, filename)
            cv2.imwrite(filepath, frame)
            print(f"Saved ingredient image: {filepath}")

def process_video(video_path):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    frame_idx = 0
    ingredient_frames = {}
    action_frames = {action: [] for action in ACTION_TOOLS.keys()}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results[0]

        detected_classes = [normalize_name(model.names[int(cls)]) for cls in detections.boxes.cls]

        ingredients = detect_ingredients(detected_classes)
        ingredient_frames[frame_idx] = ingredients

        actions = detect_actions(detected_classes, ingredients)
        for action in actions:
            action_frames[action].append(frame_idx)

        frame_idx += 1

    cap.release()

    # Compile ingredient frames dictionary (ingredient -> frames where appeared)
    ingredient_frames_dict = {}
    for ingr in INGREDIENT_OBJECTS.keys():
        ingredient_frames_dict[ingr] = [f for f, ingr_set in ingredient_frames.items() if ingr in ingr_set]

    # Assign unique frames to actions - no overlap
    unique_action_frames = assign_frames_uniquely(action_frames)

    all_ingredients = [ing for ing, frames in ingredient_frames_dict.items() if frames]
    print(f"Ingredients detected: {sorted(all_ingredients)}")
    all_actions = [action for action, frames in unique_action_frames.items() if frames]
    print(f"Actions detected without overlap: {sorted(all_actions)}")

    # Reopen video to export outputs
    cap = cv2.VideoCapture(video_path)

    # Export ingredient images
    export_ingredient_images(cap, ingredient_frames_dict)

    # Collect all scenes for all actions with start/end frames
    all_scenes = []
    for action, frames in unique_action_frames.items():
        if not frames:
            continue
        scenes = group_consecutive_frames(frames, gap=30)
        for scene_num, scene_frames in enumerate(scenes, start=1):
            all_scenes.append((scene_frames[0], scene_frames[-1], action, scene_num))

    # Sort scenes by start frame for global ordering
    all_scenes.sort(key=lambda x: x[0])

    # Export scenes in chronological order with sequential naming: scene_1_action.mp4, ...
    for seq_num, (start_frame, end_frame, action, _) in enumerate(all_scenes, start=1):
        output_name = f"scene_{seq_num}_{action}.mp4"
        output_path = os.path.join(action_clip_folder, output_name)
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        print(f"Exporting: {output_path} [frames {start_frame}-{end_frame}]")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for f in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()
    cap.release()
    cv2.destroyAllWindows()

    print(f"✅ Video segmentation with structured output completed for video {video_id}.")

if __name__ == "__main__":
    process_video(video_path)
