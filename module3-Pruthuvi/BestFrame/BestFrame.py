import cv2
import numpy as np
import os

def variance_of_laplacian(image):
    # Compute the Laplacian of the image and return the variance for sharpness measure
    return cv2.Laplacian(image, cv2.CV_64F).var()

def get_best_frame(video_path, frame_skip=5):
    print(f"  Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Could not open video {video_path}")
        return None

    best_frame = None
    best_score = 0
    frame_count = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"  Finished reading video. Processed {processed_frames} frames considered for sharpness.")
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = variance_of_laplacian(gray)
        brightness = np.mean(gray)

        # Filter out frames that are too dark or bright
        if brightness < 50:
            print(f"    Skipping frame {frame_count} due to low brightness ({brightness:.2f})")
            frame_count += 1
            continue
        elif brightness > 220:
            print(f"    Skipping frame {frame_count} due to high brightness ({brightness:.2f})")
            frame_count += 1
            continue

        score = sharpness
        print(f"    Frame {frame_count}: sharpness={sharpness:.2f}, brightness={brightness:.2f}")

        if score > best_score:
            best_score = score
            best_frame = frame.copy()
            print(f"      New best frame found at frame {frame_count} with sharpness {sharpness:.2f}")

        frame_count += 1
        processed_frames += 1

    cap.release()
    return best_frame

def process_videos(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
    else:
        print(f"Using existing output directory: {output_folder}")

    video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp4")]
    print(f"Found {len(video_files)} video(s) in {input_folder}")

    for filename in video_files:
        video_path = os.path.join(input_folder, filename)
        print(f"\nProcessing video: {filename}")

        best_frame = get_best_frame(video_path)

        if best_frame is not None:
            output_filename = os.path.splitext(filename)[0] + ".jpg"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, best_frame)
            print(f"Saved best frame as: {output_path}")
        else:
            print(f"No suitable frame found in {filename}")

if __name__ == "__main__":
    root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    best_frame_folder = os.path.abspath(os.path.dirname(__file__))
    input_folder = os.path.join(root_folder, 'MapInstructions/videos_with_sinhala_instructions')
    output_folder = new_video_folder = os.path.join(best_frame_folder, 'best_frames')

    print("Starting the video processing script...")
    process_videos(input_folder, output_folder)
    print("Processing complete.")

