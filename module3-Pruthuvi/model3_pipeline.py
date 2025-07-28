import subprocess
import sys
import os

def run_script(script_path):
    print(f"Starting {os.path.basename(script_path)}...")
    script_dir = os.path.dirname(script_path)
    try:
        subprocess.run([sys.executable, script_path], check=True, cwd=script_dir)
        print(f"{os.path.basename(script_path)} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error: {os.path.basename(script_path)} failed with return code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    root_folder = r"C:\Users\Pruthuvi\Desktop\Test"

    scripts = [
        os.path.join(root_folder, "ActionRecognition", "ActionRecognition.py"),
        os.path.join(root_folder, "GenerateInstruction", "GenerateInstruction.py"),
        os.path.join(root_folder, "MapInstructions", "MapInstructions.py"),
        os.path.join(root_folder, "BestFrame", "BestFrame.py"),
        os.path.join(root_folder, "PDFGeneration", "PDFGeneration.py"),
    ]

    print("Running Model 3 workflow sequentially...\n")

    for script in scripts:
        run_script(script)

    print("Model 3 workflow completed successfully!")
