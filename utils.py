import shutil
import os

def move_directory(source_dirs, destination_dir):
    for source_dir in source_dirs:
        dest_path = os.path.join(destination_dir, os.path.basename(source_dir))
        shutil.copytree(source_dir, dest_path)
    print("Directories moved successfully.")

# Example usage
source_dirs2 = [
    "/content/results/checkpoint-25",
    "/content/results/checkpoint-500",
    "/content/results/checkpoint-1000",
    "/content/results/checkpoint-1500",
    "/content/results/checkpoint-2000",
    "/content/results/checkpoint-2500",
    "/content/results/checkpoint-2775",
    "/content/results/runs"
]
destination_dir2 = "/content/drive/MyDrive/QLoRA-Llama2-dailydialog/results-2"
move_directory(source_dirs2, destination_dir2)
