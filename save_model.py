import shutil
import os

def save_model(source_dir, destination_dir):
    # Ensure target directory exists
    os.makedirs(destination_dir, exist_ok=True)
    dest_path = os.path.join(destination_dir, os.path.basename(source_dir))
    shutil.copytree(source_dir, dest_path)
    print(f"Model saved to {destination_dir}")

# Example usage
source_dirs1 = "/content/llama-2-7b-QLoRA-emotion"
destination_dir1 = "/content/drive/MyDrive/QLoRA-Llama2-dailydialog"
save_model(source_dirs1, destination_dir1)
