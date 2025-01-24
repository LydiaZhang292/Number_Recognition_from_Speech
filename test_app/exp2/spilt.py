# 切分数据集

import os
import shutil

def organize_audio_files(source_directory, destination_directory):
    # Ensure the destination directory exists
    os.makedirs(destination_directory, exist_ok=True)

    # Traverse the source directory
    for root, _, files in os.walk(source_directory):
        for file in files:
            if file.endswith('.wav'):
                # Parse the filename
                parts = file.split('_')
                if len(parts) == 3 and parts[2].endswith('.wav'):
                    label = parts[0].strip()  # Remove any leading/trailing spaces
                    num = parts[1].strip()
                    user_id = parts[2].replace('.wav', '').strip()

                    # Construct the new directory path
                    user_dir = os.path.join(destination_directory, user_id)
                    num_dir = os.path.join(user_dir, num)

                    # Ensure the directories exist
                    os.makedirs(num_dir, exist_ok=True)

                    # Source file path
                    source_file_path = os.path.join(root, file)
                    
                    # Destination file path
                    destination_file_path = os.path.join(num_dir, file)

                    # Debugging: Print paths to verify
                    print(f"Moving '{source_file_path}' to '{destination_file_path}'")

                    # Move or copy the file
                    shutil.copy(source_file_path, destination_file_path)
                    # If you prefer to copy instead of move, use shutil.copy()

    print("Files have been organized successfully.")

# Example usage
source_directory = r"Sdataset2\\all"
destination_directory = r'dtw\\all'
organize_audio_files(source_directory, destination_directory)
