import os

def rename_files_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.wav') and '-' in filename:
            new_filename = filename.replace('-', '_')
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            os.rename(old_file, new_file)
            print(f'Renamed: {filename} to {new_filename}')


directory_path = "Sdataset2/all"
rename_files_in_directory(directory_path)