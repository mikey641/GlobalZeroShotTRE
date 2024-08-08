import os


def rename_files_in_folder():
    # List all files in the folder
    files = os.listdir(folder_path)

    prefix = 1
    for file_name in files:
        # Construct the new file name
        new_file_name = str(prefix) + "_final.json"
        prefix += 1

        # Get the full path of the old and new file
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_file_name)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f'Renamed: {old_file_path} -> {new_file_path}')


# Example usage
folder_path = 'data/my_data/EventFullTrainExports'
rename_files_in_folder()
