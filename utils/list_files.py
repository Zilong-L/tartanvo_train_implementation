import os

def find_files_recursively(folder_path,  pose_output_file):
    with open(pose_output_file, 'w') as pose_f:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                abs_path = os.path.abspath(file_path)
                if file == 'pose_left.txt':
                    pose_f.write(abs_path + '\n')

if __name__ == "__main__":
    folder_path = input("type in the folder path you want to list files:")
    pose_output_file = folder_path+"/pose_left_paths.txt"

    if not os.path.isdir(folder_path):
        print(f"There is no such path: {folder_path}")
    else:
        find_files_recursively(folder_path,  pose_output_file)
        print(f"pose_left.txt is written into {pose_output_file}")
