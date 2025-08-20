import os
import shutil
from datetime import datetime

source_dir = fr"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\movies per camera (not arranged)"

movies_dir = r"G:\My Drive\Amitai\one halter experiments\one halter experiments 23-24.1.2024\experiment 24-1-2024 undisturbed\arranged movies"
# define the source folders and the destination folder
cams_folders = ['cam2', 'cam3', 'cam4']
dst_folder = 'new_folder'
format = "%H:%M"
START_UV_TIME = datetime.strptime(f'13:44', format)
# # create the destination folder if it does not exist
# if not os.path.exists(dst_folder):
#     os.mkdir(dst_folder)
char_ = '_'
char_H = 'H'


def find_file_in_list(files_list, substring):
    for file_name in files_list:
        if substring in file_name:
            return file_name
    else:
        return None


def turn_to_hour(substring):
    hour = substring[1:3]
    minute = substring[3:5]
    return hour, minute


def delete_directory(path):
    shutil.rmtree(path, ignore_errors=True)


def arrange_files():
    global date, hour, minute, i, src_path
    src_folder_cam1 = os.path.join(source_dir, 'cam1')
    sparse_cam1 = os.listdir(src_folder_cam1)
    # loop over the source folders
    mov_count = 1
    for mov in sparse_cam1:
        print(mov)
        substring = mov[mov.find(char_H):mov.find(char_)]
        date = mov[(mov.find(char_H) - 3):mov.find(char_H)]
        hour, minute = turn_to_hour(substring)
        format = "%H:%M"
        # t = datetime.strptime(f"{hour}:{minute}", format)
        # uv = False
        # if t > START_UV_TIME:
        #     uv = True
        movie_cam1_path = os.path.join(source_dir, 'cam1', mov)
        movies_paths = [os.path.join(source_dir, 'cam1', mov)]
        for cam in cams_folders:
            movies_list = os.listdir(os.path.join(source_dir, cam))
            mov_twin = find_file_in_list(movies_list, substring)
            movies_paths.append(os.path.join(source_dir, cam, mov_twin))

        new_dir = os.path.join(movies_dir, f'mov{mov_count}')
        readme_path = f'{new_dir}\README_mov{mov_count}.txt'
        copy_movies(hour, minute, mov_count, movies_paths, new_dir, readme_path, substring)
        mov_count += 1


def copy_movies(hour, minute, mov_count, movies_paths, new_dir, readme_path, substring):
    global i, src_path
    if os.path.exists(new_dir):
        delete_directory(new_dir)
    os.mkdir(new_dir)
    fd = os.open(readme_path, os.O_WRONLY | os.O_CREAT)
    with os.fdopen(fd, 'w') as f:
        # Write some text to the file
        f.write(f'time is {hour}:{minute}\n')
        f.write(f'common substring {substring}\n')
    for i, src_path in enumerate(movies_paths):
        new_name = f"mov{mov_count}_cam{i + 1}_sparse.mat"
        shutil.copy(src_path, os.path.join(new_dir, new_name))


# def find_3mov_pairs():
#     sparse_cam1 = os.listdir(os.path.join(source_dir, 'cam1'))
#     sparse_cam2 = os.listdir(os.path.join(source_dir, 'cam2'))
#     sparse_cam3 = os.listdir(os.path.join(source_dir, 'cam3'))
#     sparse_cam4 = os.listdir(os.path.join(source_dir, 'cam4'))
#     mov_count = 1
#     for mov in sparse_cam2:
#         substring = mov[mov.find(char_H):mov.find(char_)]
#         pair1 = find_file_in_list(sparse_cam1, substring)
#         pair2 = find_file_in_list(sparse_cam2, substring)
#         pair3 = find_file_in_list(sparse_cam3, substring)
#         pair4 = find_file_in_list(sparse_cam4, substring)
#         if pair1 == None and pair2 is not None and pair2 is not None and pair3 is not None and pair4 is not None:
#             pair2 = os.path.join(source_dir, "cam2", pair2)
#             pair3 = os.path.join(source_dir, "cam3", pair3)
#             pair4 = os.path.join(source_dir, "cam4", pair4)
#             cams_234 = [pair2, pair3, pair4]
#             new_dir = os.path.join(movies_dir, 'movies + README', "3 cameras", f'mov{mov_count}')
#             readme_path = f'{new_dir}\README_mov{mov_count}.txt'
#             hour, minute = turn_to_hour(substring)
#             format = "%H:%M"
#             t = datetime.strptime(f"{hour}:{minute}", format)
#             uv = False
#             if t > START_UV_TIME:
#                 uv = True
#             copy_movies(hour, minute, mov_count, cams_234, new_dir, readme_path, substring, uv)
#             mov_count += 1





if __name__ == "__main__":
    arrange_files()
    # find_3mov_pairs()