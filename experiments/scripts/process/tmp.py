import os

group = 'group_c'
all_files = list()
for src_file in os.listdir(f'data/time/with_time/{group}'):
    file_name = src_file.split('_')[0]
    all_files.append(file_name)

print(sorted(all_files))
