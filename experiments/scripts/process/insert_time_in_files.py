import json
import os


def main():
    # iterate over all files in data/time/without_time/
    for src_file in os.listdir('data/time/with_time/group_a'):
        with open(f'data/time/with_time/group_a/{src_file}', 'r') as f:
            data_src = json.load(f)
            file_name = src_file.split('_')[0]

        for subdir, dirs, files in os.walk('data/time/revised/Temporal'):
            for dest_file in files:
                if dest_file.split('_')[0] == file_name:
                    data_dest = json.load(open(os.path.join(subdir, dest_file)))
                    data_dest['main_doc']['time_exprs'] = data_src['main_doc']['time_exprs']
                    with open(os.path.join(subdir, dest_file), 'w') as f:
                        json.dump(data_dest, f, indent=4)


if __name__ == '__main__':
    main()
