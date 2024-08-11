import json
import os


def main():
    all_mentions = list()
    all_annot_made = list()
    file_count = 0
    for file in os.listdir('data/my_data/__annotation_count__/'):
        with open(f'data/my_data/__annotation_count__/{file}', 'r') as f:
            data = json.load(f)
            all_mentions.append(len(data['_mainAxis']['_eventIds']))
            all_annot_made.append(data['_tempAnnotationMade'])
            file_count += 1

    print(f"Total files: {file_count}")
    print(f"Total mentions: {sum(all_mentions)}")
    print(f"Avg mentions per file: {sum(all_mentions) / file_count}")
    print(f"Total annotations made: {sum(all_annot_made)}")
    print(f"Avg annotations made per file: {sum(all_annot_made) / file_count}")


if __name__ == '__main__':
    main()
