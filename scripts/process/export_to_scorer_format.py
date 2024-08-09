import json


def create_report(annot_mentions, annot_clusters):
    cluster_ids = dict()
    cluster_running_ids = 1
    scorer_lines = []
    scorer_lines.append("#begin document (ECB+/ecbplus_all); part 000")
    for ment in annot_mentions:
        if ment['axisType'] == 'main':
            ment_id = ment['m_id']
            found = False
            for cluster in annot_clusters:
                if found:
                    break

                if ment_id in cluster['cluster']:
                    if cluster['clusterId'] not in cluster_ids:
                        cluster_ids[cluster['clusterId']] = 1
                        cluster_running_ids += 1
                    scorer_lines.append(f'ECB+/ecbplus_all\t({cluster_ids[cluster["clusterId"]]})\n')
                    found = True

            if not found:
                scorer_lines.append(f'ECB+/ecbplus_all\t({cluster_running_ids})\n')
                cluster_running_ids += 1

    scorer_lines.append("#end document\n")
    return scorer_lines


def main(annot_file):
    with open(annot_file) as f:
        data1 = json.load(f)

    annot_mentions = data1['allMentions']

    annot_clusters = data1['corefClusters']

    scorer_lines1 = create_report(annot_mentions, annot_clusters)

    with open(output_file1, 'w') as f:
        f.writelines(scorer_lines1)


if __name__ == '__main__':
    annotator = 'alon'
    file_group = '135d4'
    output_file1 = f'data/my_data/output/coref/{file_group}_coref_{annotator}.txt'

    _annot1_file = f'data/my_data/coref_cause/{annotator}/{file_group}_{annotator}_FinalAnnotations.json'
    main(_annot1_file)

