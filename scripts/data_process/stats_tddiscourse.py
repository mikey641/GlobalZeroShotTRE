def get_all_relations(_tdd_file):
    with open(_tdd_file) as f:
        data = f.readlines()

    all_rels = list()
    for line in data:
        split = line.split('\t')
        file_name = split[0]
        source = split[1].strip()
        target = split[2].strip()
        relation = split[3].strip()
        all_rels.append((file_name, source, target, relation))

    return all_rels


def compute_stats(all_rels):
    # load tab separated file
    before = 0
    after = 0
    equal = 0
    vague = 0
    includes = 0
    is_included = 0
    all_docs = set()
    all_events = set()
    for line in all_rels:
        all_docs.add(line[0])
        relation = line[3]
        all_events.add(f'{line[0]}#{line[1]}')
        all_events.add(f'{line[0]}#{line[2]}')
        if relation == 'b':
            before += 1
        elif relation == 'a':
            after += 1
        elif relation == 's':
            equal += 1
        elif relation == 'v':
            vague += 1
        elif relation == 'i':
            includes += 1
        elif relation == 'ii':
            is_included += 1
        else:
            raise ValueError(f"Unknown relation: {relation}")

    assert len(all_rels) == before + after + equal + vague + includes + is_included
    print(f"Total documents: {len(all_docs)}")
    print(f"Total events: {len(all_events)}")
    print(f"Total relations: {len(all_rels)}")
    print(f"Before: {before}, percentage: {before/len(all_rels)}")
    print(f"After: {after}, percentage: {after/len(all_rels)}")
    print(f"Equal: {equal}, percentage: {equal/len(all_rels)}")
    print(f"Vague: {vague}, percentage: {vague/len(all_rels)}")
    print(f"Includes: {includes}, percentage: {includes/len(all_rels)}")
    print(f"Is Included: {is_included}, percentage: {is_included/len(all_rels)}")


if __name__ == '__main__':
    _tdd_dev_file = "data/TDDiscourse/TDDManDev.tsv"
    _tdd_test_file = "data/TDDiscourse/TDDManTest.tsv"
    _tdd_train_file = "data/TDDiscourse/TDDManTrain.tsv"

    _all_rels = get_all_relations(_tdd_dev_file)
    _all_rels.extend(get_all_relations(_tdd_test_file))
    _all_rels.extend(get_all_relations(_tdd_train_file))

    compute_stats(_all_rels)