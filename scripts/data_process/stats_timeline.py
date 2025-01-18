def get_all_relations(timeline_file):
    with open(timeline_file) as f:
        data = f.readlines()

    all_rels = list()
    for idx, line in enumerate(data):
        if idx == 0:
            # Skipping the first line of the file (header)
            continue

        split = line.split(',')
        file_name = split[0]
        source = split[1].strip()
        target = split[2].strip()
        source_id = split[3].strip()
        target_id = split[4].strip()
        relation = split[5].strip()
        all_rels.append((file_name, source, target, source_id, target_id, relation))

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
    events_per_doc = dict()
    relations_per_doc = dict()
    for line in all_rels:
        all_docs.add(line[0])

        if line[0] not in events_per_doc:
            events_per_doc[line[0]] = set()
            relations_per_doc[line[0]] = list()

        events_per_doc[line[0]].add(line[3])
        events_per_doc[line[0]].add(line[4])
        relations_per_doc[line[0]].append(line)

        relation = line[5]
        all_events.add(f'{line[0]}#{line[3]}')
        all_events.add(f'{line[0]}#{line[4]}')
        if relation == 'b':
            before += 1
        elif relation == 'a':
            after += 1
        elif relation == 'e':
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

    for doc, rels in relations_per_doc.items():
        if (len(events_per_doc[doc])**2 - len(events_per_doc[doc])) / 2 != len(rels):
            print(f"Doc: {doc}, events: {len(events_per_doc[doc])}, relations: {len(rels)}")

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
    _all_timeline_rels = "data/TimeLine/Annotated_Relations.csv"

    _all_rels = get_all_relations(_all_timeline_rels)

    compute_stats(_all_rels)
