import random

from scripts.utils.io_utils import open_input_file


def mark_events_in_text(tokens, all_mentions):
    for mention in all_mentions:
        tok_first_id = mention['tokens_ids'][0]
        tok_last_id = mention['tokens_ids'][-1]
        tokens[tok_first_id] = f'<{tokens[tok_first_id]}'
        tokens[tok_last_id] = f'{tokens[tok_last_id]}({mention["m_id"]})>'
    return " ".join(tokens)


def filter_non_events(events):
    return [e for e in events if 'axisType' in e and e['axisType'] == 'main']


def get_input_text(data):
    if data is not None:
        tokens = data['tokens']
        # all_mentions = filter_non_events(data['allMentions'])
        all_mentions = data['allMentions']
        all_mentions.sort(key=lambda x: x['tokens_ids'][0])
        all_pairs = data['allPairs']
        text = mark_events_in_text(tokens, all_mentions)
        return text, all_pairs


def get_example(file_to_use, target, reduction):
    data = open_input_file(file_to_use)
    intput_text, all_pairs = get_input_text(data)
    if reduction > 0:
        split_out = target.split('\n')
        target_pref = split_out[0:2]
        target_suffix = split_out[-2:]
        new_target = split_out[2:-3]
        sample_size = max(1, int(len(new_target) * reduction))
        indices_to_remove = random.sample(range(len(new_target)), sample_size)
        output_example = [new_target[i] for i in range(len(new_target)) if i not in indices_to_remove]
        # output_example = random.sample(new_target, sample_size)
        output_example = target_pref + output_example + target_suffix
        output_example = '\n'.join(output_example)
    else:
        output_example = target
    return intput_text, output_example


def get_reverse_label(label):
    if label == 'before':
        return 'after'
    elif label == 'after':
        return 'before'
    elif label == 'is_included':
        return 'includes'
    elif label == 'includes':
        return 'is_included'
    else:
        return label

def arrange_pairs(all_pairs, ment_dict):
    for pair in all_pairs:
        m1 = ment_dict[pair['_firstId']]
        m2 = ment_dict[pair['_secondId']]
        if m1['tokens_ids'][0] > m2['tokens_ids'][0]:
            pair['_firstId'], pair['_secondId'] = pair['_secondId'], pair['_firstId']
            pair['_relation'] = get_reverse_label(pair['_relation'])


def _capsule_order_pairs(all_pairs, ment_dict, capsule_size):
    """Return pairs in capsule order.

    1. Sort events by document position.
    2. Split into overlapping capsules of size N with overlap of 1 event.
    3. Within each capsule, emit pairs by intra-capsule distance level (1, 2, ...).
    4. Append remaining cross-capsule pairs ordered by token distance.
    5. Deduplicate — each pair appears exactly once.
    """
    # All unique event ids, sorted by token position
    all_event_ids = list({p['_firstId'] for p in all_pairs} | {p['_secondId'] for p in all_pairs})
    events_sorted = sorted(all_event_ids, key=lambda x: ment_dict[x]['tokens_ids'][0])
    n = len(events_sorted)

    # Build capsules with overlap of 1
    capsules = []
    i = 0
    while i < n:
        capsule = events_sorted[i: i + capsule_size]
        capsules.append(capsule)
        if i + capsule_size >= n:
            break
        i += capsule_size - 1  # overlap of 1

    # Lookup: frozenset -> pair dict
    pair_lookup = {}
    for p in all_pairs:
        pair_lookup[frozenset([p['_firstId'], p['_secondId']])] = p

    emitted = set()
    ordered = []  # list of pair dicts in emission order

    # Intra-capsule pairs by level
    for capsule in capsules:
        nc = len(capsule)
        for level in range(1, nc):
            for j in range(nc - level):
                e1, e2 = capsule[j], capsule[j + level]
                key = frozenset([e1, e2])
                if key in emitted:
                    continue
                if key in pair_lookup:
                    emitted.add(key)
                    ordered.append(pair_lookup[key])

    # Cross-capsule pairs ordered by token distance
    remaining = []
    for p in all_pairs:
        key = frozenset([p['_firstId'], p['_secondId']])
        if key not in emitted:
            dist = abs(ment_dict[p['_firstId']]['tokens_ids'][0] - ment_dict[p['_secondId']]['tokens_ids'][0])
            remaining.append((dist, p))
    remaining.sort(key=lambda x: x[0])
    for _, p in remaining:
        ordered.append(p)

    return ordered


def get_all_pairs(all_pairs, ment_dict, reduction, sort_by_distance=False, capsule_size=None):
    ret_pairs_dot = list()
    if reduction > 0:
        sample_size = max(1, int(len(all_pairs) * reduction))
        indices_to_remove = random.sample(range(len(all_pairs)), sample_size)
        all_pairs = [all_pairs[i] for i in range(len(all_pairs)) if i not in indices_to_remove]

    if capsule_size is not None:
        sorted_pairs = _capsule_order_pairs(all_pairs, ment_dict, capsule_size)
    else:
        pairs_with_id = []
        for pair in all_pairs:
            new_pair = pair.copy()
            new_pair['index'] = ment_dict[pair['_firstId']]['tokens_ids'][0]
            pairs_with_id.append(new_pair)

        if sort_by_distance:
            sorted_pairs = sorted(pairs_with_id, key=lambda x: abs(
                ment_dict[x['_firstId']]['tokens_ids'][0] - ment_dict[x['_secondId']]['tokens_ids'][0]
            ))
        else:
            sorted_pairs = sorted(pairs_with_id, key=lambda x: x['index'])

    for pair in sorted_pairs:
        m1 = ment_dict[pair['_firstId']]
        m2 = ment_dict[pair['_secondId']]
        first_ment = f"{m1['tokens']}({m1['m_id']})"
        second_ment = f"{m2['tokens']}({m2['m_id']})"
        ret_pairs_dot.append(f"{first_ment} -- {second_ment}")

    return ret_pairs_dot
