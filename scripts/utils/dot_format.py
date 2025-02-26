import re


def parse_dot(dot_json):
    if 'Strict graph' in dot_json:
        dot_json = dot_json.replace('Strict graph', 'strict graph')

    if 'strict graph' not in dot_json and 'Strict graph' not in dot_json:
        print("Invalid DOT file!!!!!")
        return None

    edges = dot_json[dot_json.index('strict graph')+len('strict graph {'):dot_json.rfind('```')].split(';')
    graph = [] # graph edge list
    for edge_str in edges:
        rel_list = re.findall(r'rel\s?=\s?"?([a-zA-Z_]+)"?', edge_str)

        if len(rel_list) < 1:
            break

        rel = rel_list[0].lower()
        if rel.endswith('s') and rel != 'simultaneous':
            rel = rel[:-1]

        if rel not in ['after', 'before', 'equal', 'vague', 'include', 'included', 'is_included', 'same', 'same_time', 'simultaneous', 'precede', 'during']: #['after', 'before']:
            continue

        event_pair = edge_str.split('[rel=')[0]
        if len(event_pair.split('--')) < 2:
            continue

        event_1 = event_pair.split('--')[0].lower().strip()
        event_2 = event_pair.split('--')[1].lower().strip()

        if event_1[0] == ' ':
            event_1 = event_1[1:]

        event_1 = re.sub(r'\"', '', event_1)
        event_2 = re.sub(r'\"', '', event_2)

        if len(event_1) == 0 or len(event_2) == 0:
            continue
        if event_1 == " " or event_2 == " ":
            continue
        if event_1[0] == ' ':
            event_1 = event_1[1:]
        if event_2[0] == ' ':
            event_2 = event_2[1:]
        if event_1[-1] == ' ':
            event_1 = event_1[:-1]
        if event_2[-1] == ' ':
            event_2 = event_2[:-1]

        graph.append((event_1, rel.upper(), event_2))
        # print(event_1, rel, event_2)
    #print(f"Num of duplication: {duplicate}")
    return graph
