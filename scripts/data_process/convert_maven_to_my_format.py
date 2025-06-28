#!/usr/bin/env python3
"""
Convert MAVEN-ERE format to OmniTemp format.

This script converts MAVEN-ERE JSONL files to the OmniTemp format used in the project.
The OmniTemp format has:
- tokens: list of tokenized words
- allMentions: list of event mentions with metadata
- allPairs: list of temporal relations between events
"""

import json
import argparse
import os
from typing import Dict, List, Any, Tuple
from collections import defaultdict


def handle_rel_list(pairs, clusters, rel):
    all_pairs = []
    for pair in pairs:
        if pair[0].startswith("TIME") or pair[1].startswith("TIME"):
            continue

        first_cluster = clusters.get(pair[0])
        second_cluster = clusters.get(pair[1])
        for frst_ment in first_cluster:
            for sec_ment in second_cluster:
                if frst_ment['m_id'] != sec_ment['m_id']:
                    pair = {
                        "_firstId": frst_ment['m_id'],
                        "_secondId": sec_ment['m_id'],
                        "_relation": rel  # Default relation
                    }
                    all_pairs.append(pair)

    return all_pairs


def convert_maven_to_omnitemp(maven_data: Dict[str, Any], mav_key_to_my_key: Dict[str, str]) -> Tuple[str, Dict[str, Any]]:
    """
    Convert a single MAVEN-ERE document to OmniTemp format.
    
    Args:
        maven_data: Dictionary containing MAVEN-ERE format data
        
    Returns:
        Dictionary in OmniTemp format
    """
    # Extract basic information
    doc_id = maven_data.get("id")
    title = maven_data.get("title")
    
    # Flatten tokens from sentences to a single list
    tokens = []
    for sentence_tokens in maven_data.get("tokens"):
        tokens.extend(sentence_tokens)
    
    # Process event mentions
    all_mentions = list()
    all_clusters = dict()
    events = maven_data.get("events")

    for i, event in enumerate(events):
        event_id = event.get("id")
        if event_id not in all_clusters:
            all_clusters[event_id] = []

        mentions = event.get("mention")
        for mention in mentions:
            # Get the trigger word and its position
            trigger_word = mention.get("trigger_word")
            sent_id = mention.get("sent_id")
            offset = mention.get("offset")
            mention_id = mention.get("id")
            if mention_id in mav_key_to_my_key:
                m_id = mav_key_to_my_key[mention_id]
            else:
                if len(mav_key_to_my_key) == 0:
                    m_id = 0
                else:
                    max_value = int(max(mav_key_to_my_key.values()))
                    m_id = max_value + 1

                mav_key_to_my_key[mention_id] = m_id

            # Calculate global token position
            # First, find the position of this sentence in the flattened tokens
            sentence_start = 0
            for j in range(sent_id):
                sentence_start += len(maven_data.get("tokens")[j])

            # Add the offset within the sentence
            global_token_pos_strt = list(range(sentence_start + offset[0], sentence_start + offset[1]))

            mention = {
                "tokens": trigger_word,
                "eventIndex": i,
                "event_id": event_id,
                "m_id": str(m_id),
                "doc_id": doc_id,
                "tokens_ids": global_token_pos_strt,
                "axisType": "main",  # Default to main event
                "rootAxisEventId": -1,
                "corefState": "unknown"
            }

            assert " ".join(tokens[mention['tokens_ids'][0]:mention['tokens_ids'][-1]+1]) == mention['tokens'], f"Token mismatch in mention: {mention['tokens']}"

            all_mentions.append(mention)
            all_clusters[event_id].append(mention)
    
    # Generate all pairs of events (temporal relations)
    # For now, we'll create pairs with "uncertain" relation
    # In a real scenario, you might want to extract actual temporal relations
    mav_tmp_rel = maven_data.get("temporal_relations")
    befores = mav_tmp_rel.get("BEFORE")
    equales = mav_tmp_rel.get("SIMULTANEOUS")

    all_pairs = []
    all_pairs.extend(handle_rel_list(befores, all_clusters, "before"))
    all_pairs.extend(handle_rel_list(equales, all_clusters, "equal"))

    all_doc_ment_ids = [mention['m_id'] for mention in all_mentions]
    all_pairs_ment_ids = [pair['_firstId'] for pair in all_pairs]
    all_pairs_ment_ids.extend([pair['_secondId'] for pair in all_pairs])
    assert set(all_pairs_ment_ids).issubset(set(all_doc_ment_ids)), "Error: Some pairs refer to non-existing mentions"

    # Create the final OmniTemp format
    omnitemp_data = {
        "tokens": tokens,
        "allMentions": all_mentions,
        "allPairs": all_pairs
    }
    
    return title, omnitemp_data


def process_maven_file(input_file: str, output_folder: str):
    """
    Process a MAVEN-ERE JSONL file and convert it to OmniTemp format.
    
    Args:
        input_file: Path to input MAVEN-ERE JSONL file
        output_folder: Path to output OmniTemp JSON files
    """
    mav_key_to_my_key = dict()
    processed_count = 0
    with open(input_file, 'r', encoding="latin1") as f:
        for line_num, line in enumerate(f, 1):
            # Parse JSON line
            maven_data = json.loads(line.strip())

            # Convert to OmniTemp format
            title, omnitemp_data = convert_maven_to_omnitemp(maven_data, mav_key_to_my_key)

            if len(omnitemp_data['allPairs']) > 200:
                print(f"Skipping, data with more then 200 pairs.")
                continue

            # Write the converted data
            with open(f'{output_folder}/{title}.json', 'w', encoding='utf-8') as f:
                json.dump(omnitemp_data, f, indent=2, ensure_ascii=False)

            print(f"Processed document {line_num}: {maven_data.get('id', 'unknown')}")
            processed_count += 1

            if processed_count == 20:
                print("Processed 20 documents, stopping for demonstration purposes.")
                break


def main():
    # parser = argparse.ArgumentParser(description='Convert MAVEN-ERE format to OmniTemp format')
    # parser.add_argument('input_file', help='Path to input MAVEN-ERE JSONL file')
    # parser.add_argument('output_file', help='Path to output OmniTemp JSON file')
    #
    # args = parser.parse_args()

    input_file = 'data/MAVEN-ERE/valid.jsonl'  # Example input file
    output_folder = 'data/MAVEN-ERE/valid_converted'  # Example output file
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return
    
    # Process the file
    process_maven_file(input_file, output_folder)


if __name__ == "__main__":
    main() 