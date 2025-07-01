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

import grapheme


def handle_rel_list(pairs):
    all_pairs = []
    for pair in pairs:
        frst_ment = pair['event_id_1']
        sec_ment = pair['event_id_2']
        rel = pair.get('label').lower()
        if frst_ment != sec_ment:
            pair = {
                "_firstId": str(frst_ment),
                "_secondId": str(sec_ment),
                "_relation": rel
            }
            all_pairs.append(pair)

    return all_pairs


def convert_tcrheb_to_omnitemp(tcrheb_doc: Dict[str, Any]):
    """
    Convert a single MAVEN-ERE document to OmniTemp format.
    
    Args:
        tcrheb_doc: Dictionary containing MAVEN-ERE format data
        
    Returns:
        Dictionary in OmniTemp format
    """
    # Extract basic information
    doc_id = tcrheb_doc.get('document_id')

    # Flatten tokens from sentences to a single list
    pure_text = tcrheb_doc.get('text')
    tokens = pure_text.split(" ")
    
    # Process event mentions
    events = tcrheb_doc.get("events")

    all_mentions = []
    for i, event in enumerate(events):
        m_id = event.get("event_id")
        # Get the trigger word and its position
        trigger_word = event.get("text")
        idx_start = event.get("start")
        idx_end = event.get("end")

        # Calculate global token position
        # First, find the position of this sentence in the flattened tokens
        token_strt_idx = len(grapheme.slice(pure_text, 0, idx_start).split(" ")) - 1
        token_end_idx = len(grapheme.slice(pure_text, 0, idx_end).split(" "))
        # token_strt_idx = len(pure_text[:idx_start].split(" "))
        # token_end_idx = len(pure_text[:idx_end].split(" "))

        # Add the offset within the sentence
        tok_ids = list(range(token_strt_idx, token_end_idx))

        mention = {
            "tokens": trigger_word,
            "eventIndex": i,
            "m_id": str(m_id),
            "doc_id": doc_id,
            "tokens_ids": tok_ids,
            "axisType": "main",  # Default to main event
            "rootAxisEventId": -1,
            "corefState": "unknown"
        }

        assert " ".join(tokens[mention['tokens_ids'][0]:mention['tokens_ids'][-1]+1]) == mention['tokens'], f"Token mismatch in mention: {mention['tokens']}"

        all_mentions.append(mention)
    
    # Generate all pairs of events (temporal relations)
    # For now, we'll create pairs with "uncertain" relation
    # In a real scenario, you might want to extract actual temporal relations
    tcr_rels = tcrheb_doc.get("temporal_relations")

    all_pairs = handle_rel_list(tcr_rels)
    all_doc_ment_ids = [mention['m_id'] for mention in all_mentions]
    all_pairs_ment_ids = [pair['_firstId'] for pair in all_pairs]
    all_pairs_ment_ids.extend([pair['_secondId'] for pair in all_pairs])
    assert set(all_pairs_ment_ids).issubset(set(all_doc_ment_ids)), "Error: Some pairs refer to non-existing mentions"

    # Create the final OmniTemp format
    tcrheb_ret = {
        "tokens": tokens,
        "allMentions": all_mentions,
        "allPairs": all_pairs
    }
    
    return tcrheb_ret


def process_tcrheb_file(input_file: str, output_folder: str):
    """
    Process a MAVEN-ERE JSONL file and convert it to OmniTemp format.
    
    Args:
        input_file: Path to input MAVEN-ERE JSONL file
        output_folder: Path to output OmniTemp JSON files
    """
    processed_count = 0
    with open(input_file, 'r', encoding="utf-8") as f:
        tcrheb_data = json.load(f)
        documents = tcrheb_data['documents']
        for doc_idx, tcrheb_doc in enumerate(documents):
            # Convert to OmniTemp format
            converted_doc = convert_tcrheb_to_omnitemp(tcrheb_doc)

            if len(converted_doc['allPairs']) > 200:
                print(f"Skipping, data with more then 200 pairs.")
                continue

            # Write the converted data
            with open(f'{output_folder}/{str(doc_idx)}.json', 'w', encoding='utf-8') as f:
                json.dump(converted_doc, f, indent=2, ensure_ascii=False)

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

    input_file = 'data/TRC-Heb/TRC_data.json'  # Example input file
    output_folder = 'data/TRC-Heb/converted'  # Example output file
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return
    
    # Process the file
    process_tcrheb_file(input_file, output_folder)


if __name__ == "__main__":
    main() 