import json
import os
import google.generativeai as genai

from scripts.prompting_global.run_llms import open_input_file


def calc_examples(dot_folder):
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    gemini_pro_model = genai.GenerativeModel('gemini-1.5-pro')
    dot = open_input_file(dot_folder)
    count_per_example = {}
    for key, value in dot.items():
        tokens_count = gemini_pro_model.count_tokens(value['target']).total_tokens
        count_per_example[key] = tokens_count

    print("average tokens per example: ", sum(count_per_example.values())/len(count_per_example))
    print(count_per_example)


if __name__ == "__main__":
    # calc_examples('data/DOT_format/MATRES_test_dot.json')
    # calc_examples('data/DOT_format/EventFull_test_dot.json')
    calc_examples('data/my_data/predictions/output/experiments/matres/matres_run_together_llama_70b_1pred_1exmples_task_description_v2.json')
