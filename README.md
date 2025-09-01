This repository contains the code and data used in the experiments for the paper "[Beyond Pairwise: Global Zero-shot Temporal Graph Generation](https://arxiv.org/abs/2502.11114)".

## Requirements
- Python 3.10
- Install the requirements.txt packages using pip:
- Create `data` and `output` directories in the root directory.
- Create a `.env` file in the root directory and add your OpenAI/Gemini/Together API key/s:

```.dotenv
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
TOGETHER_API_KEY=your_together_api_key
```

## Data 
Dataset we used in our experiments (in the needed format):
1) OmniTemp: https://github.com/AlonEirew/event-relation-resources/tree/main/tre_datasets/OmniTemp
2) MATRES: https://github.com/AlonEirew/event-relation-resources/tree/main/tre_datasets/MATRES/_in_OmniTemp_format
3) TBD: https://github.com/AlonEirew/event-relation-resources/tree/main/tre_datasets/TimeBankDense/_In_OmniTemp_format
4) NT_6: https://github.com/AlonEirew/event-relation-resources/tree/main/tre_datasets/NarrativeTime/_In_OmniTemp_format/exclude_overlap_rel_a1
   - Download the train and test_18ment folders

Download the dataset folders and place them in the `data` directory as follows:
```data/
    OmniTemp/
    MATRES/
    TBD/
    NT-6/
```


## Zero-shot Temporal Graph Generation
```bash
run_zsl_tre.py --test_db omni --instruct timeline --api gpt --model_name  --repeat 1
```
--test_db : omni, matres, tbd, nt
--instruct : timeline, global
--api : gpt, gemini, together
--model_name : model name according to the api used (e.g., gpt-3.5-turbo, gpt-4, gemini-1.5, deepseek-ai/DeepSeek-R1, etc.)
--repeat : number of times to repeat the experiment (for selfConsistency/globalConsistency)


## Experiment Data
* my_data/batch_req -- The files generated for the OpenAI batch request API
* predictions -- All Model predictions, file name indicate the model used and type of experiment


## Scripts
### convert_matres_and_tbd.py
- Run `convert_matres_and_tbd.py` to convert TB to my json format (will create a directory with all the files in my json format).
    - Run 'data/MATRES/orig_files/TimeBank' with `MATRES/timebank.txt` for train
    - Run 'data/MATRES/orig_files/te3-platinum' with `MATRES/platinum.txt` for test

### convert_mydata_to_dot.py
- Run `convert_mydata_to_dot.py` to convert my json format to DOT format graphs files (will create a single file containing all the documents graph in dot format).

### calc_graph_stats.py
- Run `calc_graph_stats.py` to calculate the statistics of the dataset (will output the statistics).

### model/evaluation.py
- Run `model/evaluation.py` to evaluate the model (gold vs. prediction) reuslt state will be outputed similar to the calc_graph_stats script.

### run_gpt_batch.ipynb
- Run `prompting/run_gpt_batch.ipynb` to run GPT-X in batch mode (run step by step to create request, run and generate response).

