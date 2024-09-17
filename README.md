## Data Folders
* DOT_format -- My data (EventFull) and MATRES in DOT format
* DOT_format/trans_reduced -- The data after running the reduced transitive closure algorithm
* EventFullTrainExports -- The data in the original format (Splits and all)
* MATRES -- The original MATRES data
* MATRES/in_my_format -- The MATRES data in my format

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

