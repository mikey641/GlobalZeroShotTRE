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

