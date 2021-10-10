#!/bin/bash

python train_search.py \
--layers 3 \
--init_channels 8 \
--node 2 \
--save FOCUSPATH_exp \
--dataset FocusPath \
--file_name focuspath \
--csv_path '../FocusPath Training/focuspath_train_metadata_2.csv' \
--dataset_path '../FocusPath Training/Training Patches Database 2' \
--result_path ./search_results/focuspath

