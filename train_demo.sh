#!/bin/bash

python train_model.py \
--arch DARTS-FQA \
--trainset FocusPath64 \
--traincsv_path ./data/FocusPath/focuspath_train_metadata.csv \
--training_path ./data/FocusPath/Training \
--initial_lr 0.001 \
--layers 3 \
--init_channels 20 \
--genotype focuspath \
--ckpt_path ./checkpoint/focuspath64 \
--board ./board/focuspath64 \
--result_path ./train_results/focuspath64
