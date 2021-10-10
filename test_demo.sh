#!/bin/bash

python test_model.py \
--arch DARTS-FQA \
--layers 3 \
--init_channels 20 \
--genotype focuspath \
--testing_set TCGA \
--testcsv_path ./data/TCGA@Focus/TCGA@Focus.txt \
--testing_path ./data/TCGA@Focus \
--ckpt_path ./pretrained_models/darts-fqa_focuspath.pt

