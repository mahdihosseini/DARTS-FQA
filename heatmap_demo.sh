#!/bin/bash

python heatmap_creation.py \
--arch DARTS-FQA \
--layers 3 \
--init_channels 20 \
--img 'D:/WSI Images for HeatMap/WSI_1/TCGA-02-0111-01A-01-TS1.89374f94-22f0-4c3d-a890-ee6082c14bd5.svs' \
--genotype focuspath \
--trainset focuspath64 \
--checkpoint_path ./pretrained_models/darts-fqa_focuspath.pt \
--result_path ./heatmap_outputs \
--patch_size 64 \
--stride 32 \
--i_lower 0 \
--i_upper 10000 \
--j_lower 0 \
--j_upper 10000

