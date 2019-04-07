#!/bin/bash

CUDA_VISIBLE_DEVICES='1' python train.py -input_size 64 -output_size 3 -dev_feat_list 'data/feat.valid' -dev_label_list 'data/label.valid' 
