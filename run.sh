#!/bin/bash

CUDA_VISIBLE_DEVICES='1' python train.py -batch_size 32 -input_size 64 -output_size 3 -dev_feat_list 'data/feat.valid' -dev_label_list 'data/label.valid'  -lr 0.00001 -hidden_size_RNN 128 -hidden_size_FC 64 -hidden_size_list "64" -n_layers_RNN 1
