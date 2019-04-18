#!/bin/bash

CUDA_VISIBLE_DEVICES='3' python train.py -batch_size 32 -input_size 40 -output_size 3  -dev_feat_list 'data/feat.valid' -dev_label_list 'data/label.valid'  -lr 0.00001 -hidden_size_RNN 32 -hidden_size_FC 256 -hidden_size_list "128 64" -n_layers_RNN 2
