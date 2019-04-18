#!/bin/bash

CUDA_VISIBLE_DEVICES='0' python train_epoch.py -epochs 20  -batch_size 64 -input_size 1344 -output_size 3  -lr 0.00001  -hidden_size_FC 256 -hidden_size_RNN 256 -fc_layers_number 2 -modelsave model/layer3/dnn_shuffle -train_feat_list 'data/feat.shuffle.train' -train_label_list 'data/label.shuffle.train' -logFilename 'log_dnn_shuffle_layers3.txt'
