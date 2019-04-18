#!/bin/bash

CUDA_VISIBLE_DEVICES='3' python train_epoch.py -epochs 20  -batch_size 32 -input_size 1344 -sequence_length 278 -output_size 3  -lr 0.00001  -hidden_size_FC 256 -hidden_size_RNN 256 -fc_layers_number 1 -modelsave model/layer/rnn_shuffle. -use_RNN True -train_feat_list 'data/feat.shuffle.train' -train_label_list 'data/label.shuffle.train' -logFilename 'log_rnn_shuffle_layer1.txt'
