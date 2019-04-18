#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
    Main training file using Keras.
    Author: Lihui Wang && Shaojun Gao    
    Date: 2019-04-12
''' 
import pdb
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from model import LidModel
import utils
from keras.utils import np_utils
import argparse


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=100,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=1.0,
                   help='in case of gradient explosion')
    p.add_argument('-sequence_length', type=int, default=300,
                   help='sequence length of RNN network.')
    p.add_argument('-input_size', type=int, default=64,
                   help='the size of input feature.')
    p.add_argument('-hidden_size_RNN', type=int, default=256,
                   help='number of the RNN network hidden node.')
    p.add_argument('-hidden_size_FC', type=int, default=256,
                   help='number of the FC network hidden node.')
    p.add_argument('-n_layers_RNN', type=int, default=2,
                   help='number of the RNN layers.')
    p.add_argument('-n_layers_FC', type=int, default=1,
                   help='number of the FC layers.')
    p.add_argument('-hidden_size_list', type=str, default="512 300",
                   help='the list of number of the embedding layers.')
    p.add_argument('-output_size', type=int, default=3,
                   help='number of the output class.')
    p.add_argument('-feat_dir', type=str, default='data/feats/',
                   help='the path of features file.')
    p.add_argument('-train_feat_list', type=str, default='data/feat.train',
                   help='train features list.')
    p.add_argument('-train_label_list', type=str, default='data/label.train',
                   help='train label list.')
    p.add_argument('-dev_feat_list', type=str, default='data/feat.dev',
                   help='valid features list.')
    p.add_argument('-dev_label_list', type=str, default='data/label.dev',
                   help='valid label list.')
    p.add_argument('-test_feat_list', type=str, default='data/feat.test',
                   help='test features list.')
    p.add_argument('-test_label_list', type=str, default='data/label.test',
                   help='test label list.')
    p.add_argument('-model_path', type=str, default='',
                   help='the path of the best model.')
    p.add_argument('-fc_layers_number', type=int, default=5,
                   help='the fc layers number.')
    p.add_argument('-use_RNN', type=bool, default=False,
                   help='use RNN.')
    return p.parse_args()

def main():
    args = parse_arguments()
    lidModel = LidModel(args)
    model = lidModel.bulid_model()
    #pdb.set_trace()
    featlist, labellist = utils.load_data('data/feat.train', 'data/label.train')
    test_featlist, test_labellist = utils.load_data('data/feat.valid', 'data/label.valid')

    if not args.use_RNN:
        feature_train = np.array(featlist).reshape(-1, 64)
        feature_test = np.array(test_featlist).reshape(-1, 64)
        label_tmp = np.array(labellist).reshape(-1)
        label_tmp_test = np.array(test_labellist).reshape(-1)
    else:
        feature_train = np.array(featlist)
        feature_test = np.array(test_featlist)
        label_tmp_test = np.array(test_labellist)
        label_tmp = np.array(labellist)
    label_train = np_utils.to_categorical(label_tmp)
    label_test = np_utils.to_categorical(label_tmp_test)

    his = model.fit(feature_train, label_train, batch_size=args.batch_size, nb_epoch=args.epochs, validation_split=0.1)

    pdb.set_trace()

    loss, acc = model.evaluate(feature_test, label_test)



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
