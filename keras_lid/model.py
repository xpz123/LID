#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
    LID model file using Keras.
    Author: Lihui Wang && Shaojun Gao    
    Date: 2019-04-12
''' 
import pdb
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import SGD, Adam

class LidModel(object):
    def __init__(self, args):
        self.input_size = args.input_size
        self.hidden_size_RNN = args.hidden_size_RNN
        self.hidden_size_FC = args.hidden_size_FC
        self.fc_layers_number = args.fc_layers_number
        self.output_size = args.output_size
        self.sequence_length = args.sequence_length
        self.lr = args.lr
        self.use_RNN = args.use_RNN

    def bulid_model(self):
        model = Sequential()
        if self.use_RNN:
            model.add(LSTM(self.hidden_size_RNN, return_sequences=True, dropout=0.1,
                            input_shape=(self.sequence_length, self.input_size)))
            #model.add(LSTM(self.hidden_size_RNN, return_sequences=True,dropout=0.1,
            #                input_shape=(self.sequence_length, self.hidden_size_RNN)))
            model.add(Dense(units=self.hidden_size_FC))
        else:
            model.add(Dense(units=self.hidden_size_FC, input_dim=self.input_size))
        #model.add(Activation("relu"))
        model.add(Activation("relu"))
        for i in range(1, self.fc_layers_number):
            model.add(Dense(units=self.hidden_size_FC, input_dim=self.hidden_size_FC))
            model.add(Activation("relu"))
        model.add(Dense(units=self.output_size))
        model.add(Activation('softmax'))

        #optim = SGD(lr=self.lr, decay=1e-6, momentum=0.9, nesterov=True)
        optim = SGD(lr=self.lr)
        #optim = Adam(lr=self.lr)
        model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

        return model
