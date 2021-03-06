#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
    build model的文件
    Author: Lihui Wang && Shaojun Gao
    Date: 2019-03-30
''' 

import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb


class LID_Test(nn.Module):
    def __init__(self, input_size=64):
        super(LID_Test, self).__init__()
        self.blstm = nn.LSTM(input_size, 128, 2, bidirectional=False)
        self.fc1 = nn.Sequential( nn.Linear(128, 256),  nn.ReLU(True))
        self.fc2 = nn.Sequential( nn.Linear(256, 256),  nn.ReLU(True))
        self.fc3 = nn.Sequential( nn.Linear(256, 3))
        self.LogSoftmax = nn.LogSoftmax(dim = 2)
    def forward(self, feature):
        lstm_outputs, lstm_hidden = self.blstm(feature)
        fc1_output = self.fc1(lstm_outputs)
        fc2_output = self.fc2(fc1_output)
        outputs = self.fc3(fc2_output)
        return self.LogSoftmax(outputs)


class LID_Frame(nn.Module):
    def __init__(self, input_size, hidden_size_RNN, hidden_size_FC,
                 n_layers_RNN=2, n_layers_FC=1, dropout=0.3):
        super(LID_Frame, self).__init__()
        self.input_size = input_size
        self.hidden_size_RNN = hidden_size_RNN
        self.hidden_size_FC = hidden_size_FC
        self.blstm = nn.LSTM(input_size, hidden_size_RNN, n_layers_RNN, dropout=dropout, bidirectional=False)
        #self.blstm = nn.LSTM(input_size, hidden_size_RNN, 1, dropout=dropout, bidirectional=False)
        #iif n_layers_RNN > 1:
        #    self.blstm_output = nn.LSTM(hidden_size_RNN, hidden_size_RNN, n_layers_RNN - 1, dropout=dropout, bidirectional=True)
        #else:
        #    self.blstm_output = None
        
        #self.fc = nn.Linear(hidden_size_RNN, hidden_size_FC)
        self.fc = nn.Linear(input_size, hidden_size_FC)
        self.fc_relu = nn.ReLU(hidden_size_FC)
        #self.fc_relu = nn.Tanh()

    def forward(self, feature):
        #lstm_outputs, lstm_hidden = self.blstm(feature)
        # sum bidirectional outputs
        #lstm_outputs = (lstm_outputs[:, :, :self.hidden_size_RNN] +
        #           lstm_outputs[:, :, self.hidden_size_RNN:])
        fc_outputs = self.fc_relu(self.fc(feature))
        #batch_size * hidden_size_FC
        #mean_outputs = torch.mean(fc_outputs, 0)
        #std_outputs = torch.std(fc_outputs, 0)
        #outputs = torch.cat((mean_outputs, std_outputs), 1)
        outputs = fc_outputs
        return outputs


class LID_Utt(nn.Module):
    def __init__(self, input_size, output_size, hidden_size_list):
        super(LID_Utt, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list

        self.fc_input = nn.Linear(input_size, hidden_size_list[0])
        self.fc_input_relu = nn.ReLU(hidden_size_list[0])
        #self.fc_input_relu = nn.Tanh()
        self.fc_list = list()
        if len(hidden_size_list) >= 2:
            for i in range(1, len(hidden_size_list)):
                self.fc_list.append(nn.Linear(hidden_size_list[i - 1], hidden_size_list[i]))
        self.fc_output = nn.Linear(hidden_size_list[-1], output_size)
        self.LogSoftmax = nn.LogSoftmax(dim = 2)

    def forward(self, inputs):
        fc_first_outputs = self.fc_input(inputs)
        mid_fc_outputs = self.fc_input_relu(fc_first_outputs)
        if len(self.fc_list) != 0:
            for fc_layer in self.fc_list:
                fc_layer_cuda = fc_layer.cuda()
                mid_fc_outputs = fc_layer_cuda(mid_fc_outputs)
        final_fc_outputs = self.fc_output(mid_fc_outputs)
        #mean_outputs = torch.mean(final_fc_outputs, dim = 0)
        log_outputs = self.LogSoftmax(final_fc_outputs)
        #mean_outputs = torch.mean(log_outputs, dim = 0)
        return log_outputs

class LID(nn.Module):
    def __init__(self, LID_Frame, LID_Utt):
        super(LID, self).__init__()
        self.LID_Frame = LID_Frame
        self.LID_Utt = LID_Utt

    def forward(self, inputs):
        frame_outputs = self.LID_Frame(inputs)
        #pdb.set_trace()
        outputs = self.LID_Utt(frame_outputs)
        return outputs
