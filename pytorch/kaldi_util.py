#Author: Shaojun Gao
#Utils for kaldi

import numpy as np
import os
import tensorflow as tf
import sys

import pdb

def tf_tokaldimat(data, utt, lens, count):
    if len(data.shape) != 3 or data.shape[2] != 62 or len(lens) != data.shape[1]:
        #print ('error when writing ')
        return False
    fw = open(matname, 'a')
    for k in xrange(data.shape[1]):
        fw.write(utt[k + count] + ' [ ' + '\n' )
        for i in xrange(int(lens[k])):
            for j in xrange(data.shape[2]):
                fw.write(str(data[i][k][j]))
                fw.write(' ')
            if i == data.shape[0] - 1:
                fw.write(']')
            fw.write('\n')
    fw.close()
    return True

def mat_convert(data):
    #exchange the 1th colum with the last colum to fit kaldi decoder
    if len(data.shape) != 3 or data.shape[2] != 62:
        print ('mat should be 2-D and output dim should be 62')
        return np.array([])
    res = np.zeros(data.shape)
    res[:, :, 1:] = data[:, :, 0:-1]
    res[:, :, 0] = data[:, :, -1]
    #pdb.set_trace()
    return np.log(res)

#Convert kaldi mat 
def kaldi_tonumpy(matname, expdir):
    lines = open(matname).readlines()
    utts = list()
    dics = dict()
    np_mat = list()
    for l in lines:
        tmp = l.strip()
        if tmp[-1] == '[':
            utts.append(tmp[0:-1])
            dics[utts[-1]] = list()
        elif tmp[-1] == ']':
            dics[utts[-1]].append(tmp[0: -1])
        else:
            dics[utts[-1]].append(tmp)
    for utt in utts:
        tmp = utt.strip()
        tmp = tmp.split('_')
        file_name = expdir + '/' + tmp[1].upper() + '-' + tmp[2].upper()
        fw = open(file_name, 'w')
        for k in dics[utt]:
            fw.write(k + '\n')
        fw.close()
        a = np.loadtxt(file_name)
        os.system('rm -f %s' % (file_name))
        #not elegant but short.
        a = np.transpose(a)
        np.save(file_name, a)
