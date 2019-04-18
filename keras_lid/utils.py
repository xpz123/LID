import numpy as np
import os
import sys
import pdb


def load_data(featfile, labelfile, featdir='data/feats'):
    lines = open(featfile).readlines()
    feats = list()
    labels = list()
    for l in lines:
        feat = np.load(os.path.join(featdir, l.strip() + '.npy'))
        #featcomb = combine_feat_padding(feat)
        #np.save(os.path.join(featdir, l.strip()), featcomb)
        #return
        #feats.append(feat[:, 1:])
        #feats.append(featcomb)
        feats.append(feat)
        label = np.ones(feat.shape[0], dtype=int)
        if l.split('.')[0] == 'english':
            labels.append(label * 0)
        elif l.split('.')[0] == 'hindi':
            labels.append(label)
        else:
            labels.append(label * 2)
    return feats, labels

def combine_feat_padding(feat, left=10, right=10):
    reslist = list()
    featdim = feat.shape[1]
    seqlen = feat.shape[0]
    for i in range(11, seqlen - 11):
        reslist.append(feat[i-11:i+10].reshape(-1))
    return np.array(reslist)
            


#if __name__ == '__main__':
    #load_data('data/feat.train', 'data/label.train')
    #for root, dirs, files in os.walk('data/feats'):
    #    #pdb.set_trace()
    #    for f in files:
    #        feat = np.load(os.path.join(root, f))
    #        featcomb = combine_feat_padding(feat)
    #        np.save(os.path.join(root, f), featcomb)


