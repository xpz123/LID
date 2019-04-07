import numpy as np
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pdb

#Frame shift 300 for numpy, means 3s for wav. 
def split_mat(mat, frameshift=300, initframe=300, endframe=300):
    numframes = mat.shape[1]
    remainderframe = numframes % frameshift
    leftframe = numframes - remainderframe
    if remainderframe >= (0.5 * endframe):
        realendframe = remainderframe
    else:
        realendframe = endframe + remainderframe
    numsplits = (numframes - realendframe - initframe) / frameshift
    return np.split(mat[:,initframe:numframes - realendframe], numsplits, 1)



    

class LID_Dataset(Dataset):
    def fix_data(self):
        featset = set()
        labelset = set()
        for feat in self.featlist:
            featset.add(feat.strip())
        for labelutt in self.labeldict.keys():
            labelset.add(labelutt)
        leftset = featset & labelset
        featlist = list()
        labeldict = dict()
        for utt in leftset:
            featlist.append(utt)
            labeldict[utt] = self.labeldict[utt]
        self.featlist = featlist
        self.labeldict = labeldict



    def __init__(self, featfile, labelfile, featdir):
        self.featdir = featdir
        self.featlist = open(featfile).readlines()
        labellist = open(labelfile).readlines()
        self.labeldict = dict()
        failcount = 0
        for label in labellist:
            try:
                utt = label.split()[0]
                label = int(label.split()[1])
                self.labeldict[utt] = label
            except:
                failcount += 1
        print ('Failcount for load labeldict is: ' + str(failcount))
        print ('Length of feat is: ' + str(len(self.featlist)))
        print ('Length of label is: ' + str(len(self.labeldict.keys())))
        print ('Fixing Data....')
        self.fix_data()
        print ('Final left data count is: ' + str(len(self.featlist)))

    def __len__(self):
        return len(self.featlist)

    def __getitem__(self, index):
        utt = self.featlist[index]
        #feat = np.loadtxt(os.path.join(self.featdir, utt))
        feat = np.load(os.path.join(self.featdir, utt + '.npy'))
        feat = feat.T
        label = self.labeldict[utt]
        res = {'feat':feat, 'label':label}
        return res


def Load_Dataset(args):
    print ('Load training dataset...')
    train_dataset = LID_Dataset(args.train_feat_list, args.train_label_list, args.feat_dir)
    dev_dataset = LID_Dataset(args.dev_feat_list, args.dev_label_list, args.feat_dir)
    test_dataset = LID_Dataset(args.test_feat_list, args.test_label_list, args.feat_dir)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)
    return train_dataloader.__iter__(), dev_dataloader.__iter__(), test_dataloader.__iter__


