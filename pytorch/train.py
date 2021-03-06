#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
    Main training file.
    Author: Lihui Wang && Shaojun Gao    
    Date: 2019-03-31
''' 

import os
import math
import argparse
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from model import LID_Frame, LID_Utt, LID
from util import Load_Dataset
import util
import pdb
from collections import Counter
import numpy as np

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
    p.add_argument('-input_size', type=int, default=13,
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
    p.add_argument('-output_size', type=int, default=2,
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
    p.add_argument('-isTest', type=bool, default=False,
                   help='just test.')
    return p.parse_args()

def evaluate(model, val_iter):
    model.eval()
    total_loss = 0
    total_acc = 0.0
    for b, batch in enumerate(val_iter):
        feat = batch['feat'].permute(1,0,2).float()
        label = batch['label']
        with torch.no_grad():
            feat = Variable(feat.data.cuda())
        with torch.no_grad():
            label = Variable(label.data.cuda())

        output = model(feat)
        output = output.reshape(-1, 3)
        #label = torch.transpose(label, 1, 0)
        label = label.reshape(-1).long()

        loss = F.nll_loss(output, label)
        total_acc += calAccuracy(output, label)
        if b == 0:
            np.savetxt('tmp_output.txt', output.data.max(1)[1].cpu())
            np.savetxt('tmp_label.txt', label.data.cpu())
        total_loss += loss.data

    return total_loss / len(val_iter), total_acc / len(val_iter)

def test(model, test_feats, test_labels=None):
    featsfile = open(test_feats).readlines()
    test_hyps = list()
    for featfile in featsfile:
        featfile = featfile.strip()
        feats = np.load(featfile)
        featslist = util.split_mat(feats)
        label_temp = list()
        for shortfeat in featslist:
            model.eval()
            shortfeat_tensor = torch.from_numpy(shortfeat.T[:, np.newaxis, :]).float()
            with torch.no_grad():
                shortfeat_tensor = Variable(shortfeat_tensor.data.cuda())
            output = model(shortfeat_tensor)
            pred = output.data.max(1)[1]
            label_temp.append(pred)
        label_temp_counter = Counter(label_temp)
        hyps = label_temp_counter.most_common(1)[0][0]
        test_hyps.append(hyps)

    if len(test_hyps) != len(featsfile):
        print('length of the test output != length of the test input file.') 
        return
    if test_labels == None:
        f_out = open('test_labels.txt', 'w', encoding = 'utf-8')
        for i in range(len(test_hyps)):
            f_out.write(featsfile[i] + ':/t' + str(test_hyps[i]) + '\n')
        f_out.close()
    else:
        correct = 0
        labels = open(test_labels).readlines()
        if len(test_hyps) != len(labels):
            print('length of yhe test output != length of the labels.')
            return
        for i in range(len(test_hyps)):
            if test_hyps[i] == int(labels[i].strip()):
                correct = correct + 1

        print('Accuracy: ' + str(round(correct / len(test_hyps), 3)))
            

#给定output和label计算帧准确率
def calAccuracy(output, label):
    pred = output.data.max(1)[1]
    right = float((label == pred).sum().cpu())
    return right / float(label.size()[0])


def train(e, model, optimizer, train_loader, val_loader, grad_clip):
    model.train()
    total_loss = 0
    train_iter = train_loader.__iter__()
    pdb.set_trace()
    for b, batch in enumerate(train_iter):
        model.train()
        feat = batch['feat'].permute(1,0,2).float()
        label = batch['label']
        feat, label = feat.cuda(), label.cuda()
       
        optimizer.zero_grad()
        output = model(feat)
        label = label.reshape(-1).long()
        output = output.reshape(-1, 3)
        loss = F.nll_loss(output, label)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.data

        if b % 100 == 0 and b != 0:
            total_loss = total_loss / 100
            print("[%d] [loss: %5.4f]" % (b, total_loss))
            total_loss = 0
            #val_iter = val_loader.__iter__()
            #val_loss = evaluate(model, val_iter)
            #print("[%d] [Val_loss: %5.4f]" %(b, val_loss))


def main():
    args = parse_arguments()
    assert torch.cuda.is_available()

    #print("Preparing dataset......")
    #train_iter, val_iter, test_iter = Load_Dataset(args)
    
    print("Instantiating models......")
    lid_frame = LID_Frame(args.input_size, args.hidden_size_RNN, args.hidden_size_FC,
                    n_layers_RNN=args.n_layers_RNN, n_layers_FC=args.n_layers_FC, dropout=0.5)
    hidden_size_list_utt = args.hidden_size_list.split(' ')
    hidden_size_list_utt_int = list()
    for size in hidden_size_list_utt:
        hidden_size_list_utt_int.append(int(size))
    lid_utt = LID_Utt(args.hidden_size_FC, args.output_size, hidden_size_list_utt_int)
    lid = LID(lid_frame, lid_utt).cuda()
    #optimizer = optim.Adam(lid.parameters(), lr=args.lr)
    optimizer = optim.SGD(lid.parameters(), lr=args.lr)
    print(lid)
    
    if args.isTest:
        if args.model_path == '':
            print('model_path is empty!')
            return
        print("Load the best model: " + args.model_path)
        lid.load_state_dict(torch.load(args.model_path))
        test(lid, r'data/test.list', r'data/test.lab')
        return

    print("Preparing dataset......")

    best_val_loss = None
    for e in range(1, args.epochs+1):
        train_loader, val_loader, test_loader = Load_Dataset(args)
        train(e, lid, optimizer, train_loader, val_loader, args.grad_clip)
        val_loss, val_acc = evaluate(lid, val_loader.__iter__())
        print("[Epoch:%d] val_loss:%5.4f  val_accuracy:%5.4f" % (e, val_loss, val_acc))
        
        with open(r'log/' + 'train.' + str(e) + '.log', 'w') as fw:
            fw.write("[Epoch:%d] val_loss:%5.4f" % (e, val_loss))
            fw.write("\n")

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("Saving model......")
            if not os.path.isdir("save"):
                os.makedirs("save")
            torch.save(lid.state_dict(), './save/lid_%d.pt' % (e))
            best_val_loss = val_loss
    #test_loss = evaluate(lid, test_iter)
    #print("[Test] loss:%5.2f" % test_loss)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
