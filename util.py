import numpy as np
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class LID_Dataset(Dataset):
	def fix_data(self):
		featset = set()
		labelset = set()
		for feat in self.featlist:
			featset.add(feat)
		for labelutt in self.labeldict.keys():
			labelset.add(labelutt)
		leftset = featset & labelset
		for utt in leftset:



	def __init__(self, featfile, labelfile):
		self.featlist = open(featfile).readlines()
		labellist = open(labelfile).readlines()
		self.labeldict = dict()
		failcount = 0
		for label in lablelist:
			try:
				utt = label.split()[0]
				label = int(label.split()[1])
				labeldict[utt] = label
			except:
				failcount += 1
		print ('Fixing Data')
		fix_data()

		print ('Failcount for load labeldict is: ' + str(failcount))
		print ('Final left data count is: ' + str(len(self.featlist)))

	def __len__(self):
