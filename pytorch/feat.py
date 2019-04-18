#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
    The file of dealing with feature.
    Author: Lihui Wang && Shaojun Gao    
    Date: 2019-03-30
''' 
import os
import sys
from sys import argv
from pydub import AudioSegment
import pdb

def readFile(inputfilename):
    with open(inputfilename, 'r', encoding = 'utf-8') as f_in:
        lines = f_in.readlines()

    return lines

def dealFeat(inputfilename, outputDir):
    totalLines = readFile(inputfilename)

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    count = 0
    for line in totalLines:
        line = line.strip()

        if len(line) == 0:
            continue

        if line.endswith('['):
            inputTempFilename = outputDir + '/' + line.split()[0]
            f_out = open(inputTempFilename, 'w', encoding = 'utf-8')
            count = count + 1
        elif line.endswith(']'):
            line = line.replace(']', '').strip() + '\n'
            f_out.write(line)
            f_out.close()
        else:
            f_out.write(line + '\n')

    print(str(count) + ' files have been writted!')

def load_pcm(pcmfile, framerate=16000, samplewidth=2, channel=1):
   return AudioSegment.from_raw(pcmfile, frame_rate=framerate, sample_width=samplewidth, channels=channel)


def seg_audio(audio, seglen=3000):
    duration = len(audio)
    segs = duration / seglen
    audio_list = list()
    for i in range(segs):
        audio_list.append(audio[i*seglen:(i+1)*seglen])
    return audio_list




if __name__ == '__main__':
    argvLen = len(sys.argv)
    if argvLen != 3:
        print('<Usage>: python feat.py inputfilename outputDirectory.\n')
        exit(1)

    inputfilename = argv[1]
    outputDir = argv[2]
    #pdb.set_trace()
    dealFeat(inputfilename, outputDir)
