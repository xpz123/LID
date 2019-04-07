import numpy as np
import os
import sys
import util

import pdb


if __name__ == '__main__':
    rootdir = r'data'
    subdirs = ['train', 'valid']
    langdirs = ['english', 'hindi', 'mandarin']

    langdic = dict()
    langdic['english'] = '0'
    langdic['hindi'] = '1'
    langdic['mandarin'] = '2'

    for subdir in subdirs:
        feat_fw = open(os.path.join(rootdir, 'feat.' + subdir), 'w')
        label_fw = open(os.path.join(rootdir, 'label.' + subdir), 'w')
        for lang in langdirs:
            langpath = os.path.join(rootdir, subdir, lang)
            for r, dirs, files in os.walk(langpath):
                for f in files:
                    try:
                        oriname = lang + '.' + f.split('.')[0]
                        mat = np.load(os.path.join(r, f))
                        cutmats = util.split_mat(mat)
                        for i in range(len(cutmats)):
                            name = oriname + '.' + str(i)
                            feat_fw.write(name + '\n')
                            label_fw.write(name + '  ' + langdic[lang] + '\n')
                            cutmatpath = os.path.join(rootdir, 'feats', name)
                            np.save(cutmatpath, cutmats[i])
                    except:
                        print (f)
                        print (err)


                        
            
            
    
