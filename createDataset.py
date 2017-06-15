#########################################################################
#   loadData.py
#   Pramuditha Perera
#   pramudi@Amazon.com
#   Created on 06/08/2017
#   This file contains the code to load the image files of a given dataset
#########################################################################
import os
import subprocess
import mxnet as mx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def createDataset(dname):
    
    # Place the installed folder in the MXNET HOME DIR
    MXNET_HOME = '../'
    #os.system('python %s/tools/im2rec.py --list=1 --recursive=1 --shuffle=1 --test-ratio=0.2 data/caltech data/101_ObjectCategories'%MXNET_HOME)


    os.system('python %s/tools/im2rec.py --list=1 --exts=".png" --recursive=1 --shuffle=1 --test-ratio=0.2 input %s'%(MXNET_HOME,dname))
    os.system("python %s/tools/im2rec.py --num-thread=4 --pass-through=1 input  %s"%(MXNET_HOME,dname))

    data_iter = mx.io.ImageRecordIter(
        path_imgrec="input.rec", # the target record file
        data_shape=(3, 227, 227), # output data shape. An 227x227 region will be cropped from the original image.
        batch_size=4, # number of samples per batch
        resize=256 # resize the shorter edge to 256 before cropping
        # ... you can add more augumentation options here. use help(mx.io.ImageRecordIter) to see all possible choices
        )
        
    data_iter.reset()
    return(data_iter)

def getBatch(data_iter):
    batch = data_iter.next()
    data = batch.data[0]
    return(data)

def main():
    print("Creating database files..")
    db = createDataset("SR_training_datasets/data")
    data = getBatch(db)



if  __name__ =='__main__':
    main()
