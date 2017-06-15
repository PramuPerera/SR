#########################################################################
#   ImageDatabase.py
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


class ImageDatabase(object):

    def __init__(self, dname, batchsize =16, resz = 256, nc =3, sz =128, tempname ="input"):
        self.dname = dname
        MXNET_HOME = "../"
        os.system('python %s/tools/im2rec.py --list=1 --exts=".png" --recursive=1 --shuffle=1 --train-ratio=0.8 --test-ratio=0.2 %s %s'%(MXNET_HOME,tempname, dname))
        os.system("python %s/tools/im2rec.py --num-thread=4 --pass-through=1 %s  %s"%(MXNET_HOME,tempname, dname))
        self.data_iter_train = mx.io.ImageRecordIter(
                                               path_imgrec=tempname+"_train.rec", # the target record file
                                               data_shape=(nc, sz, sz), # output data shape. An 227x227 region will be cropped from the original image.
                                               batch_size=batchsize, # number of samples per batch
                                               resize=resz ,
                                                )
        self.data_iter_test = mx.io.ImageRecordIter(
                                       path_imgrec=tempname+"_test.rec", # the target record file
                                       data_shape=(nc, sz, sz), # output data shape. An 227x227 region will be cropped from the original image.
                                       batch_size=batchsize, # number of samples per batch
                                       resize=resz ,
                                        )

        self.data_iter_train.reset()
        self.data_iter_test.reset()

    def reset(self):
        self.data_iter_train.reset()
        self.data_iter_test.reset()

    def getTrainIterator(self):
        return(self.data_iter_train)

    def getTestIterator(self):
        return(self.data_iter_test)

    def getDataBatch(self):
      batch = self.data_iter.next()
      data = batch.data[0]
      return(data)


class TestDatabase(object):

    def __init__(self, dname, batchsize =16, resz = 256, nc =3, sz =128, tempname ="input"):
        self.dname = dname
        MXNET_HOME = "../"
        os.system('python %s/tools/im2rec.py --list=1 --exts=".png" --recursive=1 --shuffle=1  %s %s'%(MXNET_HOME,tempname, dname))
        os.system("python %s/tools/im2rec.py --num-thread=1 --pass-through=1 %s  %s"%(MXNET_HOME,tempname, dname))
        self.data_iter_test = mx.io.ImageRecordIter(
                                       path_imgrec=tempname+".rec", # the target record file
                                       data_shape=(nc, sz, sz), # output data shape. An 227x227 region will be cropped from the original image.
                                       batch_size=batchsize, # number of samples per batch
                                       resize=resz ,
                                        )

        self.data_iter_test.reset()

    def reset(self):
        self.data_iter_test.reset()

    def getTestIterator(self):
        return(self.data_iter_test)
