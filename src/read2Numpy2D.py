import os, struct
from array import array as pyarray
import scipy as sp
from numpy.core.fromnumeric import sort

def getData(digits, dataset = "training", path = "."):
    """this function is modified from http://g.sweyla.com/blog/2012/mnist-numpy/ ,
     which returning 3D array.for our needs 2D array is better"""
    if dataset is "training":
        fname_img = os.path.join('rawdata\\train-images.idx3-ubyte')
        fname_lbl = os.path.join('rawdata\\train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join('rawdata\\t10k-images.idx3-ubyte')
        fname_lbl = os.path.join('rawdata\\t10k-labels.idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    flbl = open(fname_lbl, 'rb')
    _, _ = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    _, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    ind = [ k for k in xrange(size) if lbl[k] in digits ]
    N = len(ind)
    images = sp.zeros((N, rows*cols), dtype=sp.uint8)
    labels = sp.zeros((N, 1), dtype=sp.int8)
    for i in xrange(len(ind)):
        images[i] = sp.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows*cols))
        images[i][0]=1
        """because we want to classify by w*x,the bias b will be absorbed in
         the samples - (1,x1,x2....xn) """ 
        labels[i] = lbl[ind[i]]
    return images, labels
def ReduceSetDimension(reduced,img):
    "given a set of pictures and a relevant pixels's list, returns the reduced set"
    images = sp.zeros((len(img),len(reduced)), dtype=sp.uint8)
    numOfSamples = len(img)
    for s in xrange(numOfSamples):
        images[s]=reduceSampleDimension(img[s],reduced)
    return images  
def reduceSampleDimension(smpl,reduced):
        """given a picture and relevant pixels, return the reduced picture"""
        rsmple = sp.zeros(len(reduced), dtype=sp.uint8)
        for i,r in enumerate(reduced):
            rsmple.itemset(i,smpl[r])
        return rsmple
def getReductionList(colnum,rownum,sizeOfM=28):
    """"given the rows and columns we are intrested on and the size of the Picture,
    and returns the relevant Pixels's numbers. because the 0-place represents b in 
    the classifier (w*x)+b it must be a part of the reduced set"""
    reduced = [0]
    for c in colnum:
        for i in xrange(0,sizeOfM):
            reduced.append(c+i*sizeOfM)
    for r in rownum:
        for i in xrange(0,sizeOfM):
            reduced.append(i+r*sizeOfM)
    return sort(list(set(reduced)))  
def BinaryLabels(labels,binProperty):
    """gets a label list and the property to become labeled as '1', and return a labeled set
    where the BinProperty becomes '1', and everything else is '0'"""
    bLabels = sp.zeros((len(labels),1), dtype=sp.uint8)
    for i in xrange(0,len(labels)):
        if int(labels[i])==binProperty:
            bLabels[i]=1
    return bLabels

