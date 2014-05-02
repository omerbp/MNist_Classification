import scipy as sp

import read2Numpy2D as rn
import random
import time

signTolabel = lambda x: 0 if x<0 else 1 
labelTosign = lambda x: -1 if x==0 else 1 
"for our needs minus sign will be represented as 0, so the labels can be {0,1}"
class Perceptron():
    def __init__(self,training_data,bin_training_label,T,alpha=1.0,D=0):
        """this is the famously known Perceptron. pleas note that the training label input is BINARY,
        and that the classifier is normalized by L2 norm."""
        numOfSamples = len(training_data)
        numOfFeatures = len(training_data[0])
        self.w = sp.zeros(numOfFeatures, dtype=sp.float64)
        for _ in xrange(T): 
            if str(type(D)).__eq__("<type 'int'>"):   
                j = random.randint(0,numOfSamples-1)
            else:
                j=sp.random.choice(xrange(0,numOfSamples), p=D)
            x = training_data[j] 
            result = sp.dot(self.w, x) 
            if signTolabel(result)==bin_training_label[j]:
                continue
            signOfX = labelTosign(bin_training_label[j])
            self.w += alpha * signOfX*x
        if sp.sum(self.w) != 0 : #l1norm>0 because other the next line is MathError
            self.w =self.w/ sp.sqrt(sp.dot(self.w,self.w)) 
    def GetClassifier(self):
        return self.w
    def Classify(self,x):
        return signTolabel(sp.dot(self.w,x))
    def DotProduct(self,x):
        return sp.dot(self.w,x)

class MultiDigitPerceptron():
    """trains a perceptron for each one of the input digits. uses the Maximum Distance from the
    classifying hyper plane as a tie-breaker."""
    def __init__(self,digits,training_data,training_label,T):
        self.w = [x for x in xrange(0,10)]
        self.digits = digits
        tp = time.time()
        for digit in digits:
            print "    training the "+str(digit)+"-Perceptron"
            bin_training_label = rn.BinaryLabels(training_label,digit)
            d_perceptron = Perceptron(training_data,bin_training_label,T)
            self.w[digit]=d_perceptron
        print "Done Training the Perceptron, total time of "+str(time.time()-tp)
    def Classify(self,x):
        p=[-10 for _ in xrange(0,10)]
        for digit in self.digits:
            p[digit]=(self.w)[digit].DotProduct(x)
        for digit in xrange(0,10):
            if p[digit] == max(p):
                return digit 

def TestClassifier(data,labels,classifier):
    """this function gets the data in array, where every element is K vector of uint8
    labels are a vector of real labels, and the classifier is a class that supports the method Classify """
    error=0.0
    numOfSamples = len(data)
    for i in xrange(0,numOfSamples):
        if classifier.Classify(data[i])!=labels[i]:
            error+=1.0/numOfSamples
    return error
            